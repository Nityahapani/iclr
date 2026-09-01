"""
Matched-perturbation control (Part 3). Addresses: "how do you know I_A isn't
just deleting a convenient large direction?" by constructing I_R matched to
I_A in perturbation norm AND immediate effect size on the target example
(zor), then comparing both interventions' downstream effects on:
  (a) the target behavior (zor, by construction ~matched)
  (b) UNRELATED behavior (filler objects -- should be near-zero for a
      genuine causally-specific I_A, and we check I_R too)
"""
import json
import copy
import numpy as np
import torch
import torch.nn.functional as Fnn

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, FILLER_OBJECTS, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import jacobian_zor_red_vs_blue, matched_random_intervention, ablate_along_J


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def run_matched_control_test(seed: int, checkpoint_step_fraction: float = 1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=0.005)

    rng = np.random.RandomState(seed + 1)
    n_steps = int(3000 * checkpoint_step_fraction)
    for step in range(n_steps):
        objs, ctxs, labels = ds_B.sample_batch(32, rng)
        logits = model_AB(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    match_result = matched_random_intervention(model_AB, zor_id, ctx_red, J_A,
                                                 n_candidates=200, seed=seed + 500)

    v_R = match_result["v_R"]
    perturbation_norm_A = match_result["perturbation_norm_A"]

    filler_ids = torch.tensor([OBJ2ID[o] for o in FILLER_OBJECTS], dtype=torch.long)
    filler_ctx = torch.tensor([CTX2ID["CTX_RED"]] * len(FILLER_OBJECTS), dtype=torch.long)
    filler_labels = torch.tensor([COLOR2ID[filler_mapping[o]] for o in FILLER_OBJECTS], dtype=torch.long)

    with torch.no_grad():
        pre_filler_logits = model_AB(filler_ids, filler_ctx)
        pre_filler_acc = (pre_filler_logits.argmax(-1) == filler_labels).float().mean().item()

    # ablate_along_J is single-example; loop over fillers individually
    def batch_ablate_predict(v):
        preds = []
        for i in range(len(filler_ids)):
            oid = filler_ids[i:i+1]
            cid = filler_ctx[i:i+1]
            logits = ablate_along_J(model_AB, oid, cid, v, alpha=1.0)
            preds.append(logits.argmax(-1).item())
        return torch.tensor(preds)

    preds_IA = batch_ablate_predict(J_A)
    post_filler_acc_IA = (preds_IA == filler_labels).float().mean().item()

    preds_IR = batch_ablate_predict(v_R)
    post_filler_acc_IR = (preds_IR == filler_labels).float().mean().item()

    delta_filler_IA = abs(pre_filler_acc - post_filler_acc_IA)
    delta_filler_IR = abs(pre_filler_acc - post_filler_acc_IR)

    print(f"seed={seed}: perturbation_norm_A={perturbation_norm_A:.4f}, "
          f"C_A={match_result['C_A']:.4f}, C_R_matched={match_result['C_R_matched']:.4f} "
          f"(effect gap={match_result['matched_effect_gap']:.4f})")
    print(f"  filler accuracy: pre={pre_filler_acc:.3f} after I_A={post_filler_acc_IA:.3f} "
          f"(delta={delta_filler_IA:.3f})  after I_R={post_filler_acc_IR:.3f} (delta={delta_filler_IR:.3f})")

    result = {
        "seed": seed,
        "C_A": match_result["C_A"], "C_R_matched": match_result["C_R_matched"],
        "matched_effect_gap": match_result["matched_effect_gap"],
        "perturbation_norm_A": perturbation_norm_A,
        "pre_filler_acc": pre_filler_acc,
        "post_filler_acc_IA": post_filler_acc_IA, "post_filler_acc_IR": post_filler_acc_IR,
        "delta_filler_IA": delta_filler_IA, "delta_filler_IR": delta_filler_IR,
        "specificity_confirmed": bool(delta_filler_IA < 0.1 and delta_filler_IR > delta_filler_IA),
    }
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        results.append(run_matched_control_test(s))

    with open("/home/claude/iclr/results/matched_control_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== SUMMARY ===")
    C_A_vals = [r["C_A"] for r in results]
    C_R_vals = [r["C_R_matched"] for r in results]
    gaps = [r["matched_effect_gap"] for r in results]
    filler_IA = [r["delta_filler_IA"] for r in results]
    filler_IR = [r["delta_filler_IR"] for r in results]
    n_specific = sum(r["specificity_confirmed"] for r in results)

    print(f"|C_A| vs |C_R| matched (should be close by construction): "
          f"mean(C_A)={np.mean(np.abs(C_A_vals)):.3f}, mean(C_R)={np.mean(np.abs(C_R_vals)):.3f}, "
          f"mean effect gap={np.mean(gaps):.4f}")
    print(f"Filler-accuracy disruption: I_A mean delta={np.mean(filler_IA):.4f}, "
          f"I_R mean delta={np.mean(filler_IR):.4f}")
    print(f"Specificity confirmed (I_A leaves fillers alone, I_R doesn't as cleanly): {n_specific}/{len(results)} seeds")
