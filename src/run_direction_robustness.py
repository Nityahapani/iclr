"""
Independent-direction robustness (Part 5). Four independently-constructed
phase-A causal directions, frozen at theta_A, tested via ablation on the
SAME M_AB models:

  A: J_A -- gradient of the OUTPUT MARGIN w.r.t. h (used throughout).
  B: gradient of the CLASSIFICATION LOSS (cross-entropy toward 'red').
  C: PAIRED ACTIVATION DIFFERENCE, h(zor,CTX_RED) - h(vex,CTX_BLUE).
  D: LINEAR PROBE fit on a DISJOINT input set (filler objects only).
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (jacobian_zor_red_vs_blue, direction_B_behavioral_loss_grad,
                        direction_C_paired_difference, direction_D_disjoint_inputs,
                        ablate_along_J, cosine_alignment)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def margin(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(obj_id, ctx_id)
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def run_direction_robustness(seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    theta_A_state = copy.deepcopy(model_A.state_dict())

    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    dir_B = direction_B_behavioral_loss_grad(model_A, SPECIAL_OBJECT)
    dir_C = direction_C_paired_difference(model_A, SPECIAL_OBJECT, "vex")
    dir_D = direction_D_disjoint_inputs(model_A, ctx_name="CTX_RED")

    if dir_D is None:
        print(f"[{run_name}] WARNING: could not fit Direction D")

    directions = {"A": J_A, "B": dir_B, "C": dir_C, "D": dir_D}
    print(f"[{run_name}] pairwise cosine alignment at theta_A:")
    names = [n for n, d in directions.items() if d is not None]
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            c = cosine_alignment(directions[n1], directions[n2])
            print(f"    cos({n1},{n2}) = {c:.4f}")

    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=3000)

    m_normal = margin(model_AB, SPECIAL_OBJECT)

    ablation_effects = {}
    for name, direction in directions.items():
        if direction is None:
            continue
        obj_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
        ctx_id = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
        logits_ablated = ablate_along_J(model_AB, obj_id, ctx_id, direction, alpha=1.0)
        m_ablated = (logits_ablated[0, COLOR2ID["blue"]] - logits_ablated[0, COLOR2ID["red"]]).item()
        C_effect = m_ablated - m_normal
        ablation_effects[name] = C_effect
        print(f"[{run_name}] Direction {name}: ablation effect C = {C_effect:.4f} "
              f"(m_normal={m_normal:.3f} -> m_ablated={m_ablated:.3f})")

    result = {"run_name": run_name, "seed": seed, "m_normal": m_normal,
              "ablation_effects": ablation_effects,
              "theta_A_pairwise_cosine": {
                  f"{n1}_{n2}": cosine_alignment(directions[n1], directions[n2])
                  for i, n1 in enumerate(names) for n2 in names[i+1:]
              }}
    with open(f"/home/claude/iclr/results/direction_robustness_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_direction_robustness(s, run_name=f"seed{s}"))
        print()

    print("\n=== SUMMARY: ablation effect (C) per direction, across seeds ===")
    for name in ["A", "B", "C", "D"]:
        vals = [r["ablation_effects"].get(name) for r in results if name in r["ablation_effects"]]
        vals = np.array(vals)
        if len(vals) > 0:
            print(f"Direction {name}: n={len(vals)}, mean={vals.mean():.3f}, std={vals.std(ddof=1):.3f}, "
                  f"n_effect_>1={int((np.abs(vals)>1.0).sum())}/{len(vals)}")

    with open("/home/claude/iclr/results/direction_robustness_summary.json", "w") as f:
        json.dump({name: [r["ablation_effects"].get(name) for r in results if name in r["ablation_effects"]]
                    for name in ["A", "B", "C", "D"]}, f, indent=2)
