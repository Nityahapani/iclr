"""
Experiment: Hidden-Dimension / Model-Capacity Sweep

The most interesting unresolved question: what happens when the network has
enough capacity to simply PRESERVE the old solution separately rather than
reuse/repurpose it?

We sweep hidden_dim in {8, 16, 32, 64, 128, 256} across seeds {1234..1240},
measuring:
  1. Probability of behavioral forgetting  (does B-training actually erase A behavior?)
  2. Historical causal accessibility       (frac_remaining = |delta_A(T)| / |delta_A(A)|)
  3. Lineage specificity                   (own-vs-foreign theta_A patching gap)
  4. Degree of interference                (Gamma_AB: conditional interaction between A and B mechanisms)

Key question: does increasing capacity REDUCE interference and eventually allow
the network to store old+new solutions in non-overlapping subspaces?
If so, frac_remaining should STAY HIGH (latent persistence), but Gamma_AB
should DROP (no interference because mechanisms don't share substrate).
If capacity doesn't help (mechanisms always overlap), Gamma_AB stays high regardless.
"""

import copy
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/iclr")

from src.task import (
    make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
    SPECIAL_OBJECT, SHAM_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
    FILLER_OBJECTS
)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import (
    jacobian_zor_red_vs_blue, cosine_alignment, causal_mediation_effect,
    activation_patch_from_theta_A, gamma_AB_interaction, find_B_mechanism_direction,
    ablate_along_J
)


SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
HIDDEN_DIMS = [8, 16, 32, 64, 128, 256]

BASE_CONFIG = {
    "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005,
    "weight_decay": 0.0, "batch_size": 32,
}


def zor_behavior(model):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_red)
        pred = logits.argmax(-1).item()
        margin = (logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]).item()
    return pred == COLOR2ID["red"], margin


def run_one(hidden_dim, seed, verbose=False):
    cfg = {**BASE_CONFIG, "hidden_dim": hidden_dim}

    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    def new_model():
        return TinyClassifier(
            VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
            embed_dim=cfg["embed_dim"], ctx_embed_dim=cfg["ctx_embed_dim"],
            hidden_dim=hidden_dim
        )

    # ---- Phase A ----
    model_A = new_model()
    log_A = train_phase(model_A, ds_A, steps=cfg["phase_A_steps"], batch_size=cfg["batch_size"],
                        lr=cfg["phase_A_lr"], seed=seed, eval_every=cfg["phase_A_steps"])
    is_red_A, margin_A = zor_behavior(model_A)
    if not is_red_A:
        return {"hidden_dim": hidden_dim, "seed": seed, "status": "FAILED_PHASE_A"}

    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    theta_A_state = copy.deepcopy(model_A.state_dict())
    theta_A_embed = model_A.embed.weight[OBJ2ID[SPECIAL_OBJECT]].detach().clone()

    # ---- Phase B (treatment) ----
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))

    # Adam with weight_decay
    opt_AB = torch.optim.Adam(model_AB.parameters(), lr=cfg["phase_B_lr"],
                               weight_decay=cfg["weight_decay"])
    log_AB = _train_loop(model_AB, ds_B, opt_AB, cfg["phase_B_steps"],
                          cfg["batch_size"], seed, eval_every=50)

    # ---- B-only control (matched distribution) ----
    model_B = new_model()
    log_B = train_phase(model_B, ds_B_only, steps=cfg["phase_B_steps"] * 2,
                         batch_size=cfg["batch_size"], lr=cfg["phase_B_lr"],
                         seed=seed, eval_every=50)

    # KL-match B_only to AB's final distribution
    matched_entry, matched_kl = find_matched_checkpoint(log_AB, log_B)
    model_B.load_state_dict(matched_entry["state_dict"])

    # Check behavioral forgetting
    is_red_T_AB, margin_T_AB = zor_behavior(model_AB)
    is_red_T_B, margin_T_B = zor_behavior(model_B)
    behavioral_forgetting = not is_red_T_AB  # True = behavior was overwritten

    # ---- Historical Causal Accessibility (frac_remaining) ----
    med_T = causal_mediation_effect(model_AB, J_A, SPECIAL_OBJECT, "red", "blue", alpha=1.0)
    med_ref = causal_mediation_effect(model_A, J_A, SPECIAL_OBJECT, "red", "blue", alpha=1.0)
    frac_remaining = abs(med_T["delta_A"]) / (abs(med_ref["delta_A"]) + 1e-9)

    # ---- Lineage Specificity ----
    # Compare patching own theta_A vs patching a foreign theta_A
    # Foreign = use seed+1 mod 7 as foreign lineage
    foreign_seed = SEEDS[(SEEDS.index(seed) + 1) % len(SEEDS)]
    torch.manual_seed(foreign_seed)
    np.random.seed(foreign_seed)
    filler_foreign = make_filler_mapping(seed=foreign_seed)
    ds_A_foreign = PhaseDataset(filler_foreign, phase="A")
    model_A_foreign = new_model()
    log_Af = train_phase(model_A_foreign, ds_A_foreign, steps=cfg["phase_A_steps"],
                          batch_size=cfg["batch_size"], lr=cfg["phase_A_lr"],
                          seed=foreign_seed, eval_every=cfg["phase_A_steps"])
    is_red_Af, _ = zor_behavior(model_A_foreign)

    if is_red_Af:
        own_patch = activation_patch_from_theta_A(model_AB, model_A, SPECIAL_OBJECT)
        foreign_patch = activation_patch_from_theta_A(model_AB, model_A_foreign, SPECIAL_OBJECT)
        lineage_specificity = own_patch["m_patched"] - foreign_patch["m_patched"]  # negative = own is more red
        own_restores_red = own_patch["patch_restores_red"]
        foreign_restores_red = foreign_patch["patch_restores_red"]
    else:
        lineage_specificity = None
        own_restores_red = None
        foreign_restores_red = None

    # ---- Degree of Interference (Gamma_AB) ----
    J_B = find_B_mechanism_direction(model_AB, ctx_name="CTX_RED")
    gamma_result = gamma_AB_interaction(model_AB, J_A, J_B, SPECIAL_OBJECT)

    # Also measure subspace overlap between J_A and J_B
    cos_JA_JB = cosine_alignment(J_A, J_B)

    result = {
        "hidden_dim": hidden_dim,
        "seed": seed,
        "status": "OK",
        "behavioral_forgetting": behavioral_forgetting,
        "margin_T_AB": margin_T_AB,
        "frac_remaining": frac_remaining,
        "delta_A_at_T": med_T["delta_A"],
        "delta_A_reference": med_ref["delta_A"],
        "lineage_specificity_gap": lineage_specificity,
        "own_restores_red": own_restores_red,
        "foreign_restores_red": foreign_restores_red,
        "Gamma_AB": gamma_result["Gamma_AB"],
        "Gamma_BA": gamma_result["Gamma_BA"],
        "C_A": gamma_result["C_A"],
        "C_B": gamma_result["C_B"],
        "cos_JA_JB": cos_JA_JB,
        "matched_kl": matched_kl,
    }
    if verbose:
        ls_str = f"{lineage_specificity:.3f}" if lineage_specificity is not None else "N/A"
        print(f"  hidden={hidden_dim} seed={seed}: "
              f"forgot={behavioral_forgetting} frac={frac_remaining:.3f} "
              f"Gamma={gamma_result['Gamma_AB']:.3f} cos(JA,JB)={cos_JA_JB:.3f} "
              f"lineage_gap={ls_str}")
    return result


def _train_loop(model, dataset, opt, steps, batch_size, seed, eval_every=50):
    rng = np.random.RandomState(seed)
    log = []
    eval_set = dataset.full_eval_set()
    eval_objs, eval_ctxs, eval_labels = eval_set
    for step in range(steps):
        objs, ctxs, labels = dataset.sample_batch(batch_size, rng)
        logits = model(objs, ctxs)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                eval_logits = model(eval_objs, eval_ctxs)
                eval_probs = F.softmax(eval_logits, dim=-1)
                acc = (eval_logits.argmax(-1) == eval_labels).float().mean().item()
            log.append({"step": step, "train_loss": loss.item(), "eval_acc": acc,
                         "eval_probs": eval_probs.clone(), "state_dict": copy.deepcopy(model.state_dict())})
    return log


if __name__ == "__main__":
    results = []
    total = len(HIDDEN_DIMS) * len(SEEDS)
    done = 0

    print(f"Capacity sweep: {len(HIDDEN_DIMS)} hidden dims × {len(SEEDS)} seeds = {total} runs")
    for hidden_dim in HIDDEN_DIMS:
        print(f"\n=== hidden_dim={hidden_dim} ===")
        for seed in SEEDS:
            r = run_one(hidden_dim, seed, verbose=True)
            results.append(r)
            done += 1
            print(f"  [{done}/{total}] done")

    outpath = "/home/claude/iclr/results/capacity_sweep.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Quick summary per hidden_dim
    print("\n=== SUMMARY ===")
    print(f"{'hidden_dim':>10} | {'P(forget)':>9} | {'frac_remain':>11} | {'|Gamma_AB|':>10} | {'cos(JA,JB)':>10} | {'lineage_gap':>11}")
    print("-" * 75)
    for hd in HIDDEN_DIMS:
        subset = [r for r in results if r.get("hidden_dim") == hd and r.get("status") == "OK"]
        if not subset:
            continue
        p_forget = np.mean([r["behavioral_forgetting"] for r in subset])
        frac = np.mean([r["frac_remaining"] for r in subset])
        gamma = np.mean([abs(r["Gamma_AB"]) for r in subset])
        cos_jj = np.mean([r["cos_JA_JB"] for r in subset])
        ls = [r["lineage_specificity_gap"] for r in subset if r.get("lineage_specificity_gap") is not None]
        ls_mean = np.mean(ls) if ls else float("nan")
        print(f"{hd:>10} | {p_forget:>9.3f} | {frac:>11.3f} | {gamma:>10.3f} | {cos_jj:>10.3f} | {ls_mean:>11.3f}")
