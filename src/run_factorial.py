"""
Preregistered factorial experiment (frozen after the exploratory sweep).

RQ: Does the joint (rho_A, fraction_mediation_remaining) relationship shift
systematically across capacity/regularization conditions, revealing whether
new learning masks an old mechanism (Hypothesis B, latent persistence) vs.
actually rewrites/erases it (Hypothesis A, fossil) -- and whether this is a
continuum rather than a discrete regime?

Conditions (architecture and hyperparameters FROZEN before running seeds --
no further tuning based on results):
  C1 "wide_no_decay"    : hidden_dim=32, bottleneck=None, weight_decay=0.0
                           (baseline; establishes latent persistence)
  C2 "sparse_no_decay"  : bottleneck_dim=2, weight_decay=0.0
                           (tests capacity scarcity alone)
  C3 "sparse_decay"     : bottleneck_dim=2, weight_decay=0.1, phase_B_steps=8000
                           (strongest plausible overwrite condition found
                           during exploration -- used as-is, NOT re-tuned)

n=20 seeds per condition. NO thresholding into fossil/non-fossil categories.
We report the raw (rho_A, frac_remaining) pairs and their distributions per
condition, plus rank correlation between rho_A and frac_remaining within
each condition (does structure-function coupling change across conditions?).
"""
import json
import numpy as np
from src.sweep_core import run_single_config

BASE = {
    "hidden_dim": 32, "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005,
    "weight_decay": 0.0, "batch_size": 32,
}

CONDITIONS = {
    "C1_wide_no_decay": {**BASE},
    "C2_sparse_no_decay": {**BASE, "bottleneck_dim": 2},
    "C3_sparse_decay": {**BASE, "bottleneck_dim": 2, "weight_decay": 0.1, "phase_B_steps": 8000},
}

SEEDS = list(range(1234, 1234 + 20))


def run_factorial():
    all_results = {name: [] for name in CONDITIONS}

    for cond_name, cfg in CONDITIONS.items():
        print(f"\n=== Condition: {cond_name} ===  config={cfg}")
        for seed in SEEDS:
            r = run_single_config(cfg, seed=seed)
            all_results[cond_name].append(r)
            status = r.get("status")
            if status == "OK":
                print(f"  seed={seed}: rho_A_AB={r['rho_A_AB']:.4f} frac_rem={r['fraction_mediation_remaining']:.4f}")
            else:
                print(f"  seed={seed}: {status}")

    with open("/home/claude/iclr/results/factorial_v1.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\n=== FACTORIAL SUMMARY (no thresholding, raw distributions) ===")
    summary = {}
    for cond_name, results in all_results.items():
        ok = [r for r in results if r.get("status") == "OK"]
        n_ok, n_total = len(ok), len(results)
        if n_ok == 0:
            print(f"{cond_name}: 0/{n_total} succeeded")
            summary[cond_name] = {"n_ok": 0, "n_total": n_total}
            continue
        rhos = np.array([r["rho_A_AB"] for r in ok])
        fracs = np.array([r["fraction_mediation_remaining"] for r in ok])
        rho_B = np.array([r["rho_A_B"] for r in ok])  # B-only control's alignment, same seeds

        from scipy.stats import spearmanr
        rho_corr, rho_p = spearmanr(rhos, fracs)

        print(f"\n{cond_name}: {n_ok}/{n_total} succeeded")
        print(f"  rho_A_AB:      mean={rhos.mean():.4f} std={rhos.std():.4f} range=[{rhos.min():.4f}, {rhos.max():.4f}]")
        print(f"  rho_A_B (ctrl):mean={rho_B.mean():.4f} std={rho_B.std():.4f}")
        print(f"  frac_remaining:mean={fracs.mean():.4f} std={fracs.std():.4f} range=[{fracs.min():.4f}, {fracs.max():.4f}]")
        print(f"  Spearman corr(rho_A_AB, frac_remaining) within condition: rho={rho_corr:.3f} p={rho_p:.4f}")

        summary[cond_name] = {
            "n_ok": n_ok, "n_total": n_total,
            "rho_A_AB_mean": float(rhos.mean()), "rho_A_AB_std": float(rhos.std()),
            "rho_A_AB_min": float(rhos.min()), "rho_A_AB_max": float(rhos.max()),
            "rho_A_B_control_mean": float(rho_B.mean()),
            "frac_remaining_mean": float(fracs.mean()), "frac_remaining_std": float(fracs.std()),
            "frac_remaining_min": float(fracs.min()), "frac_remaining_max": float(fracs.max()),
            "spearman_rho_vs_frac": float(rho_corr), "spearman_p": float(rho_p),
            "raw_pairs": [{"seed": r["seed"], "rho_A_AB": r["rho_A_AB"],
                           "frac_remaining": r["fraction_mediation_remaining"]} for r in ok],
        }

    with open("/home/claude/iclr/results/factorial_v1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    run_factorial()
