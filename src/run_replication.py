"""
Multi-seed replication of the candidate Hypothesis-A regime found in the
sweep (weight_decay=0.12, otherwise baseline config). Also replicates the
baseline (wd=0.0) config across the same seeds as a paired comparison, so we
get a real paired-difference estimate rather than two separate single points.
"""
import json
import numpy as np
from src.sweep_core import run_single_config

CANDIDATE_CONFIG = {
    "hidden_dim": 32, "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005,
    "weight_decay": 0.12, "batch_size": 32,
}
BASELINE_CONFIG = {**CANDIDATE_CONFIG, "weight_decay": 0.0}

SEEDS = list(range(1234, 1234 + 20))  # 20 seeds


def run_replication():
    results = {"candidate": [], "baseline": []}

    for seed in SEEDS:
        print(f"\n--- seed {seed}: candidate (wd=0.12) ---")
        r_cand = run_single_config(CANDIDATE_CONFIG, seed=seed, verbose=True)
        results["candidate"].append(r_cand)

        print(f"--- seed {seed}: baseline (wd=0.0) ---")
        r_base = run_single_config(BASELINE_CONFIG, seed=seed, verbose=True)
        results["baseline"].append(r_base)

    with open("/home/claude/iclr/results/replication_wd012.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summarize
    def summarize(rs, label):
        ok = [r for r in rs if r.get("status") == "OK"]
        n_ok = len(ok)
        n_total = len(rs)
        if n_ok == 0:
            print(f"{label}: 0/{n_total} succeeded")
            return None
        fracs = [r["fraction_mediation_remaining"] for r in ok]
        gaps = [r["rho_A_gap"] for r in ok]
        hyps = [r["hypothesis"] for r in ok]
        n_fossil = sum(1 for h in hyps if h == "A_fossil")
        n_latent = sum(1 for h in hyps if h == "B_latent_persistence")
        n_ambig = sum(1 for h in hyps if h == "AMBIGUOUS")
        print(f"{label}: {n_ok}/{n_total} succeeded (erasure worked)")
        print(f"  fraction_mediation_remaining: mean={np.mean(fracs):.3f} std={np.std(fracs):.3f} "
              f"min={np.min(fracs):.3f} max={np.max(fracs):.3f}")
        print(f"  rho_A_gap: mean={np.mean(gaps):.3f} std={np.std(gaps):.3f}")
        print(f"  hypothesis counts: A_fossil={n_fossil} B_latent={n_latent} AMBIGUOUS={n_ambig}")
        return {"n_ok": n_ok, "n_total": n_total, "fracs": fracs, "gaps": gaps,
                "n_fossil": n_fossil, "n_latent": n_latent, "n_ambig": n_ambig}

    print("\n\n=== REPLICATION SUMMARY ===")
    cand_summary = summarize(results["candidate"], "CANDIDATE (wd=0.12)")
    base_summary = summarize(results["baseline"], "BASELINE (wd=0.0)")

    # Paired difference on seeds where BOTH succeeded
    paired_diffs = []
    for r_c, r_b in zip(results["candidate"], results["baseline"]):
        if r_c.get("status") == "OK" and r_b.get("status") == "OK":
            paired_diffs.append(r_c["fraction_mediation_remaining"] - r_b["fraction_mediation_remaining"])
    if paired_diffs:
        print(f"\nPaired diff (candidate - baseline) on fraction_remaining, n={len(paired_diffs)}:")
        print(f"  mean={np.mean(paired_diffs):.3f} std={np.std(paired_diffs):.3f}")
        if len(paired_diffs) >= 2:
            from scipy import stats as scipy_stats
            try:
                t_stat, p_val = scipy_stats.ttest_1samp(paired_diffs, 0)
                print(f"  paired t-test vs 0: t={t_stat:.3f} p={p_val:.5f}")
            except ImportError:
                print("  (scipy not available for t-test)")

    summary_out = {
        "candidate_summary": cand_summary,
        "baseline_summary": base_summary,
        "paired_diffs": paired_diffs,
    }
    with open("/home/claude/iclr/results/replication_wd012_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, default=str)


if __name__ == "__main__":
    run_replication()
