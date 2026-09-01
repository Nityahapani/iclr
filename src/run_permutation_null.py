"""
Permutation null for the LOSO result, replacing the random-direction control.

Rationale: rather than asking whether a single noisy scalar (C_r) predicts
m(t) (a per-timepoint control, contaminated by autocorrelation and prone to
spurious per-seed alignment, as found in the previous run), we ask a
harder, seed-level question: if we scramble WHICH causal trajectory
(C_A+C_B, as a whole time series) gets paired with WHICH behavioral
trajectory (m(t), as a whole time series) across the 7 seeds, does the
same LOSO fitting procedure still achieve R^2=0.98?

This preserves all temporal autocorrelation WITHIN each trajectory (we never
shuffle timepoints), and tests the actual claim: does this SPECIFIC pairing
of causal-to-behavioral trajectory (same seed to same seed) explain far more
variance than assigning trajectories at random across seeds.

Procedure:
1. There are 7 seeds, each with (causal_trajectory, behavioral_trajectory).
2. For each derangement sigma of the 7 seeds, seed s's behavioral
   trajectory is paired with seed sigma(s)'s causal trajectory (sigma(s) !=
   s for all s, i.e. every seed is mismatched).
3. Run the EXACT SAME LOSO procedure on this shuffled pairing: for each held
   out seed s, fit (slope, intercept) on the other 6 (mismatched) pairs,
   predict s's (mismatched) behavioral trajectory from sigma(s)'s causal
   trajectory, compute R^2_s. Average over the 7 folds -> one null LOSO-R^2
   per permutation.
4. Repeat over all 1854 derangements of 7 elements (exact, exhaustive) to
   build a null distribution.
5. Compare the OBSERVED LOSO-R^2 (0.980) against this null distribution.
"""
import json
import itertools
import numpy as np

from src.run_loso_test import load_trajectory, fit_affine, r2

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]


def compute_loso_r2_for_pairing(m_by_seed, X_by_seed, pairing):
    per_fold_r2 = []
    for held_out in SEEDS:
        train_seeds = [s for s in SEEDS if s != held_out]
        X_train = np.concatenate([X_by_seed[pairing[s]] for s in train_seeds])
        y_train = np.concatenate([m_by_seed[s] for s in train_seeds])
        slope, intercept = fit_affine(X_train, y_train)

        X_test = X_by_seed[pairing[held_out]]
        y_test = m_by_seed[held_out]
        y_pred = slope * X_test + intercept
        per_fold_r2.append(r2(y_test, y_pred))
    return np.mean(per_fold_r2)


def run_permutation_null(n_max_permutations=2000, seed_for_rng=0):
    all_data = {s: load_trajectory(s) for s in SEEDS}
    m_by_seed = {s: all_data[s][0] for s in SEEDS}
    X_by_seed = {s: all_data[s][1] + all_data[s][2] for s in SEEDS}

    identity_pairing = {s: s for s in SEEDS}
    observed_loso_r2 = compute_loso_r2_for_pairing(m_by_seed, X_by_seed, identity_pairing)
    print(f"Observed LOSO-R^2 (correct pairing): {observed_loso_r2:.5f}")

    indices = list(range(7))
    derangements = []
    for perm in itertools.permutations(indices):
        if all(perm[i] != i for i in range(7)):
            derangements.append(perm)

    print(f"Total derangements of 7 seeds: {len(derangements)}")
    if len(derangements) > n_max_permutations:
        rng = np.random.RandomState(seed_for_rng)
        chosen_idx = rng.choice(len(derangements), size=n_max_permutations, replace=False)
        derangements = [derangements[i] for i in chosen_idx]
        print(f"Subsampled to {n_max_permutations} derangements for tractability")

    null_r2s = []
    for perm in derangements:
        pairing = {SEEDS[i]: SEEDS[perm[i]] for i in range(7)}
        null_r2 = compute_loso_r2_for_pairing(m_by_seed, X_by_seed, pairing)
        null_r2s.append(null_r2)

    null_r2s = np.array(null_r2s)
    print(f"\nNull distribution (n={len(null_r2s)} derangements):")
    print(f"  mean={null_r2s.mean():.5f} std={null_r2s.std():.5f} median={np.median(null_r2s):.5f}")
    print(f"  min={null_r2s.min():.5f} max={null_r2s.max():.5f}")
    print(f"  95th percentile={np.percentile(null_r2s, 95):.5f}")
    print(f"  99th percentile={np.percentile(null_r2s, 99):.5f}")

    p_value = np.mean(null_r2s >= observed_loso_r2)
    print(f"\nObserved LOSO-R^2 = {observed_loso_r2:.5f}")
    print(f"Permutation p-value (fraction of null >= observed) = {p_value:.6f} "
          f"({int(p_value*len(null_r2s))}/{len(null_r2s)} derangements matched or exceeded observed)")

    out = {
        "observed_loso_r2": float(observed_loso_r2),
        "n_derangements": len(null_r2s),
        "null_mean": float(null_r2s.mean()), "null_std": float(null_r2s.std()),
        "null_median": float(np.median(null_r2s)),
        "null_min": float(null_r2s.min()), "null_max": float(null_r2s.max()),
        "null_95th_pct": float(np.percentile(null_r2s, 95)),
        "null_99th_pct": float(np.percentile(null_r2s, 99)),
        "p_value": float(p_value),
        "null_r2_distribution": null_r2s.tolist(),
    }
    with open("/home/claude/iclr/results/permutation_null.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run_permutation_null()
