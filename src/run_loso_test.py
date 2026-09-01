"""
Final validation, C1 only (per reviewer decision to drop C3 from the
headline claim -- C3 remains documented as a separate, unstable finding).

TEST A: Leave-one-seed-out (LOSO) generalization.
For each seed s in {1234..1240}: fit (slope, intercept) on the pooled
checkpoints from the OTHER 6 seeds, freeze, predict seed s's ENTIRE
trajectory (all of it, not a held-out half -- s was never used in fitting),
report R^2_s. This tests whether the additive relationship generalizes
across models, not just across checkpoints within one model.

TEST B: Negative control. Predict m(t) from C_r(t) (matched random-direction
intervention) instead of C_A(t)+C_B(t), same in-sample/out-of-sample
checkpoint split used throughout this project. Expect R^2 ~ 0.
"""
import json
import numpy as np

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]


def load_trajectory(seed):
    d = json.load(open(f"/home/claude/iclr/results/full_interaction_test_C1_seed{seed}.json"))
    traj = d["trajectory"]
    m = np.array([pt["m"] for pt in traj])
    C_A = np.array([pt["C_A"] for pt in traj])
    C_B = np.array([pt["C_B"] for pt in traj])
    C_r = np.array([pt["C_r"] for pt in traj])
    return m, C_A, C_B, C_r


def fit_affine(X, y):
    A_mat = np.vstack([X, np.ones_like(X)]).T
    coefs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    return coefs


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def test_A_loso():
    print("=" * 60)
    print("TEST A: Leave-one-seed-out generalization")
    print("=" * 60)

    all_data = {s: load_trajectory(s) for s in SEEDS}
    loso_r2s = {}

    for held_out in SEEDS:
        train_seeds = [s for s in SEEDS if s != held_out]
        X_train = np.concatenate([all_data[s][1] + all_data[s][2] for s in train_seeds])
        y_train = np.concatenate([all_data[s][0] for s in train_seeds])

        slope, intercept = fit_affine(X_train, y_train)

        m_test, C_A_test, C_B_test, _ = all_data[held_out]
        X_test = C_A_test + C_B_test
        y_pred = slope * X_test + intercept
        r2_s = r2(m_test, y_pred)
        loso_r2s[held_out] = r2_s
        print(f"  held-out seed {held_out}: fit on other {len(train_seeds)} seeds "
              f"(slope={slope:.4f}, intercept={intercept:.4f}) -> R^2_s = {r2_s:.5f}")

    r2_vals = np.array(list(loso_r2s.values()))
    print(f"\nLOSO-R^2 across {len(SEEDS)} held-out seeds: mean={r2_vals.mean():.5f} "
          f"std={r2_vals.std():.5f} min={r2_vals.min():.5f} max={r2_vals.max():.5f}")

    return loso_r2s


def test_B_negative_control():
    print("\n" + "=" * 60)
    print("TEST B: negative control (predict m(t) from C_r(t), random direction)")
    print("=" * 60)

    r2s = {}
    for seed in SEEDS:
        m, C_A, C_B, C_r = load_trajectory(seed)
        idx = np.arange(len(m))
        train_idx = idx[idx % 2 == 0]
        test_idx = idx[idx % 2 == 1]

        slope, intercept = fit_affine(C_r[train_idx], m[train_idx])
        y_pred = slope * C_r[test_idx] + intercept
        r2_s = r2(m[test_idx], y_pred)
        r2s[seed] = r2_s
        print(f"  seed {seed}: R^2 (m ~ C_r) out-of-sample = {r2_s:.5f}")

    r2_vals = np.array(list(r2s.values()))
    print(f"\nNegative control R^2 across {len(SEEDS)} seeds: mean={r2_vals.mean():.5f} "
          f"std={r2_vals.std():.5f}")
    return r2s


if __name__ == "__main__":
    loso_results = test_A_loso()
    control_results = test_B_negative_control()

    out = {
        "loso_r2_per_seed": loso_results,
        "loso_r2_mean": float(np.mean(list(loso_results.values()))),
        "loso_r2_std": float(np.std(list(loso_results.values()))),
        "negative_control_r2_per_seed": control_results,
        "negative_control_r2_mean": float(np.mean(list(control_results.values()))),
        "negative_control_r2_std": float(np.std(list(control_results.values()))),
    }
    with open("/home/claude/iclr/results/loso_and_negative_control.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n\n=== FINAL COMPARISON ===")
    print(f"Causal directions (C_A + C_B) -> LOSO R^2 = {out['loso_r2_mean']:.4f} +/- {out['loso_r2_std']:.4f}")
    print(f"Random direction (C_r)        -> R^2 = {out['negative_control_r2_mean']:.4f} +/- {out['negative_control_r2_std']:.4f}")
