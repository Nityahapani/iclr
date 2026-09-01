"""
PREREGISTERED final test. Same frozen checkpoints/seeds as before, no new
architecture, no tuning. At each checkpoint (same trajectory generation as
run_gamma_test.py) we now record ALL of: C_A, C_B, C_B|A, C_A|B, and fit two
out-of-sample models:

  Model 1 (additive):     m ~ a1*(C_A + C_B) + b1
  Model 2 (+interaction): m ~ a2*(C_A + C_B) + c2*(C_A * C_B) + b2

Both calibrated on the SAME in-sample (even-index) checkpoints, evaluated
out-of-sample on the SAME held-out (odd-index) checkpoints. We report
Delta_R^2 = R^2(Model 2) - R^2(Model 1) out-of-sample.

Then the decisive collapse comparison:
  ratio_A = C_B|A / C_B   (how much of B's effect survives once A is removed)
  ratio_B = C_A|B / C_A   (how much of A's effect survives once B is removed)
Symmetric collapse: ratio_A ~ ratio_B (both mechanisms lose similar fractions
  of their effect when the other is removed -- shared/interfering substrate,
  no directional preference).
Asymmetric collapse: ratio_A far from ratio_B -- directional gating (one
  mechanism's causal access depends on the other much more than vice versa).
"""
import json
import copy
import numpy as np
import torch
import torch.nn.functional as Fnn

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (jacobian_zor_red_vs_blue, find_B_mechanism_direction, gamma_AB_interaction,
                        calibrated_random_direction, causal_mediation_effect)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def out_of_sample_r2(X_train_cols, y_train, X_test_cols, y_test):
    A_mat_train = np.vstack(X_train_cols + [np.ones_like(y_train)]).T
    coefs, _, _, _ = np.linalg.lstsq(A_mat_train, y_train, rcond=None)
    A_mat_test = np.vstack(X_test_cols + [np.ones_like(y_test)]).T
    y_pred = A_mat_test @ coefs
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return r2, coefs


def run_full_test(config: dict, seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dim = config.get("hidden_dim", 32)
    bottleneck_dim = config.get("bottleneck_dim", None)
    phase_A_steps = config.get("phase_A_steps", 600)
    phase_A_lr = config.get("phase_A_lr", 0.01)
    phase_B_steps = config.get("phase_B_steps", 3000)
    phase_B_lr = config.get("phase_B_lr", 0.005)
    weight_decay = config.get("weight_decay", 0.0)
    batch_size = config.get("batch_size", 32)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    train_phase(model_A, ds_A, steps=phase_A_steps, batch_size=batch_size,
                lr=phase_A_lr, seed=seed, eval_every=phase_A_steps)
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    v_rand_fixed = calibrated_random_direction(J_A, seed=seed + 999)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=phase_B_lr, weight_decay=weight_decay)

    rng = np.random.RandomState(seed + 1)
    eval_every = max(1, phase_B_steps // 300)

    trajectory = []
    for step in range(phase_B_steps):
        objs, ctxs, labels = ds_B.sample_batch(batch_size, rng)
        logits = model_AB(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == phase_B_steps - 1:
            J_B_t = find_B_mechanism_direction(model_AB, ctx_name="CTX_RED")
            gr = gamma_AB_interaction(model_AB, J_A, J_B_t, SPECIAL_OBJECT)

            # random-direction control: same single-ablation methodology as
            # C_A, applied to the fixed matched-norm random direction instead
            zor_id_t = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
            ctx_red_t = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
            with torch.no_grad():
                h_t = model_AB.hidden(zor_id_t, ctx_red_t).squeeze(0)
                logits_normal_t = model_AB.fc2(h_t.unsqueeze(0))
                m_normal_t = (logits_normal_t[0, COLOR2ID["blue"]] - logits_normal_t[0, COLOR2ID["red"]]).item()
                coeff_r = (v_rand_fixed @ h_t) / ((v_rand_fixed @ v_rand_fixed) + 1e-9)
                h_r = h_t - coeff_r * v_rand_fixed
                logits_r = model_AB.fc2(h_r.unsqueeze(0))
                m_r = (logits_r[0, COLOR2ID["blue"]] - logits_r[0, COLOR2ID["red"]]).item()
            C_r_t = m_r - m_normal_t

            trajectory.append({
                "step": step, "m": gr["m_h"],
                "C_A": gr["C_A"], "C_B": gr["C_B"], "C_r": C_r_t,
                "C_B_given_A": gr["C_B_given_A"], "C_A_given_B": gr["C_A_given_B"],
                "Gamma_AB": gr["Gamma_AB"], "Gamma_BA": gr["Gamma_BA"],
            })

    m_arr = np.array([pt["m"] for pt in trajectory])
    C_A_arr = np.array([pt["C_A"] for pt in trajectory])
    C_B_arr = np.array([pt["C_B"] for pt in trajectory])
    sum_arr = C_A_arr + C_B_arr
    prod_arr = C_A_arr * C_B_arr

    idx = np.arange(len(trajectory))
    train_idx = idx[idx % 2 == 0]
    test_idx = idx[idx % 2 == 1]

    r2_additive, coefs_additive = out_of_sample_r2(
        [sum_arr[train_idx]], m_arr[train_idx], [sum_arr[test_idx]], m_arr[test_idx])
    r2_interaction, coefs_interaction = out_of_sample_r2(
        [sum_arr[train_idx], prod_arr[train_idx]], m_arr[train_idx],
        [sum_arr[test_idx], prod_arr[test_idx]], m_arr[test_idx])

    delta_r2 = r2_interaction - r2_additive

    print(f"[{run_name}] Model 1 (additive) out-of-sample R^2 = {r2_additive:.5f}")
    print(f"[{run_name}] Model 2 (additive+interaction) out-of-sample R^2 = {r2_interaction:.5f}")
    print(f"[{run_name}] Delta R^2 (interaction improvement) = {delta_r2:.5f}")
    print(f"[{run_name}] Model 2 interaction coefficient (c2) = {coefs_interaction[1]:.5f}")

    ratio_A_list, ratio_B_list = [], []
    for pt in trajectory:
        if abs(pt["C_B"]) > 1e-6:
            ratio_A_list.append(pt["C_B_given_A"] / pt["C_B"])
        if abs(pt["C_A"]) > 1e-6:
            ratio_B_list.append(pt["C_A_given_B"] / pt["C_A"])
    ratio_A_arr = np.array(ratio_A_list)
    ratio_B_arr = np.array(ratio_B_list)

    print(f"[{run_name}] ratio_A = C_B|A / C_B: mean={ratio_A_arr.mean():.4f} std={ratio_A_arr.std():.4f}")
    print(f"[{run_name}] ratio_B = C_A|B / C_A: mean={ratio_B_arr.mean():.4f} std={ratio_B_arr.std():.4f}")

    n = min(len(ratio_A_arr), len(ratio_B_arr))
    from scipy.stats import wilcoxon
    w_stat, w_p = None, None
    if n >= 5:
        try:
            w_stat, w_p = wilcoxon(ratio_A_arr[:n], ratio_B_arr[:n])
            print(f"[{run_name}] Wilcoxon (ratio_A vs ratio_B, paired by checkpoint): W={w_stat:.2f} p={w_p:.4f}")
        except Exception as e:
            print(f"[{run_name}] Wilcoxon test failed: {e}")

    symmetry_gap = abs(ratio_A_arr.mean() - ratio_B_arr.mean())
    print(f"[{run_name}] symmetry gap |mean(ratio_A) - mean(ratio_B)| = {symmetry_gap:.4f}")
    if symmetry_gap < 0.15:
        verdict = "SYMMETRIC collapse (shared/interfering substrate, no directional gating evidence)"
    else:
        verdict = "ASYMMETRIC collapse (evidence for directional causal gating)"
    print(f"[{run_name}] VERDICT: {verdict}")

    result = {
        "run_name": run_name, "config": config, "seed": seed,
        "r2_additive": float(r2_additive), "r2_interaction": float(r2_interaction),
        "delta_r2": float(delta_r2),
        "interaction_coef": float(coefs_interaction[1]),
        "ratio_A_mean": float(ratio_A_arr.mean()), "ratio_A_std": float(ratio_A_arr.std()),
        "ratio_B_mean": float(ratio_B_arr.mean()), "ratio_B_std": float(ratio_B_arr.std()),
        "symmetry_gap": float(symmetry_gap),
        "wilcoxon_p": float(w_p) if w_p is not None else None,
        "verdict": verdict,
        "trajectory": trajectory,
    }
    with open(f"/home/claude/iclr/results/full_interaction_test_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    BASE = {
        "hidden_dim": 32, "phase_A_steps": 600, "phase_A_lr": 0.01,
        "phase_B_steps": 3000, "phase_B_lr": 0.005, "weight_decay": 0.0, "batch_size": 32,
    }
    print("=" * 60)
    print("Full interaction test: C1 baseline")
    print("=" * 60)
    run_full_test(BASE, seed=1234, run_name="C1_baseline")
