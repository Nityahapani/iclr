"""
Test 1: additive causal decomposition of behavioral updating.
Test 3: E(t) superposition test (generalizing I_AB=0 across the whole trajectory).

Both use J_A (frozen at theta_A) and J_B (defined per-checkpoint as the
model's OWN current blue-vs-red causal gradient -- NOT frozen, since B's
mechanism is itself evolving during training). At each checkpoint we
independently measure C_A(t) and C_B(t) via single-ablation interventions
(same methodology as the decisive causal trajectory experiment), then:

TEST 1 (additive model): calibrate a single scale+offset (a, b) via linear
regression on an IN-SAMPLE half of checkpoints (fit ONCE, frozen), predicting
  m(t) - m(0) ~ a * (C_A(t) + C_B(t)) + b
then evaluate R^2 OUT-OF-SAMPLE on the held-out other half of checkpoints.
This is a real train/test split on checkpoints, not a flexible fit to the
whole trajectory.

TEST 3 (superposition): E(t) = C_{A+B}(t) - C_A(t) - C_B(t), computed at
every checkpoint (extending the single final-checkpoint I_AB test across the
whole trajectory). Prediction: E(t) ~ 0 throughout, not just at the end.
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
from src.probe import (jacobian_zor_red_vs_blue, find_B_mechanism_direction,
                        C_A_and_C_B_at_checkpoint, double_intervention_margin)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def zor_margin(model, ctx_name="CTX_RED"):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_id)
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def run_additive_and_superposition_tests(config: dict, seed: int, run_name: str):
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
    m_0 = zor_margin(model_A)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=phase_B_lr, weight_decay=weight_decay)

    rng = np.random.RandomState(seed + 1)
    eval_every = max(1, phase_B_steps // 300)

    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

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
            m_t, C_A_t, C_B_t = C_A_and_C_B_at_checkpoint(model_AB, J_A, J_B_t, SPECIAL_OBJECT)

            rb_margin_AB = double_intervention_margin(model_AB, zor_id, ctx_red, J_A, J_B_t,
                                                        alpha=1.0, do_A=True, do_B=True)
            m_AB_combined = -rb_margin_AB
            C_AB_t = m_AB_combined - m_t
            E_t = C_AB_t - C_A_t - C_B_t

            trajectory.append({
                "step": step, "m_t": m_t, "C_A": C_A_t, "C_B": C_B_t,
                "C_AB_combined": C_AB_t, "E_t": E_t,
            })

    steps_arr = np.array([pt["step"] for pt in trajectory])
    m_arr = np.array([pt["m_t"] for pt in trajectory])
    sum_C_arr = np.array([pt["C_A"] + pt["C_B"] for pt in trajectory])
    target = m_arr - m_0

    idx = np.arange(len(trajectory))
    train_idx = idx[idx % 2 == 0]
    test_idx = idx[idx % 2 == 1]

    X_train, y_train = sum_C_arr[train_idx], target[train_idx]
    X_test, y_test = sum_C_arr[test_idx], target[test_idx]

    A_mat = np.vstack([X_train, np.ones_like(X_train)]).T
    (a_coef, b_coef), _, _, _ = np.linalg.lstsq(A_mat, y_train, rcond=None)

    y_pred_test = a_coef * X_test + b_coef
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2_out_of_sample = 1 - ss_res / (ss_tot + 1e-12)

    print(f"[{run_name}] TEST 1 (additive model): calibration a={a_coef:.4f} b={b_coef:.4f} "
          f"(fit on {len(train_idx)} in-sample checkpoints)")
    print(f"[{run_name}] TEST 1: out-of-sample R^2 on {len(test_idx)} held-out checkpoints = {r2_out_of_sample:.4f}")

    E_vals = np.array([pt["E_t"] for pt in trajectory])
    C_A_vals = np.array([pt["C_A"] for pt in trajectory])
    C_B_vals = np.array([pt["C_B"] for pt in trajectory])
    typical_scale = np.mean(np.abs(C_A_vals) + np.abs(C_B_vals)) / 2
    E_relative = np.abs(E_vals) / (typical_scale + 1e-9)

    print(f"[{run_name}] TEST 3 (superposition): E(t) mean={E_vals.mean():.4f} std={E_vals.std():.4f} "
          f"max|E|={np.abs(E_vals).max():.4f}")
    print(f"[{run_name}] TEST 3: typical |C_A|+|C_B| scale = {typical_scale:.4f}, "
          f"mean relative |E|/scale = {E_relative.mean():.4f}")

    result = {
        "run_name": run_name, "config": config, "seed": seed,
        "m_0": m_0,
        "test1_additive_model": {
            "a": float(a_coef), "b": float(b_coef),
            "r2_out_of_sample": float(r2_out_of_sample),
            "n_train": len(train_idx), "n_test": len(test_idx),
        },
        "test3_superposition": {
            "E_mean": float(E_vals.mean()), "E_std": float(E_vals.std()),
            "E_max_abs": float(np.abs(E_vals).max()),
            "typical_scale": float(typical_scale),
            "mean_relative_E": float(E_relative.mean()),
        },
        "trajectory": trajectory,
    }
    with open(f"/home/claude/iclr/results/additive_superposition_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    BASE = {
        "hidden_dim": 32, "phase_A_steps": 600, "phase_A_lr": 0.01,
        "phase_B_steps": 3000, "phase_B_lr": 0.005, "weight_decay": 0.0, "batch_size": 32,
    }
    run_additive_and_superposition_tests(BASE, seed=1234, run_name="C1_baseline")
