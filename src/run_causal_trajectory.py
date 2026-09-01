"""
The decisive causal experiment (per reviewer's final design). No Fisher, no
threshold hunting, no v_A optimization -- just the one causal intervention
tracked over the whole trajectory, a matched-random-direction control for
specificity, and the A+B interaction test at the final checkpoint.

Run first on the C1 baseline config (ordinary B training, no decay/
bottleneck) -- this is where we already know behavior flips fast and the
mechanism persists (Hypothesis B / row 1 of the reviewer's table). We verify
that causally and specifically here, then repeat on the weight-decay config
to causally confirm row 3 (mechanism erodes) using the SAME single-
intervention design, no archaeology machinery.
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
from src.probe import (jacobian_zor_red_vs_blue, causal_mediation_effect,
                        calibrated_random_direction, find_B_mechanism_direction,
                        double_intervention_margin)

SEED = 1234


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def zor_margin(model, ctx_name="CTX_RED"):
    """m(t) = logit_blue - logit_red for zor (positive = predicts blue)."""
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_id)
        m = (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()
    return m


def run_causal_trajectory(config: dict, seed: int, run_name: str):
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

    # --- Step 1: establish the old mechanism at theta_A ---
    model_A = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    log_A = train_phase(model_A, ds_A, steps=phase_A_steps, batch_size=batch_size,
                         lr=phase_A_lr, seed=seed, eval_every=phase_A_steps)
    phase_A_acc = log_A[-1]["eval_acc"]

    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    m_A_at_theta_A = zor_margin(model_A)  # should be << 0 (predicts red)

    mediation_at_A = causal_mediation_effect(model_A, J_A, SPECIAL_OBJECT, "red", "blue", alpha=1.0)
    delta_A_reference = -mediation_at_A["delta_A"]  # flip to blue-positive convention
    print(f"[{run_name}] Phase A acc={phase_A_acc:.3f}, m(theta_A)={m_A_at_theta_A:.3f} (should be <<0), "
          f"Delta_A(theta_A)={delta_A_reference:.3f} (should be >>0)")
    assert delta_A_reference > 1.0, "Delta_A(theta_A) not clearly positive -- mechanism not established, STOP"

    v_rand = calibrated_random_direction(J_A, seed=seed + 999)

    theta_A_state = copy.deepcopy(model_A.state_dict())

    # --- Step 2+3: train A->B, recording m(t), C_A(t), C_r(t) at each checkpoint ---
    model_AB = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=phase_B_lr, weight_decay=weight_decay)

    rng = np.random.RandomState(seed + 1)
    trajectory = []
    t_flip = None
    eval_every = max(1, phase_B_steps // 300)

    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    for step in range(phase_B_steps):
        objs, ctxs, labels = ds_B.sample_batch(batch_size, rng)
        logits = model_AB(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == phase_B_steps - 1:
            m_t = zor_margin(model_AB)
            if t_flip is None and m_t > 0:
                t_flip = step

            with torch.no_grad():
                normal_logits = model_AB(zor_id, ctx_red)
                m_normal = (normal_logits[0, COLOR2ID["blue"]] - normal_logits[0, COLOR2ID["red"]]).item()

                h = model_AB.hidden(zor_id, ctx_red).squeeze(0)
                coeff_A = (J_A @ h) / ((J_A @ J_A) + 1e-9)
                h_IA = h - coeff_A * J_A
                logits_IA = model_AB.fc2(h_IA.unsqueeze(0))
                m_IA = (logits_IA[0, COLOR2ID["blue"]] - logits_IA[0, COLOR2ID["red"]]).item()

                coeff_r = (v_rand @ h) / ((v_rand @ v_rand) + 1e-9)
                h_Ir = h - coeff_r * v_rand
                logits_Ir = model_AB.fc2(h_Ir.unsqueeze(0))
                m_Ir = (logits_Ir[0, COLOR2ID["blue"]] - logits_Ir[0, COLOR2ID["red"]]).item()

            C_A_t = m_IA - m_normal
            C_r_t = m_Ir - m_normal

            trajectory.append({
                "step": step, "m_normal": m_normal, "m_IA": m_IA, "m_Ir": m_Ir,
                "C_A": C_A_t, "C_r": C_r_t,
            })

    print(f"[{run_name}] behavioral flip (predicts blue) at step: {t_flip}")

    # --- Step 4: killer double-intervention at the FINAL checkpoint ---
    J_B = find_B_mechanism_direction(model_AB, ctx_name="CTX_RED")

    def blue_margin_via(do_A, do_B):
        rb_margin = double_intervention_margin(model_AB, zor_id, ctx_red, J_A, J_B, alpha=1.0, do_A=do_A, do_B=do_B)
        return -rb_margin  # red-vs-blue -> blue-vs-red convention

    m_intact = blue_margin_via(False, False)
    m_A = blue_margin_via(True, False)
    m_B = blue_margin_via(False, True)
    m_AB_both = blue_margin_via(True, True)

    Delta_A = m_A - m_intact
    Delta_B = m_B - m_intact
    Delta_AB = m_AB_both - m_intact
    I_AB = Delta_AB - Delta_A - Delta_B

    print(f"[{run_name}] FINAL double-intervention (blue-positive margin):")
    print(f"  intact:         m={m_intact:.3f} ({'blue' if m_intact>0 else 'red'})")
    print(f"  A intervention: m={m_A:.3f} ({'blue' if m_A>0 else 'red'})  Delta_A={Delta_A:.3f}")
    print(f"  B intervention: m={m_B:.3f} ({'blue' if m_B>0 else 'red'})  Delta_B={Delta_B:.3f}")
    print(f"  A+B:            m={m_AB_both:.3f} ({'blue' if m_AB_both>0 else 'red'})  Delta_AB={Delta_AB:.3f}")
    print(f"  Interaction I_AB = Delta_AB - Delta_A - Delta_B = {I_AB:.3f}")

    final_pt = trajectory[-1]
    print(f"[{run_name}] SUMMARY: t_flip={t_flip}, final C_A={final_pt['C_A']:.3f}, final C_r={final_pt['C_r']:.3f}, "
          f"specificity (|C_A| >> |C_r|): {abs(final_pt['C_A']) > 3 * abs(final_pt['C_r']) + 0.1}")

    result = {
        "run_name": run_name, "config": config, "seed": seed,
        "phase_A_acc": phase_A_acc,
        "delta_A_at_theta_A": delta_A_reference,
        "t_flip": t_flip,
        "trajectory": trajectory,
        "final_double_intervention": {
            "m_intact": m_intact, "m_A": m_A, "m_B": m_B, "m_AB": m_AB_both,
            "Delta_A": Delta_A, "Delta_B": Delta_B, "Delta_AB": Delta_AB, "I_AB": I_AB,
        },
    }
    with open(f"/home/claude/iclr/results/causal_trajectory_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    BASE = {
        "hidden_dim": 32, "phase_A_steps": 600, "phase_A_lr": 0.01,
        "phase_B_steps": 3000, "phase_B_lr": 0.005, "weight_decay": 0.0, "batch_size": 32,
    }
    print("=" * 60)
    print("RUN 1: C1 baseline (ordinary B training)")
    print("=" * 60)
    run_causal_trajectory(BASE, seed=SEED, run_name="C1_baseline")

    print()
    print("=" * 60)
    print("RUN 2: weight decay (row 3 candidate -- mechanism should erode)")
    print("=" * 60)
    DECAY_CONFIG = {**BASE, "bottleneck_dim": 2, "weight_decay": 0.1, "phase_B_steps": 8000}
    run_causal_trajectory(DECAY_CONFIG, seed=SEED, run_name="C3_decay")
