"""
Neural Archaeology -- killer experiment, FROZEN design (v3: causal Jacobian).

Central object: not a vector, a RELATIONSHIP.
  J(o,c) = grad_h [logit_red(o,c) - logit_blue(o,c)]
This is the model's own local causal gradient for the red/blue decision on
object o in context c -- not a probe, not a subtraction, the actual
sensitivity direction the model uses.

J_A^zor = J(zor, CTX_RED) at theta_A (immediately after phase A, zor=red is
          context-dependent and behaviorally live).

Historical signature: after B-training to a matched theta_T, does the FINAL
model's own Jacobian for zor, J_T^zor, retain alignment with J_A^zor -- more
so than a from-scratch B-only control does with the SAME frozen J_A^zor?

  rho_A(M) = cos( J_A^zor, J_M^zor )

Prediction:  rho_A(M_AB) > rho_A(M_B)   [population-level; single-seed here is a pilot]

Critical dissociation (history vs. ordinary inertia):
  Vary amount of B-training t = 0, t1, t2, ..., T.
  Track rho_A(t) alongside BEHAVIORAL persistence of zor=red (does the model
  still predict red for zor at that checkpoint?).
  - Ordinary inertia: rho_A(t) decays in lockstep with behavior, converging
    smoothly toward the B-only baseline.
  - Genuine fossil: rho_A(t) remains elevated above the B-only baseline well
    after behavioral red-prediction has already hit chance/blue.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import jacobian_zor_red_vs_blue, cosine_alignment

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS = {}


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES)


def zor_red_ctx_behavior(model):
    """Does the model currently predict red for zor in CTX_RED? Returns
    (predicted_label_is_red: bool, red_blue_margin: float)."""
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_red)
        pred = logits.argmax(-1).item()
        margin = (logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]).item()
    return (pred == COLOR2ID["red"]), margin


def run():
    filler_mapping = make_filler_mapping(seed=SEED)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    # ============ STAGE 1: Train A, capture J_A^zor ============
    model_A = new_model()
    log_A = train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=SEED, eval_every=600)
    RESULTS["phase_A_final_acc"] = log_A[-1]["eval_acc"]
    print(f"[Stage 1] Phase A final eval acc: {log_A[-1]['eval_acc']:.3f}")

    J_A_zor = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    RESULTS["J_A_zor_norm"] = J_A_zor.norm().item()
    is_red, margin = zor_red_ctx_behavior(model_A)
    RESULTS["zor_red_behavior_at_theta_A"] = {"is_red": is_red, "margin": margin}
    print(f"[Stage 1] J_A_zor norm={J_A_zor.norm().item():.4f}; zor pred red={is_red}, margin={margin:.3f}")
    assert is_red, "zor must actually predict red at theta_A -- STOP if not"

    theta_A_state = copy.deepcopy(model_A.state_dict())

    # ============ STAGE 2: Train A->B (fine-grained checkpoints) and B-only (matched final) ============
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=SEED + 1, eval_every=10)

    model_B = new_model()
    torch.manual_seed(SEED + 2)
    log_B = train_phase(model_B, ds_B_only, steps=3000, batch_size=32, lr=0.005, seed=SEED + 3, eval_every=10)

    matched_entry, matched_kl = find_matched_checkpoint(log_B, log_AB)
    print(f"[Stage 2] best KL(M_AB, M_B_final) = {matched_kl:.5f} at AB step {matched_entry['step']}")
    RESULTS["matching_kl"] = matched_kl
    RESULTS["matching_step"] = matched_entry["step"]

    model_AB_T = new_model()
    model_AB_T.load_state_dict(matched_entry["state_dict"])
    model_B_T = new_model()
    model_B_T.load_state_dict(log_B[-1]["state_dict"])

    theta_T_acc_AB = matched_entry["eval_acc"]
    theta_T_acc_B = log_B[-1]["eval_acc"]
    RESULTS["theta_T_acc_AB"] = theta_T_acc_AB
    RESULTS["theta_T_acc_B"] = theta_T_acc_B
    print(f"[Stage 2] theta_T: M_AB acc={theta_T_acc_AB:.3f}, M_B acc={theta_T_acc_B:.3f}")

    # Behavioral erasure check
    is_red_T_AB, margin_T_AB = zor_red_ctx_behavior(model_AB_T)
    is_red_T_B, margin_T_B = zor_red_ctx_behavior(model_B_T)
    RESULTS["behavioral_erasure_AB"] = {"is_red": is_red_T_AB, "margin": margin_T_AB}
    RESULTS["behavioral_erasure_B"] = {"is_red": is_red_T_B, "margin": margin_T_B}
    print(f"[Stage 2] at theta_T: M_AB zor-red={is_red_T_AB} (margin={margin_T_AB:.3f}), "
          f"M_B zor-red={is_red_T_B} (margin={margin_T_B:.3f})")
    behavioral_erasure_confirmed = (not is_red_T_AB)
    RESULTS["behavioral_erasure_confirmed"] = behavioral_erasure_confirmed
    assert behavioral_erasure_confirmed, "zor still predicts red at theta_T -- B-training insufficient, STOP"

    # ============ STAGE 3: ARCHAEOLOGY -- rho_A alignment ============
    J_T_zor_AB = jacobian_zor_red_vs_blue(model_AB_T, ctx_name="CTX_RED")
    J_T_zor_B = jacobian_zor_red_vs_blue(model_B_T, ctx_name="CTX_RED")

    rho_A_AB = cosine_alignment(J_A_zor, J_T_zor_AB)
    rho_A_B = cosine_alignment(J_A_zor, J_T_zor_B)
    RESULTS["rho_A_M_AB"] = rho_A_AB
    RESULTS["rho_A_M_B"] = rho_A_B
    RESULTS["rho_A_gap"] = rho_A_AB - rho_A_B
    print(f"[Stage 3] rho_A(M_AB) = cos(J_A_zor, J_T_zor_AB) = {rho_A_AB:.4f}")
    print(f"[Stage 3] rho_A(M_B)  = cos(J_A_zor, J_T_zor_B)  = {rho_A_B:.4f}")
    print(f"[Stage 3] GAP (treatment - control) = {rho_A_AB - rho_A_B:.4f}")
    print(f"[Stage 3] Prediction rho_A(M_AB) > rho_A(M_B): {rho_A_AB > rho_A_B}")

    # ============ STAGE 4: DISSOCIATION -- history vs ordinary inertia ============
    # Walk the AB checkpoints in order, tracking (a) behavioral persistence of
    # red for zor, and (b) rho_A(t) at each checkpoint, against the B-only
    # final baseline rho_A(M_B) as a reference line.
    trajectory = []
    for entry in log_AB:
        m = new_model()
        m.load_state_dict(entry["state_dict"])
        is_red_t, margin_t = zor_red_ctx_behavior(m)
        J_t = jacobian_zor_red_vs_blue(m, ctx_name="CTX_RED")
        rho_t = cosine_alignment(J_A_zor, J_t)
        trajectory.append({
            "step": entry["step"],
            "zor_red_margin": margin_t,
            "zor_predicts_red": is_red_t,
            "rho_A": rho_t,
            "eval_acc": entry["eval_acc"],
        })

    RESULTS["trajectory"] = trajectory
    RESULTS["rho_A_B_only_baseline"] = rho_A_B

    # Find the step at which behavior flips (red -> not-red) and compare to
    # where rho_A(t) actually reaches the B-only baseline.
    behavior_flip_step = None
    for pt in trajectory:
        if not pt["zor_predicts_red"]:
            behavior_flip_step = pt["step"]
            break
    RESULTS["behavior_flip_step"] = behavior_flip_step

    rho_at_flip = None
    if behavior_flip_step is not None:
        for pt in trajectory:
            if pt["step"] == behavior_flip_step:
                rho_at_flip = pt["rho_A"]
                break
    RESULTS["rho_A_at_behavior_flip_step"] = rho_at_flip
    print(f"[Stage 4] Behavior (zor->red) flips at step {behavior_flip_step}; "
          f"rho_A at that step = {rho_at_flip}")
    print(f"[Stage 4] rho_A at final matched step = {rho_A_AB:.4f}; B-only baseline = {rho_A_B:.4f}")

    if rho_at_flip is not None:
        dissociation = rho_at_flip - rho_A_B
        RESULTS["dissociation_margin"] = dissociation
        print(f"[Stage 4] Dissociation margin (rho_A at flip minus B-only baseline) = {dissociation:.4f}")
        print(f"[Stage 4] {'GENUINE FOSSIL SIGNAL (single seed, needs replication)' if dissociation > 0.05 else 'Consistent with ordinary inertia / no clear dissociation'}")

    RESULTS["n_note"] = ("Single-seed pilot (n=1). This establishes the measurement pipeline works "
                          "end-to-end (Jacobian construction, KL-matching, trajectory tracking). "
                          "A real claim requires >=20-50 seeds per condition, a paired statistical "
                          "test (e.g. paired t-test or Wilcoxon on rho_A(M_AB) - rho_A(M_B) across "
                          "seeds), and ideally repeating Stage 4 across seeds to see if the "
                          "dissociation-margin sign is consistent.")

    with open("/home/claude/iclr/results/pilot_run_v3.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print("\n=== SUMMARY (scalars only) ===")
    for k, v in RESULTS.items():
        if k != "trajectory":
            print(f"{k}: {v}")

    return RESULTS


if __name__ == "__main__":
    run()
