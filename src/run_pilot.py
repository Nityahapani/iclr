"""
Neural Archaeology -- killer experiment, v4.

Adds two things to the frozen v3 design (rho_A, cosine alignment of causal
Jacobians):

1. CAUSAL MEDIATION TEST (Delta_A). rho_A tells us whether the OLD causal
   geometry (J_A^zor) still resembles the model's current geometry. It does
   NOT tell us whether that geometry is still functionally live. Delta_A
   directly intervenes: remove the component of the model's CURRENT hidden
   state along the FROZEN J_A direction, and measure the change in the
   red-vs-blue margin.
     - Delta_A stays large  -> Hypothesis B (latent persistence): the old
       computation is still causally active, just overridden/dominated by
       something else downstream. Interesting, but not "a fossil."
     - Delta_A -> 0 while rho_A stays ~1 -> Hypothesis A (fossil): the
       geometry survives but has become causally inert. This is the
       structure-survives-function-disappears result the paper wants.
   Tracked across the FULL B-training trajectory, alongside rho_A(t) and
   behavioral persistence, to see all three curves together.

2. SHAM-C SPECIFICITY CONTROL. A second object (fenn) undergoes a matched
   "phase C" concurrently with zor's phase A: same frequency, same number of
   steps, same context-dependent structure, same final phase-accuracy --
   but teaches fenn=green (not red) for CTX_RED. Both zor and fenn are then
   overwritten to unconditional blue in phase B. We ask:
     rho_A(A->B, measured on zor)   vs   rho_A(A->B, measured on fenn's own
     analogous J_C^fenn geometry)
   The real test: is rho_A specific to WHAT was learned (zor's red binding)
   or would ANY object that underwent an earlier phase show the same
   alignment merely because "this region of parameter space was optimized
   earlier"? We check this by computing J_A^zor vs J_T^fenn cross-alignment
   (should be LOW if rho_A is content-specific, not just phase-specific) and
   comparing each object's own historical alignment (zor's rho_A vs fenn's
   own analogous rho_C) side by side.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, SHAM_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import (jacobian_zor_red_vs_blue, jacobian_fenn_green_vs_blue,
                        cosine_alignment, causal_mediation_effect)

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS = {}


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES)


def zor_red_ctx_behavior(model):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_red)
        pred = logits.argmax(-1).item()
        margin = (logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]).item()
    return (pred == COLOR2ID["red"]), margin


def fenn_green_ctx_behavior(model):
    fenn_id = torch.tensor([OBJ2ID[SHAM_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(fenn_id, ctx_red)
        pred = logits.argmax(-1).item()
        margin = (logits[0, COLOR2ID["green"]] - logits[0, COLOR2ID["blue"]]).item()
    return (pred == COLOR2ID["green"]), margin


def run():
    filler_mapping = make_filler_mapping(seed=SEED)
    ds_A = PhaseDataset(filler_mapping, phase="A")   # also IS phase "C" for fenn, concurrently
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    # ============ STAGE 1: Train A (=C for fenn), capture J_A^zor and J_C^fenn ============
    model_A = new_model()
    log_A = train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=SEED, eval_every=600)
    RESULTS["phase_A_final_acc"] = log_A[-1]["eval_acc"]
    print(f"[Stage 1] Phase A(=C) final eval acc: {log_A[-1]['eval_acc']:.3f}")

    J_A_zor = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    J_C_fenn = jacobian_fenn_green_vs_blue(model_A, ctx_name="CTX_RED")
    RESULTS["J_A_zor_norm"] = J_A_zor.norm().item()
    RESULTS["J_C_fenn_norm"] = J_C_fenn.norm().item()

    is_red, margin_red = zor_red_ctx_behavior(model_A)
    is_green, margin_green = fenn_green_ctx_behavior(model_A)
    RESULTS["zor_red_behavior_at_theta_A"] = {"is_red": is_red, "margin": margin_red}
    RESULTS["fenn_green_behavior_at_theta_C"] = {"is_green": is_green, "margin": margin_green}
    print(f"[Stage 1] zor pred red={is_red} (margin={margin_red:.3f}); "
          f"fenn pred green={is_green} (margin={margin_green:.3f})")
    assert is_red and is_green, "zor/fenn must both show their trained binding at theta_A/C -- STOP if not"

    # Cross-alignment sanity: J_A_zor and J_C_fenn should be ~unrelated (different colors/objects)
    cross_align_A_C = cosine_alignment(J_A_zor, J_C_fenn)
    RESULTS["cross_alignment_J_A_zor_vs_J_C_fenn"] = cross_align_A_C
    print(f"[Stage 1] cross-alignment cos(J_A_zor, J_C_fenn) at theta_A = {cross_align_A_C:.4f} "
          f"(sanity: should not be trivially ~1, these are different bindings)")

    theta_A_state = copy.deepcopy(model_A.state_dict())

    # ============ STAGE 2: Train A->B and B-only (matched) ============
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

    is_red_T_AB, margin_T_AB = zor_red_ctx_behavior(model_AB_T)
    is_green_T_AB, margin_green_T_AB = fenn_green_ctx_behavior(model_AB_T)
    is_red_T_B, margin_T_B = zor_red_ctx_behavior(model_B_T)
    RESULTS["behavioral_erasure_AB_zor"] = {"is_red": is_red_T_AB, "margin": margin_T_AB}
    RESULTS["behavioral_erasure_AB_fenn"] = {"is_green": is_green_T_AB, "margin": margin_green_T_AB}
    print(f"[Stage 2] at theta_T: M_AB zor-red={is_red_T_AB} (margin={margin_T_AB:.3f}), "
          f"M_AB fenn-green={is_green_T_AB} (margin={margin_green_T_AB:.3f})")
    assert (not is_red_T_AB) and (not is_green_T_AB), "behavioral erasure incomplete for zor or fenn -- STOP"

    # ============ STAGE 3: ARCHAEOLOGY -- rho_A alignment (as in v3) ============
    J_T_zor_AB = jacobian_zor_red_vs_blue(model_AB_T, ctx_name="CTX_RED")
    J_T_zor_B = jacobian_zor_red_vs_blue(model_B_T, ctx_name="CTX_RED")
    J_T_fenn_AB = jacobian_fenn_green_vs_blue(model_AB_T, ctx_name="CTX_RED")

    rho_A_AB = cosine_alignment(J_A_zor, J_T_zor_AB)
    rho_A_B = cosine_alignment(J_A_zor, J_T_zor_B)
    rho_C_fenn_AB = cosine_alignment(J_C_fenn, J_T_fenn_AB)  # fenn's OWN historical alignment

    RESULTS["rho_A_M_AB_zor"] = rho_A_AB
    RESULTS["rho_A_M_B_zor"] = rho_A_B
    RESULTS["rho_C_M_AB_fenn"] = rho_C_fenn_AB
    print(f"[Stage 3] rho_A(M_AB, zor) = {rho_A_AB:.4f}   rho_A(M_B, zor) = {rho_A_B:.4f}")
    print(f"[Stage 3] rho_C(M_AB, fenn, own history) = {rho_C_fenn_AB:.4f}  "
          f"(should ALSO be high -- fenn has its own real history too, this isn't the specificity test)")

    # THE ACTUAL SPECIFICITY TEST: cross-object alignment. Does zor's final
    # geometry (J_T_zor_AB) align with FENN's frozen historical direction
    # (J_C_fenn, a DIFFERENT binding)? If rho_A were merely "this parameter
    # region was optimized earlier" rather than content-specific, we might
    # expect J_T_zor_AB to spuriously align with J_C_fenn too (since both
    # zor and fenn share the same phase-A/C training window). We want this
    # LOW, showing specificity to WHAT was learned, not just WHEN.
    cross_rho_zorT_vs_Cfenn = cosine_alignment(J_T_zor_AB, J_C_fenn)
    RESULTS["cross_rho_zorT_vs_J_C_fenn"] = cross_rho_zorT_vs_Cfenn
    print(f"[Stage 3, SPECIFICITY TEST] cos(J_T_zor_AB, J_C_fenn) = {cross_rho_zorT_vs_Cfenn:.4f} "
          f"(want LOW -- zor's final geometry should align with ITS OWN history, not fenn's)")
    print(f"[Stage 3, SPECIFICITY TEST] compare: rho_A(zor's own history)={rho_A_AB:.4f} "
          f"vs cross-alignment(zor's final, fenn's history)={cross_rho_zorT_vs_Cfenn:.4f}")
    specificity_confirmed = rho_A_AB > cross_rho_zorT_vs_Cfenn + 0.3  # require a real gap, not noise
    RESULTS["specificity_confirmed"] = specificity_confirmed
    print(f"[Stage 3, SPECIFICITY TEST] specificity_confirmed (own-history alignment clearly exceeds "
          f"cross-object alignment): {specificity_confirmed}")

    # ============ STAGE 4/5: TRAJECTORY -- rho_A(t), behavior(t), AND Delta_A(t) together ============
    trajectory = []
    for entry in log_AB:
        m = new_model()
        m.load_state_dict(entry["state_dict"])
        is_red_t, margin_t = zor_red_ctx_behavior(m)
        J_t = jacobian_zor_red_vs_blue(m, ctx_name="CTX_RED")
        rho_t = cosine_alignment(J_A_zor, J_t)
        mediation = causal_mediation_effect(m, J_A_zor, SPECIAL_OBJECT, "red", "blue", alpha=1.0)
        trajectory.append({
            "step": entry["step"],
            "zor_red_margin": margin_t,
            "zor_predicts_red": is_red_t,
            "rho_A": rho_t,
            "delta_A": mediation["delta_A"],
            "m_normal": mediation["m_normal"],
            "m_intervened": mediation["m_intervened"],
            "eval_acc": entry["eval_acc"],
        })

    RESULTS["trajectory"] = trajectory
    RESULTS["rho_A_B_only_baseline"] = rho_A_B

    behavior_flip_step = next((pt["step"] for pt in trajectory if not pt["zor_predicts_red"]), None)
    RESULTS["behavior_flip_step"] = behavior_flip_step

    final_pt = trajectory[-1]
    print(f"[Stage 4/5] behavior flips at step {behavior_flip_step}")
    print(f"[Stage 4/5] at FINAL step {final_pt['step']}: rho_A={final_pt['rho_A']:.4f}, "
          f"delta_A={final_pt['delta_A']:.4f} (m_normal={final_pt['m_normal']:.3f}, "
          f"m_intervened={final_pt['m_intervened']:.3f})")

    # Also compute Delta_A at theta_A itself (step 0, before any B-training) as
    # a reference for "how large is the effect when the mechanism is DEFINITELY
    # still live" -- this calibrates what "large" vs "~0" means for delta_A.
    mediation_at_A = causal_mediation_effect(model_A, J_A_zor, SPECIAL_OBJECT, "red", "blue", alpha=1.0)
    RESULTS["delta_A_at_theta_A_reference"] = mediation_at_A
    print(f"[Stage 4/5] REFERENCE delta_A at theta_A (mechanism known-live): "
          f"delta_A={mediation_at_A['delta_A']:.4f} (m_normal={mediation_at_A['m_normal']:.3f}, "
          f"m_intervened={mediation_at_A['m_intervened']:.3f})")

    # Hypothesis classification
    final_delta_A = final_pt["delta_A"]
    reference_delta_A = mediation_at_A["delta_A"]
    fraction_of_reference_effect = (abs(final_delta_A) / (abs(reference_delta_A) + 1e-9))
    RESULTS["fraction_of_reference_mediation_effect_remaining"] = fraction_of_reference_effect
    print(f"[Stage 4/5] final |delta_A| is {fraction_of_reference_effect*100:.1f}% of the theta_A reference "
          f"effect magnitude")
    if fraction_of_reference_effect < 0.15:
        hypothesis = "A (fossil): structure (rho_A~1) survives, function (delta_A) has collapsed"
    elif fraction_of_reference_effect > 0.5:
        hypothesis = "B (latent persistence): old mechanism still substantially causally active"
    else:
        hypothesis = "AMBIGUOUS: partial mediation effect remains, needs more granular trajectory + seeds"
    RESULTS["hypothesis_classification"] = hypothesis
    print(f"[Stage 4/5] CLASSIFICATION: {hypothesis}")

    RESULTS["n_note"] = ("Single-seed pilot (n=1). Establishes the full measurement pipeline "
                          "(rho_A structural alignment, delta_A causal mediation, cross-object "
                          "specificity control) end-to-end. A real claim requires >=20-50 seeds "
                          "per condition with paired statistical tests on rho_A gap, delta_A "
                          "fraction-remaining, and the specificity gap.")

    with open("/home/claude/iclr/results/pilot_run_v4.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print("\n=== SUMMARY (scalars only) ===")
    for k, v in RESULTS.items():
        if k != "trajectory":
            print(f"{k}: {v}")

    return RESULTS


if __name__ == "__main__":
    run()
