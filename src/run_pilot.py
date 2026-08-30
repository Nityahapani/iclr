"""
Neural Archaeology -- killer experiment, corrected design (v2: diff-in-diff v_A).

Stage 1: Train A until convergence. Build v_A via difference-in-differences
         (zor's red/blue-context contrast MINUS control object's contrast).
         Verify causal necessity + SPECIFICITY at theta_A (control object and
         fillers should be far less affected by v_A ablation than zor is).

Stage 2: Train A->B (treatment) until KL-matched to a from-scratch B-only
         (control). Verify zor is behaviorally blue in BOTH contexts at
         theta_T (its context-dependence, not just its color, is erased --
         the phase-B data makes zor context-INDEPENDENT).

Stage 3: Archaeology. Compute C_A(M) = v_A^T [h(zor,red_ctx) - h(zor,blue_ctx)]
         for M_AB and M_B. Prediction: C_A(M_AB) != C_A(M_B), specifically
         |C_A(M_AB)| should retain more structure than the matched control,
         even though zor is behaviorally/causally context-independent (blue)
         in both models.

Stage 4: Causal archaeology. Ablate v_A from hidden state at theta_T. Check:
         (a) task behavior for zor stays blue in both contexts (Delta_task ~ 0)
         (b) C_A(M) collapses toward the sham/random-direction baseline.
"""
import json
import copy
import numpy as np
import torch
import torch.nn.functional as Fnn

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, CONTROL_OBJECT, FILLER_OBJECTS,
                       VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import (build_v_A_diff_in_diff, C_statistic, causal_ablate_and_forward,
                        find_minimal_flipping_alpha_ctx, verify_v_A_causal_ctx,
                        build_v_rand, build_v_sham_diff_in_diff)
from src.fisher import fisher_curvature_along_v

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS = {}


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES)


def run():
    filler_mapping = make_filler_mapping(seed=SEED)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    # ============ STAGE 1 ============
    model_A = new_model()
    log_A = train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=SEED, eval_every=600)
    RESULTS["phase_A_final_acc"] = log_A[-1]["eval_acc"]
    print(f"[Stage 1] Phase A final eval acc: {log_A[-1]['eval_acc']:.3f}")

    v_A, raw_diff, diff_magnitude = build_v_A_diff_in_diff(model_A)
    RESULTS["v_A_diff_magnitude"] = diff_magnitude
    print(f"[Stage 1] v_A diff-in-diff magnitude: {diff_magnitude:.4f}")

    min_alpha = find_minimal_flipping_alpha_ctx(model_A, v_A)
    RESULTS["min_flipping_alpha"] = min_alpha
    causal_check = verify_v_A_causal_ctx(model_A, v_A, filler_mapping, alpha=min_alpha)
    RESULTS["v_A_causal_check_at_theta_A"] = causal_check
    print(f"[Stage 1] min flipping alpha={min_alpha}")
    print(f"[Stage 1] causal check: {json.dumps(causal_check, indent=2, default=str)}")

    if not causal_check["causal_effect_confirmed"]:
        print("[Stage 1] FAILED causal_effect_confirmed -- trying larger alpha sweep before aborting")
        for a in (3.5, 4.0, 5.0, 6.0, 8.0):
            causal_check = verify_v_A_causal_ctx(model_A, v_A, filler_mapping, alpha=a)
            print(f"  alpha={a}: {causal_check}")
            if causal_check["causal_effect_confirmed"]:
                min_alpha = a
                RESULTS["min_flipping_alpha"] = min_alpha
                RESULTS["v_A_causal_check_at_theta_A"] = causal_check
                break
    assert causal_check["causal_effect_confirmed"], "v_A still not causally necessary+specific at theta_A -- STOP"
    print(f"[Stage 1] PASSED: v_A causally necessary for zor AND specific (control/fillers unaffected)")

    theta_A_state = copy.deepcopy(model_A.state_dict())

    # ============ STAGE 2 ============
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=SEED + 1, eval_every=20)

    model_B = new_model()
    torch.manual_seed(SEED + 2)
    log_B = train_phase(model_B, ds_B_only, steps=3000, batch_size=32, lr=0.005, seed=SEED + 3, eval_every=20)

    matched_entry, matched_kl = find_matched_checkpoint(log_B, log_AB)
    print(f"[Stage 2] best KL(M_AB, M_B_final) = {matched_kl:.5f} at AB step {matched_entry['step']}")
    RESULTS["matching_kl"] = matched_kl
    RESULTS["matching_step"] = matched_entry["step"]
    model_AB.load_state_dict(matched_entry["state_dict"])
    theta_T_acc_AB = matched_entry["eval_acc"]
    theta_T_acc_B = log_B[-1]["eval_acc"]
    RESULTS["theta_T_acc_AB"] = theta_T_acc_AB
    RESULTS["theta_T_acc_B"] = theta_T_acc_B
    print(f"[Stage 2] theta_T: M_AB acc={theta_T_acc_AB:.3f}, M_B acc={theta_T_acc_B:.3f}")

    # Behavioral/causal erasure check: zor should be BLUE in both contexts at theta_T
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    ctx_blue = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)
    blue_label = COLOR2ID["blue"]
    with torch.no_grad():
        zor_red_ctx_pred = model_AB(zor_id, ctx_red).argmax(-1).item()
        zor_blue_ctx_pred = model_AB(zor_id, ctx_blue).argmax(-1).item()
    behavioral_erasure = (zor_red_ctx_pred == blue_label and zor_blue_ctx_pred == blue_label)
    RESULTS["behavioral_erasure_confirmed"] = behavioral_erasure
    print(f"[Stage 2] zor pred in CTX_RED={zor_red_ctx_pred}, CTX_BLUE={zor_blue_ctx_pred} "
          f"(both should be blue_id={blue_label}) -- erasure confirmed: {behavioral_erasure}")

    # Causal erasure: does ablating v_A restore context-dependence at theta_T?
    post_ablate_red = causal_ablate_and_forward(model_AB, zor_id, ctx_red, v_A, alpha=min_alpha)
    post_ablate_red_pred = post_ablate_red.argmax(-1).item()
    causal_erasure_confirmed = (post_ablate_red_pred == blue_label)  # should NOT flip back to red
    RESULTS["causal_erasure_confirmed"] = causal_erasure_confirmed
    print(f"[Stage 2] ablating v_A at theta_T on zor+CTX_RED -> pred={post_ablate_red_pred} "
          f"(causal erasure confirmed: {causal_erasure_confirmed})")

    # ============ STAGE 3: ARCHAEOLOGY ============
    C_A_on_AB = C_statistic(model_AB, v_A, SPECIAL_OBJECT)
    C_A_on_B = C_statistic(model_B, v_A, SPECIAL_OBJECT)
    RESULTS["C_A_on_M_AB"] = C_A_on_AB
    RESULTS["C_A_on_M_B"] = C_A_on_B
    RESULTS["C_A_gap"] = C_A_on_AB - C_A_on_B
    print(f"[Stage 3] C_A(M_AB)={C_A_on_AB:.4f}  C_A(M_B)={C_A_on_B:.4f}  gap={C_A_on_AB - C_A_on_B:.4f}")

    v_sham = build_v_sham_diff_in_diff(model_A, seed=999)
    hidden_dim = model_AB.fc1.out_features
    v_rand = build_v_rand(hidden_dim, seed=42)

    C_sham_on_AB = C_statistic(model_AB, v_sham, SPECIAL_OBJECT)
    C_sham_on_B = C_statistic(model_B, v_sham, SPECIAL_OBJECT)
    C_rand_on_AB = C_statistic(model_AB, v_rand, SPECIAL_OBJECT)
    C_rand_on_B = C_statistic(model_B, v_rand, SPECIAL_OBJECT)
    RESULTS["C_sham_on_M_AB"] = C_sham_on_AB
    RESULTS["C_sham_on_M_B"] = C_sham_on_B
    RESULTS["C_rand_on_M_AB"] = C_rand_on_AB
    RESULTS["C_rand_on_M_B"] = C_rand_on_B
    print(f"[Stage 3, controls] C_sham: AB={C_sham_on_AB:.4f} B={C_sham_on_B:.4f}  |  "
          f"C_rand: AB={C_rand_on_AB:.4f} B={C_rand_on_B:.4f}")

    # ============ STAGE 4: CAUSAL ARCHAEOLOGY (KNOCKOUT) ============
    eval_objs, eval_ctxs, eval_labels = ds_B.full_eval_set()

    with torch.no_grad():
        base_logits = model_AB(eval_objs, eval_ctxs)
        base_acc = (base_logits.argmax(-1) == eval_labels).float().mean().item()

    logits_after_vA = causal_ablate_and_forward(model_AB, eval_objs, eval_ctxs, v_A, alpha=min_alpha)
    acc_after_vA = (logits_after_vA.argmax(-1) == eval_labels).float().mean().item()

    logits_after_vrand = causal_ablate_and_forward(model_AB, eval_objs, eval_ctxs, v_rand, alpha=min_alpha)
    acc_after_vrand = (logits_after_vrand.argmax(-1) == eval_labels).float().mean().item()

    delta_task_vA = acc_after_vA - base_acc
    delta_task_vrand = acc_after_vrand - base_acc
    RESULTS["knockout_base_acc"] = base_acc
    RESULTS["knockout_acc_after_vA"] = acc_after_vA
    RESULTS["knockout_acc_after_vrand"] = acc_after_vrand
    RESULTS["delta_task_vA"] = delta_task_vA
    RESULTS["delta_task_vrand"] = delta_task_vrand
    print(f"[Stage 4] task acc: base={base_acc:.3f} after v_A={acc_after_vA:.3f} (Δ={delta_task_vA:+.3f}) "
          f"after v_rand={acc_after_vrand:.3f} (Δ={delta_task_vrand:+.3f})")

    # Archaeological collapse: recompute C_A using a hidden state that already has v_A ablated
    def C_statistic_with_ablation(model, v_measure, v_ablate, object_name, alpha):
        obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
        cr = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
        cb = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)
        with torch.no_grad():
            h_red = model.hidden(obj_id, cr).squeeze(0)
            h_blue = model.hidden(obj_id, cb).squeeze(0)
            proj_r = (h_red @ v_ablate) * v_ablate
            proj_b = (h_blue @ v_ablate) * v_ablate
            h_red_ab = h_red - alpha * proj_r
            h_blue_ab = h_blue - alpha * proj_b
            c = (v_measure @ (h_red_ab - h_blue_ab)).item()
        return c

    C_A_after_vA_ablation = C_statistic_with_ablation(model_AB, v_A, v_A, SPECIAL_OBJECT, min_alpha)
    C_A_after_vrand_ablation = C_statistic_with_ablation(model_AB, v_A, v_rand, SPECIAL_OBJECT, min_alpha)
    RESULTS["C_A_after_vA_ablation"] = C_A_after_vA_ablation
    RESULTS["C_A_after_vrand_ablation"] = C_A_after_vrand_ablation
    print(f"[Stage 4] C_A(M_AB) after removing v_A component: {C_A_after_vA_ablation:.4f} "
          f"(pre-ablation was {C_A_on_AB:.4f})")
    print(f"[Stage 4] C_A(M_AB) after removing v_rand component (control): {C_A_after_vrand_ablation:.4f}")

    RESULTS["n_note"] = ("Single-seed pilot (n=1). Effect sizes here establish feasibility of the "
                          "measurement pipeline; a multi-seed replication (>=20 seeds per condition) "
                          "with paired statistical tests is required before any claim about C_A(M_AB) "
                          "vs C_A(M_B) being a reliable population-level effect.")

    with open("/home/claude/iclr/results/pilot_run_v2.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print("\n=== SUMMARY ===")
    for k, v in RESULTS.items():
        print(f"{k}: {v}")

    return RESULTS


if __name__ == "__main__":
    run()
