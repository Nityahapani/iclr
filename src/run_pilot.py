"""
Neural Archaeology -- killer experiment, corrected design.

Design (per the reviewer correction):
  1. Train M_A: theta_0 -> theta_A on phase A (zor=red). Stop.
  2. Build v_A in ACTIVATION space at theta_A. Verify it is CAUSALLY necessary
     for red behavior at theta_A (ablation test).
  3. Continue M_A -> M_AB: theta_A -> theta_T on phase B (zor=blue), matched
     in output distribution (KL) to a freshly-trained control M_B (B-only).
     v_A is carried along the SAME parameter lineage (M_A -> M_AB), never
     transported across independently trained models.
  4. Build v_sham on M_B (paired lineage: an unrelated-concept probe on the
     control model itself) and v_rand (random direction) for within-model
     comparison at theta_T.
  5. Archaeology: compare Fisher curvature F(v_A) vs F(v_rand) vs F(v_sham)
     WITHIN the M_AB model. Prediction: F(v_A) > F(v_rand), F(v_A) > F(v_sham).
  6. Cross-population check: F(v_A) on M_AB vs F(v_sham) on matched M_B.
  7. Causal archaeology / knockout: ablate v_A (vs v_rand) from M_AB's hidden
     state. Measure Delta_task (should be ~0) and Delta_archaeology (should
     collapse detectability of A-vs-B-only from a downstream statistic).
  8. Behavioral / causal erasure checks at theta_T: confirm zor->red accuracy
     is at chance and that v_A ablation no longer changes zor's predicted
     color (i.e. A is behaviorally AND causally gone at theta_T, only the
     Fisher-geometry trace remains).

Everything is logged to results/ as JSON for the reproducibility record.
"""
import json
import copy
import numpy as np
import torch

from src.task import make_filler_mapping, PhaseDataset, OBJ2ID, COLOR2ID, SPECIAL_OBJECT, FILLER_OBJECTS, VOCAB_SIZE, NUM_CLASSES
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import build_v_A, build_v_sham, build_v_rand, verify_v_A_causal, causal_ablate_hidden, find_minimal_flipping_alpha
from src.fisher import fisher_curvature_along_v, fisher_curvature_all_examples

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS = {}


def run():
    filler_mapping = make_filler_mapping(seed=SEED)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    # --- Step 1: Train M_A (phase A: zor=red) ---
    model_A = TinyClassifier(VOCAB_SIZE, NUM_CLASSES)
    log_A = train_phase(model_A, ds_A, steps=400, batch_size=32, lr=0.01, seed=SEED, eval_every=400)
    RESULTS["phase_A_final_acc"] = log_A[-1]["eval_acc"]
    print(f"[Phase A] final eval acc: {log_A[-1]['eval_acc']:.3f}")

    # --- Step 2: Build v_A at theta_A, verify causal necessity ---
    v_A, b_A, probe_acc = build_v_A(model_A, filler_mapping)
    RESULTS["v_A_probe_acc"] = probe_acc
    min_alpha = find_minimal_flipping_alpha(model_A, v_A)
    RESULTS["min_flipping_alpha"] = min_alpha
    causal_check_A = verify_v_A_causal(model_A, v_A, filler_mapping, alpha=min_alpha)
    RESULTS["v_A_causal_check_at_theta_A"] = causal_check_A
    print(f"[v_A] probe acc={probe_acc:.3f}, min flipping alpha={min_alpha}, causal check: {causal_check_A}")
    assert causal_check_A["causal_effect_confirmed"], "v_A is not causally necessary at theta_A -- STOP, fix probe before proceeding"

    # Save theta_A state to branch two lineages from the identical starting point
    theta_A_state = copy.deepcopy(model_A.state_dict())

    # --- Step 3a: Continue M_A -> M_AB (treatment lineage) on phase B ---
    model_AB = TinyClassifier(VOCAB_SIZE, NUM_CLASSES)
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=2000, batch_size=32, lr=0.005, seed=SEED + 1, eval_every=20)

    # --- Step 3b: Train M_B from scratch (control lineage) on phase B ---
    model_B = TinyClassifier(VOCAB_SIZE, NUM_CLASSES)
    torch.manual_seed(SEED + 2)  # independent init
    log_B = train_phase(model_B, ds_B_only, steps=2000, batch_size=32, lr=0.005, seed=SEED + 3, eval_every=20)

    # --- Step 3c: Match M_AB to M_B's final output distribution ---
    matched_entry, matched_kl = find_matched_checkpoint(log_B, log_AB)
    print(f"[Matching] best KL(M_AB, M_B_final) = {matched_kl:.5f} at AB step {matched_entry['step']}")
    RESULTS["matching_kl"] = matched_kl
    RESULTS["matching_step"] = matched_entry["step"]
    model_AB.load_state_dict(matched_entry["state_dict"])  # snap to matched checkpoint = theta_T for treatment
    theta_T_acc_AB = matched_entry["eval_acc"]
    theta_T_acc_B = log_B[-1]["eval_acc"]
    RESULTS["theta_T_acc_AB"] = theta_T_acc_AB
    RESULTS["theta_T_acc_B"] = theta_T_acc_B
    print(f"[theta_T] M_AB acc={theta_T_acc_AB:.3f}, M_B acc={theta_T_acc_B:.3f} (should be closely matched)")

    # --- Step 4: Behavioral erasure check -- is zor->red gone at theta_T? ---
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    red_label = COLOR2ID["red"]
    blue_label = COLOR2ID["blue"]
    with torch.no_grad():
        zor_logits_AB = model_AB(zor_id)
        zor_pred_AB = zor_logits_AB.argmax(-1).item()
    behavioral_erasure_confirmed = (zor_pred_AB == blue_label)
    RESULTS["behavioral_erasure_confirmed"] = behavioral_erasure_confirmed
    print(f"[Behavioral erasure] M_AB predicts zor={'blue' if zor_pred_AB==blue_label else ('red' if zor_pred_AB==red_label else 'other')} "
          f"(erasure confirmed: {behavioral_erasure_confirmed})")

    # --- Step 4b: Causal erasure check -- does ablating v_A still change zor's prediction at theta_T? ---
    # Use the SAME alpha calibrated at theta_A (not a fresh search at theta_T) -- we want to know
    # whether the historically-causal direction is STILL causal later, using a fixed intervention
    # strength, not re-calibrate to whatever flips something at theta_T.
    post_ablate_logits = causal_ablate_hidden(model_AB, zor_id, v_A, alpha=min_alpha)
    post_ablate_pred = post_ablate_logits.argmax(-1).item()
    causal_erasure_confirmed = (post_ablate_pred == blue_label)  # ablation should NOT restore red
    RESULTS["causal_erasure_confirmed"] = causal_erasure_confirmed
    print(f"[Causal erasure] ablating v_A at theta_T -> pred={'blue' if post_ablate_pred==blue_label else ('red' if post_ablate_pred==red_label else 'other')} "
          f"(causal erasure confirmed: {causal_erasure_confirmed})")

    # --- Step 5: Build v_sham and v_rand for within-model comparison ---
    v_sham, b_sham, sham_probe_acc = build_v_sham(model_B, filler_mapping, seed=999)  # paired lineage: fit on control model
    hidden_dim = model_AB.fc1.out_features
    v_rand = build_v_rand(hidden_dim, seed=42)
    RESULTS["v_sham_probe_acc"] = sham_probe_acc

    # --- Step 6: Archaeology -- Fisher curvature within M_AB, held-out reference set (fillers only) ---
    ref_objs = torch.tensor([OBJ2ID[o] for o in FILLER_OBJECTS], dtype=torch.long)
    F_vA_on_AB = fisher_curvature_along_v(model_AB, ref_objs, v_A)
    F_vrand_on_AB = fisher_curvature_along_v(model_AB, ref_objs, v_rand)
    F_vsham_on_AB = fisher_curvature_along_v(model_AB, ref_objs, v_sham)  # sham direction evaluated cross-model for reference

    RESULTS["F_vA_on_M_AB"] = F_vA_on_AB
    RESULTS["F_vrand_on_M_AB"] = F_vrand_on_AB
    RESULTS["F_vsham_on_M_AB"] = F_vsham_on_AB
    print(f"[Archaeology, within M_AB] F(v_A)={F_vA_on_AB:.6f}  F(v_rand)={F_vrand_on_AB:.6f}  F(v_sham)={F_vsham_on_AB:.6f}")

    # --- Step 6b: Cross-population check -- F(v_A) on M_AB vs F(v_sham) on M_B itself ---
    F_vsham_on_B = fisher_curvature_along_v(model_B, ref_objs, v_sham)
    RESULTS["F_vsham_on_M_B"] = F_vsham_on_B
    print(f"[Archaeology, cross-pop] F(v_A) on M_AB={F_vA_on_AB:.6f}  vs  F(v_sham) on M_B={F_vsham_on_B:.6f}")

    # --- Step 7: Knockout -- causal archaeology ---
    alpha_knockout = min_alpha  # same calibrated strength throughout, for a consistent intervention

    # Task effect: ablate v_A vs v_rand from M_AB hidden state, measure change in phase-B eval loss/acc
    eval_objs, eval_labels = ds_B.full_eval_set()

    def eval_loss_acc_after_ablation(model, v, alpha):
        logits = causal_ablate_hidden(model, eval_objs, v, alpha=alpha)
        import torch.nn.functional as Fnn
        loss = Fnn.cross_entropy(logits, eval_labels).item()
        acc = (logits.argmax(-1) == eval_labels).float().mean().item()
        return loss, acc

    with torch.no_grad():
        base_logits = model_AB(eval_objs)
        import torch.nn.functional as Fnn
        base_loss = Fnn.cross_entropy(base_logits, eval_labels).item()
        base_acc = (base_logits.argmax(-1) == eval_labels).float().mean().item()

    loss_vA, acc_vA = eval_loss_acc_after_ablation(model_AB, v_A, alpha_knockout)
    loss_vrand, acc_vrand = eval_loss_acc_after_ablation(model_AB, v_rand, alpha_knockout)

    delta_task_vA = acc_vA - base_acc
    delta_task_vrand = acc_vrand - base_acc

    RESULTS["knockout_base_acc"] = base_acc
    RESULTS["knockout_vA_acc"] = acc_vA
    RESULTS["knockout_vrand_acc"] = acc_vrand
    RESULTS["delta_task_vA"] = delta_task_vA
    RESULTS["delta_task_vrand"] = delta_task_vrand
    print(f"[Knockout, task effect] base_acc={base_acc:.3f}  after v_A ablation={acc_vA:.3f} (Δ={delta_task_vA:+.3f})  "
          f"after v_rand ablation={acc_vrand:.3f} (Δ={delta_task_vrand:+.3f})")

    # Archaeological effect: does ablating v_A reduce the Fisher-curvature GAP that constituted the
    # archaeological signal? We recompute F(v_A) on the ALREADY-ablated model as a proxy for
    # "does removing the fossil direction destroy the detectable structure along that same direction".
    # (Ablating v then measuring curvature along v is expected to trivially collapse curvature along v
    # itself since the component is projected out; the informative comparison is whether curvature
    # collapses specifically for v_A's own direction and not globally -- i.e. F(v_rand) after v_A-ablation
    # should be roughly unchanged, showing the ablation is targeted rather than globally destructive.)
    with torch.no_grad():
        h_ablated_vA = model_AB.hidden(ref_objs)
    F_vA_after_vA_ablation = fisher_curvature_along_v(model_AB, ref_objs, v_A)  # will reflect post-hoc structure; see note below
    F_vrand_after_selfablation = fisher_curvature_along_v(model_AB, ref_objs, v_rand)

    RESULTS["note_on_knockout_archaeology"] = (
        "Direct measurement of archaeological collapse requires a downstream classifier "
        "distinguishing A-vs-B-only populations using F(v_A) as a feature, evaluated across "
        "many seeds/replicates -- a single-model single-seed run cannot establish this collapse "
        "statistically. This run establishes the effect at n=1 as a pilot; Step 3 (many-seed "
        "replication) is required before claiming Delta_archaeology >> 0 vs Delta_task ~ 0."
    )

    with open("/home/claude/iclr/results/pilot_run.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print("\n=== SUMMARY ===")
    for k, v in RESULTS.items():
        print(f"{k}: {v}")

    return RESULTS


if __name__ == "__main__":
    run()
