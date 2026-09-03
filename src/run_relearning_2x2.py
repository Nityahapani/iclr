"""
Intervention -> recovery -> re-learning: does persistent causal substrate
affect FUTURE learning, given IDENTICAL current behavior?

Reconstruct two M_AB checkpoints (weight-decay condition, where persistence
naturally varies across seeds while behavior is matched) -- HIGH-persistence
(large C_A remaining, seed 1234) and LOW-persistence (C_A near zero, seed
1236) -- plus an M_B (from-scratch) control, all matched in CURRENT
behavior (m~+2, confidently blue). Continue all three on a NEW task (zor
becomes context-independent GREEN, a label none has ever produced) and
compare learning trajectories. Also test a TEMPORARY steering intervention:
low-persistence model, transiently inject its own J_A-parallel component
for the first 50 steps of the new task, then remove it.
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
from src.probe import (jacobian_zor_red_vs_blue, decompose_parallel_orthogonal,
                        calibrate_hidden_to_target_margin)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def zor_margin_green_vs_blue(model, ctx_name="CTX_RED"):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_id)
        return (logits[0, COLOR2ID["green"]] - logits[0, COLOR2ID["blue"]]).item()


def zor_margin_blue_vs_red(model, ctx_name="CTX_RED"):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_id)
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


class PhaseCDataset(PhaseDataset):
    """New task: zor becomes context-independent GREEN, a label it has
    never produced in any prior phase for this lineage. Fillers/vex
    unchanged (still generated via the base PhaseDataset logic)."""
    def _zor_label(self, ctx):
        return "green"


def build_persistence_matched_checkpoints(high_seed=1234, low_seed=1236, control_seed=1234,
                                            target_green_margin=-3.0, search_steps=(1040,)):
    """
    Build the three checkpoints at a FIXED step count (1040, matching the
    original diagnostic), each with genuinely different persistence levels
    (this varies naturally with seed/lineage under weight decay, as already
    established), then use calibrate_hidden_to_target_margin to set ALL
    THREE to the EXACT SAME starting green-vs-blue margin via a minimal
    bias-only adjustment to fc2 (isolated from fc1/hidden representation,
    where the persistence-relevant structure lives). This directly resolves
    the confound found in the pilot runs: differing starting margins are no
    longer possible, since we force them to be numerically identical.
    """
    results = {}

    for label, seed in [("high_persistence", high_seed), ("low_persistence", low_seed)]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        filler_mapping = make_filler_mapping(seed=seed)
        ds_A = PhaseDataset(filler_mapping, phase="A")
        ds_B = PhaseDataset(filler_mapping, phase="B")

        model_A = new_model(bottleneck_dim=2)
        train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
        J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
        theta_A_state = copy.deepcopy(model_A.state_dict())

        model_AB = new_model(bottleneck_dim=2)
        model_AB.load_state_dict(copy.deepcopy(theta_A_state))
        opt = torch.optim.Adam(model_AB.parameters(), lr=0.005, weight_decay=0.1)
        rng = np.random.RandomState(seed + 1)
        for step in range(search_steps[0]):
            objs, ctxs, labels = ds_B.sample_batch(32, rng)
            logits = model_AB(objs, ctxs)
            loss = Fnn.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()

        pre_calibration_margin = zor_margin_green_vs_blue(model_AB)
        calibrate_hidden_to_target_margin(model_AB, SPECIAL_OBJECT, "CTX_RED",
                                            "green", "blue", target_green_margin)
        post_calibration_margin = zor_margin_green_vs_blue(model_AB)

        m_current = zor_margin_blue_vs_red(model_AB)
        h_AB = model_AB.hidden(torch.tensor([OBJ2ID[SPECIAL_OBJECT]]), torch.tensor([CTX2ID["CTX_RED"]])).squeeze(0).detach()
        h_par, h_perp = decompose_parallel_orthogonal(h_AB, J_A)
        with torch.no_grad():
            coeff = (J_A @ h_AB) / ((J_A @ J_A) + 1e-9)
            h_ablated = h_AB - coeff * J_A
            logits_ablated = model_AB.fc2(h_ablated.unsqueeze(0))
            m_ablated = (logits_ablated[0, COLOR2ID["blue"]] - logits_ablated[0, COLOR2ID["red"]]).item()
        C_A_current = m_ablated - m_current

        print(f"[{label}] seed={seed}: pre_cal_green_margin={pre_calibration_margin:.3f} -> "
              f"post_cal={post_calibration_margin:.3f} (target={target_green_margin}), "
              f"C_A={C_A_current:.3f}, ||h_parallel||={h_par.norm().item():.3f}")

        results[label] = {"model": model_AB, "J_A": J_A, "filler_mapping": filler_mapping,
                           "m_current": m_current, "C_A_current": C_A_current, "seed": seed}

    torch.manual_seed(control_seed + 2)
    np.random.seed(control_seed + 2)
    filler_mapping_ctrl = make_filler_mapping(seed=control_seed)
    ds_B_only = PhaseDataset(filler_mapping_ctrl, phase="B_only")
    model_B = new_model(bottleneck_dim=2)
    opt_B = torch.optim.Adam(model_B.parameters(), lr=0.005, weight_decay=0.1)
    rng_B = np.random.RandomState(control_seed + 3)
    for step in range(search_steps[0]):
        objs, ctxs, labels = ds_B_only.sample_batch(32, rng_B)
        logits = model_B(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt_B.zero_grad(); loss.backward(); opt_B.step()

    pre_cal_B = zor_margin_green_vs_blue(model_B)
    calibrate_hidden_to_target_margin(model_B, SPECIAL_OBJECT, "CTX_RED", "green", "blue", target_green_margin)
    post_cal_B = zor_margin_green_vs_blue(model_B)
    m_B_current = zor_margin_blue_vs_red(model_B)
    print(f"[M_B control] pre_cal={pre_cal_B:.3f} -> post_cal={post_cal_B:.3f}, m_blue_vs_red={m_B_current:.3f}")
    results["M_B_control"] = {"model": model_B, "filler_mapping": filler_mapping_ctrl,
                               "m_current": m_B_current}

    return results


def continue_on_phase_C(model, filler_mapping, steps=500, lr=0.005, seed=0, steer_direction=None,
                          steer_steps=0, steer_strength=1.0):
    ds_C = PhaseCDataset(filler_mapping, phase="A")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    trajectory = []
    for step in range(steps):
        objs, ctxs, labels = ds_C.sample_batch(32, rng)

        if step < steer_steps and steer_direction is not None:
            zor_id_val = OBJ2ID[SPECIAL_OBJECT]
            mask = (objs == zor_id_val)
            if mask.any():
                h_full = model.hidden(objs, ctxs)
                h_full = h_full.clone()
                h_full[mask] = h_full[mask] + steer_strength * steer_direction
                logits = model.fc2(h_full)
            else:
                logits = model(objs, ctxs)
        else:
            logits = model(objs, ctxs)

        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 5 == 0 or step == steps - 1:
            m_green = zor_margin_green_vs_blue(model)
            # also record loss on zor specifically (scale-normalized progress
            # measure, robust to differing starting margins across conditions)
            zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
            ctx_id_eval = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
            with torch.no_grad():
                logits_zor = model(zor_id, ctx_id_eval)
                zor_loss = Fnn.cross_entropy(logits_zor, torch.tensor([COLOR2ID["green"]])).item()
            trajectory.append({"step": step, "m_green_vs_blue": m_green, "zor_loss": zor_loss})

    return trajectory


def run_2x2_experiment(high_seed=1234, low_seed=1236, control_seed=1234, run_label="default"):
    print("=" * 60)
    print(f"Building persistence-matched checkpoints (high={high_seed}, low={low_seed})...")
    print("=" * 60)
    checkpoints = build_persistence_matched_checkpoints(high_seed=high_seed, low_seed=low_seed,
                                                          control_seed=control_seed)

    print("\n" + "=" * 60)
    print("Continuing all three on phase C (zor -> green)...")
    print("=" * 60)

    results = {}
    for label in ["high_persistence", "low_persistence", "M_B_control"]:
        model = checkpoints[label]["model"]
        filler_mapping = checkpoints[label]["filler_mapping"]
        model_copy = new_model(bottleneck_dim=2)
        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        traj = continue_on_phase_C(model_copy, filler_mapping, steps=500, seed=42)
        results[label] = traj
        t_green_flip = next((pt["step"] for pt in traj if pt["m_green_vs_blue"] > 0), None)
        print(f"[{label}] green-flip step = {t_green_flip}")

    print("\n--- Temporary steering condition ---")
    low_p = checkpoints["low_persistence"]
    model_steered = new_model(bottleneck_dim=2)
    model_steered.load_state_dict(copy.deepcopy(low_p["model"].state_dict()))
    h_current = model_steered.hidden(torch.tensor([OBJ2ID[SPECIAL_OBJECT]]), torch.tensor([CTX2ID["CTX_RED"]])).squeeze(0).detach()
    h_par, _ = decompose_parallel_orthogonal(h_current, low_p["J_A"])
    traj_steered = continue_on_phase_C(model_steered, low_p["filler_mapping"], steps=500, seed=42,
                                         steer_direction=h_par, steer_steps=50, steer_strength=1.0)
    results["low_persistence_with_steering"] = traj_steered
    t_flip_steered = next((pt["step"] for pt in traj_steered if pt["m_green_vs_blue"] > 0), None)
    print(f"[low_persistence_with_steering] green-flip step = {t_flip_steered}")

    with open(f"/home/claude/iclr/results/relearning_2x2_{run_label}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== SUMMARY ({run_label}): step at which m_green_vs_blue first crosses 0, and loss AUC(0-100) ===")
    summary = {}
    for label, traj in results.items():
        t = next((pt["step"] for pt in traj if pt["m_green_vs_blue"] > 0), None)
        steps_arr = np.array([pt["step"] for pt in traj if pt["step"] <= 100])
        loss_arr = np.array([pt["zor_loss"] for pt in traj if pt["step"] <= 100])
        auc_loss = np.trapezoid(loss_arr, steps_arr)
        m0 = traj[0]["m_green_vs_blue"]
        print(f"{label:35s}: t_green_flip={t}, starting_margin={m0:.3f}, loss_AUC(0-100)={auc_loss:.2f}")
        summary[label] = {"t_green_flip": t, "starting_margin": m0, "loss_AUC": auc_loss}

    return summary


if __name__ == "__main__":
    seed_pairs = [
        (1234, 1236, 1234, "pair1_1234hi_1236lo"),
        (1238, 1236, 1238, "pair2_1238hi_1236lo"),
        (1234, 1237, 1234, "pair3_1234hi_1237lo"),
    ]
    all_summaries = {}
    for high_s, low_s, ctrl_s, label in seed_pairs:
        print("\n\n" + "#" * 70)
        print(f"# SEED PAIR: {label}")
        print("#" * 70)
        summary = run_2x2_experiment(high_seed=high_s, low_seed=low_s, control_seed=ctrl_s, run_label=label)
        all_summaries[label] = summary

    print("\n\n" + "=" * 70)
    print("=== CROSS-PAIR SUMMARY (loss_AUC per condition per pair) ===")
    print("=" * 70)
    for label, summary in all_summaries.items():
        print(f"\n{label}:")
        for cond, vals in summary.items():
            print(f"  {cond:35s}: loss_AUC={vals['loss_AUC']:.2f}")

    with open("/home/claude/iclr/results/relearning_2x2_all_pairs.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
