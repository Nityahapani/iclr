"""
Experiment: Natural Forward-Pass Participation

This experiment distinguishes PRESERVED COMPUTATION from MERELY PRESERVED
COUNTERFACTUAL USEFULNESS.

The existing experiments (patching, ablation) show that J_A can be patched
back in and works. But that only shows counterfactual usefulness, not that
J_A actually participates in the NATURAL, NON-INTERVENED forward pass
during B behavior.

We add a new test: does the model's ACTUAL HIDDEN STATE during natural B
behavior have a substantial component along J_A, AND does that component
causally influence the output even without any intervention?

Three sub-tests:
  1. h·J_A projection magnitude during natural forward pass (is the J_A
     direction "active" / does h have substantial mass along it?)
  2. Readout sensitivity to J_A component (what fraction of the output
     logit difference is explained by h's J_A component during natural pass?)
  3. Trajectory plot: across B-training, track BOTH the ablation effect
     (counterfactual) AND the natural projection of h onto J_A, together
     showing that:
     - J_A-parallel component of h stays large (J_A→intermediate stays live)
     - AND ablation along J_A still changes output (J_A→output still causal)
     This establishes: JA → intermediate computation → output
     remains causally instantiated during ORDINARY B behavior.
"""

import copy
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/iclr")

from src.task import (
    make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
    SPECIAL_OBJECT, SHAM_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
    FILLER_OBJECTS
)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import (
    jacobian_zor_red_vs_blue, cosine_alignment, causal_mediation_effect,
    ablate_along_J, find_B_mechanism_direction, decompose_parallel_orthogonal
)


SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]

BASE_CONFIG = {
    "hidden_dim": 32, "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005,
    "weight_decay": 0.0, "batch_size": 32,
}


def zor_behavior(model):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_red)
        pred = logits.argmax(-1).item()
        margin = (logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]).item()
    return pred == COLOR2ID["red"], margin


def measure_natural_participation(model, J_A, J_B=None):
    """
    Core natural-participation measurement.

    For the natural forward pass on zor@CTX_RED:
    1. h_natural = model.hidden(zor, CTX_RED)   -- actual hidden state
    2. h_parallel = projection of h onto J_A    -- J_A-parallel component
    3. h_perp = remainder
    4. Logit when routing through h_parallel alone (via model.fc2(h_parallel))
    5. Logit when routing through h_perp alone
    6. Full natural logit (baseline)

    This decomposes HOW MUCH the J_A direction contributes to the actual
    output during the natural B-behavior forward pass.

    Returns a dict with all the key quantities.
    """
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    red_idx = COLOR2ID["red"]
    blue_idx = COLOR2ID["blue"]

    J_A_unit = J_A / (J_A.norm() + 1e-9)

    with torch.no_grad():
        h = model.hidden(zor_id, ctx_red).squeeze(0)

        # Full natural logit/margin
        logits_natural = model.fc2(h.unsqueeze(0))
        m_natural = (logits_natural[0, blue_idx] - logits_natural[0, red_idx]).item()
        pred_natural = logits_natural.argmax(-1).item()

        # Decompose h into J_A-parallel and orthogonal
        coeff_JA = h @ J_A_unit  # scalar: magnitude of h's projection onto J_A
        h_parallel = coeff_JA * J_A_unit
        h_perp = h - h_parallel

        # What does the readout say about each component alone?
        logits_parallel = model.fc2(h_parallel.unsqueeze(0))
        m_parallel = (logits_parallel[0, blue_idx] - logits_parallel[0, red_idx]).item()

        logits_perp = model.fc2(h_perp.unsqueeze(0))
        m_perp = (logits_perp[0, blue_idx] - logits_perp[0, red_idx]).item()

        # How much of the natural margin is explained by the J_A-parallel component?
        # (additive decomposition since readout is linear)
        fraction_natural_from_JA = m_parallel / (abs(m_natural) + 1e-9)

        # Ablation effect (counterfactual) for comparison
        logits_ablated = ablate_along_J(model, zor_id, ctx_red, J_A, alpha=1.0)
        m_ablated = (logits_ablated[0, blue_idx] - logits_ablated[0, red_idx]).item()
        ablation_effect = m_ablated - m_natural  # negative = ablating J_A reduces blue prediction

        # h_parallel magnitude (dimensionless: ||h_parallel|| / ||h||)
        h_norm = h.norm().item()
        h_parallel_frac = h_parallel.norm().item() / (h_norm + 1e-9)

        # J_A-direction projection of h (signed scalar)
        JA_projection = coeff_JA.item()

        # J_B analysis if provided
        JB_projection = None
        h_parallel_B_frac = None
        if J_B is not None:
            J_B_unit = J_B / (J_B.norm() + 1e-9)
            coeff_JB = h @ J_B_unit
            h_parallel_B = coeff_JB * J_B_unit
            JB_projection = coeff_JB.item()
            h_parallel_B_frac = h_parallel_B.norm().item() / (h_norm + 1e-9)

    return {
        "m_natural": m_natural,
        "pred_natural": pred_natural,
        "m_parallel": m_parallel,
        "m_perp": m_perp,
        "ablation_effect": ablation_effect,
        "m_ablated": m_ablated,
        "fraction_natural_from_JA": fraction_natural_from_JA,
        "JA_projection": JA_projection,
        "h_parallel_frac": h_parallel_frac,
        "h_norm": h_norm,
        "JB_projection": JB_projection,
        "h_parallel_B_frac": h_parallel_B_frac,
    }


def _train_loop_with_snapshots(model, dataset, lr, weight_decay, steps, batch_size, seed,
                                 eval_every=50):
    """Train and return log with snapshots."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.RandomState(seed)
    log = []
    eval_set = dataset.full_eval_set()
    eval_objs, eval_ctxs, eval_labels = eval_set
    for step in range(steps):
        objs, ctxs, labels = dataset.sample_batch(batch_size, rng)
        logits = model(objs, ctxs)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                eval_logits = model(eval_objs, eval_ctxs)
                eval_probs = F.softmax(eval_logits, dim=-1)
                acc = (eval_logits.argmax(-1) == eval_labels).float().mean().item()
            log.append({
                "step": step, "train_loss": loss.item(), "eval_acc": acc,
                "eval_probs": eval_probs.clone(),
                "state_dict": copy.deepcopy(model.state_dict())
            })
    return log


def run_trajectory_for_seed(seed, verbose=False):
    """
    Run the full trajectory for one seed, tracking natural participation
    AND counterfactual accessibility together at every checkpoint.

    This directly establishes (or refutes):
    JA → intermediate computation → output
    remains causally instantiated during ordinary B behavior.
    """
    cfg = BASE_CONFIG.copy()
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    def new_model():
        return TinyClassifier(
            VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
            embed_dim=cfg["embed_dim"], ctx_embed_dim=cfg["ctx_embed_dim"],
            hidden_dim=cfg["hidden_dim"]
        )

    # Phase A
    model_A = new_model()
    train_phase(model_A, ds_A, steps=cfg["phase_A_steps"], batch_size=cfg["batch_size"],
                lr=cfg["phase_A_lr"], seed=seed, eval_every=cfg["phase_A_steps"])
    is_red_A, margin_A = zor_behavior(model_A)
    if not is_red_A:
        return {"seed": seed, "status": "FAILED_PHASE_A"}

    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    J_A_unit = J_A / (J_A.norm() + 1e-9)
    theta_A_state = copy.deepcopy(model_A.state_dict())

    # Phase B with trajectory snapshots
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = _train_loop_with_snapshots(
        model_AB, ds_B, lr=cfg["phase_B_lr"], weight_decay=cfg["weight_decay"],
        steps=cfg["phase_B_steps"], batch_size=cfg["batch_size"], seed=seed, eval_every=50
    )

    # Find t_flip: first step where behavior flips to blue
    t_flip = None
    trajectory = []

    for entry in log_AB:
        step = entry["step"]
        # Load checkpoint
        m = new_model()
        m.load_state_dict(entry["state_dict"])

        is_red_t, margin_t = zor_behavior(m)
        if t_flip is None and not is_red_t:
            t_flip = step

        # Get J_B at this checkpoint (current B mechanism)
        J_B = find_B_mechanism_direction(m, ctx_name="CTX_RED")

        # Natural participation measurement
        np_metrics = measure_natural_participation(m, J_A, J_B=J_B)

        trajectory.append({
            "step": step,
            "is_red": is_red_t,
            "margin": margin_t,
            "t_flip_is_here": (t_flip == step),
            # Natural participation (non-intervened)
            "JA_projection": np_metrics["JA_projection"],
            "h_parallel_frac": np_metrics["h_parallel_frac"],
            "fraction_natural_from_JA": np_metrics["fraction_natural_from_JA"],
            "m_parallel": np_metrics["m_parallel"],
            "m_perp": np_metrics["m_perp"],
            # Counterfactual accessibility (ablation)
            "ablation_effect": np_metrics["ablation_effect"],
            # J_B comparison
            "JB_projection": np_metrics["JB_projection"],
            "h_parallel_B_frac": np_metrics["h_parallel_B_frac"],
            # Cosine between J_A and current J_B
            "cos_JA_JB": cosine_alignment(J_A, J_B),
        })

        if verbose and step % 500 == 0:
            print(f"  step={step}: is_red={is_red_t} "
                  f"JA_proj={np_metrics['JA_projection']:.3f} "
                  f"ablation_eff={np_metrics['ablation_effect']:.3f} "
                  f"frac_from_JA={np_metrics['fraction_natural_from_JA']:.3f}")

    # Final checkpoint summary
    final_entry = trajectory[-1]
    result = {
        "seed": seed,
        "status": "OK",
        "t_flip": t_flip,
        "final_is_red": final_entry["is_red"],
        "final_JA_projection": final_entry["JA_projection"],
        "final_h_parallel_frac": final_entry["h_parallel_frac"],
        "final_fraction_natural_from_JA": final_entry["fraction_natural_from_JA"],
        "final_ablation_effect": final_entry["ablation_effect"],
        "final_cos_JA_JB": final_entry["cos_JA_JB"],
        "trajectory": trajectory,
    }
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT: Natural Forward-Pass Participation")
    print("Testing whether J_A participates in NATURAL (non-intervened)")
    print("B-behavior computation, not just counterfactual patches.")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        r = run_trajectory_for_seed(seed, verbose=True)
        all_results.append(r)
        if r["status"] == "OK":
            print(f"  t_flip={r['t_flip']}")
            print(f"  Final JA_projection={r['final_JA_projection']:.3f}")
            print(f"  Final h_parallel_frac={r['final_h_parallel_frac']:.3f}")
            print(f"  Final frac_natural_from_JA={r['final_fraction_natural_from_JA']:.3f}")
            print(f"  Final ablation_effect={r['final_ablation_effect']:.3f}")

    # Strip state_dicts for JSON serialization
    def strip(r):
        if "trajectory" in r:
            for t in r["trajectory"]:
                t.pop("state_dict", None)
        return r

    outpath = "/home/claude/iclr/results/natural_participation.json"
    with open(outpath, "w") as f:
        json.dump([strip(r) for r in all_results], f, indent=2)
    print(f"\nSaved to {outpath}")

    # Cross-seed summary
    ok_results = [r for r in all_results if r["status"] == "OK"]
    print("\n=== CROSS-SEED SUMMARY ===")
    print(f"n={len(ok_results)} seeds")
    print(f"Mean t_flip: {np.mean([r['t_flip'] or -1 for r in ok_results]):.1f}")
    print(f"Mean final JA_projection: {np.mean([r['final_JA_projection'] for r in ok_results]):.3f}")
    print(f"Mean final h_parallel_frac: {np.mean([r['final_h_parallel_frac'] for r in ok_results]):.3f}")
    print(f"Mean final frac_natural_from_JA: {np.mean([r['final_fraction_natural_from_JA'] for r in ok_results]):.3f}")
    print(f"Mean final ablation_effect: {np.mean([r['final_ablation_effect'] for r in ok_results]):.3f}")
    print()
    print("KEY QUESTION: Is JA→h→output causal chain live during natural B behavior?")
    print("Evidence for YES:")
    print(f"  - h has mass along J_A: {np.mean([r['final_h_parallel_frac'] for r in ok_results]):.3f} fraction of ||h||")
    print(f"  - Ablation along J_A changes output: {np.mean([abs(r['final_ablation_effect']) for r in ok_results]):.3f} margin change")
    print(f"  - J_A-parallel component explains {np.mean([r['final_fraction_natural_from_JA'] for r in ok_results]):.3f} of natural margin")
