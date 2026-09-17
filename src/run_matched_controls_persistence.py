"""
Experiment: Matched Controls Persistence

The objection: persistence of J_A might follow trivially from how it was
selected — any direction with the same initial ablation effect and readout
alignment would persist equally. If so, 'A-specific' adds nothing.

Key structural fact: J_A = W_red - W_blue exactly (cos = 1.000000 in all
seeds). This means matching readout alignment = matching the direction itself.
Genuinely different matched controls must therefore be constructed by one of:
  (a) Being derived from a DIFFERENT model state (different training history)
      but matched to have the same initial ablation effect on mAB at t=0
  (b) Being orthogonal to J_A by construction, then scaled to match the
      initial ablation effect — testing whether persistence is specific to
      J_A's orientation vs any same-scale direction
  (c) Tracking how matched controls DIVERGE from J_A during B-training
      as J_A's causal effect grows — persistence is A-specific if J_A's
      trajectory differs from matched controls with identical t=0 properties

Design: at each B-training checkpoint, measure four directions:
  J_A:     frozen at mA (the historical direction)
  D_Bgrad: the gradient direction at the CURRENT checkpoint (tracks where
           B-training is pushing things — exactly matched to J_A at t=0
           before B-training starts, then diverges if A-specific)
  D_rand_matched: 50 random directions each scaled so their initial
           ablation effect on h_AB at t=0 equals J_A's initial effect.
           These are the matched-geometry controls.
  D_filler_non_red: Jacobian for non-red fillers at mA — same readout,
           different object, different A-content.

For each direction and each checkpoint, measure:
  1. frac_remaining = |abl_delta(t)| / |abl_delta_ref|
     where abl_delta_ref is measured at the SAME t=0 for all directions
     (so they start equal and we track divergence)
  2. h·d projection (natural participation)
  3. cos(d, J_A) to track geometric drift

The test: if persistence is A-specific, J_A's frac_remaining should
be LARGER than matched controls' frac_remaining at t > 0, even though
all controls have identical frac_remaining = 1.0 at t = 0.

If persistence is just a geometry artifact, all directions should decay
at the same rate.
"""

import copy
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, "/home/claude/iclr")

from src.task import (
    make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
    SPECIAL_OBJECT, FILLER_OBJECTS, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import jacobian_zor_red_vs_blue, jacobian_of_margin, cosine_alignment

SEEDS          = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
N_RAND_MATCHED = 50     # random matched directions per seed
RECORD_EVERY   = 100    # checkpoint interval
CFG = dict(hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
           phase_A_steps=600, phase_A_lr=0.01,
           phase_B_steps=3000, phase_B_lr=0.005, batch_size=32)


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                          embed_dim=CFG["embed_dim"],
                          ctx_embed_dim=CFG["ctx_embed_dim"],
                          hidden_dim=CFG["hidden_dim"])


def abl_effect(model, h, direction, blue_cls, red_cls):
    """blue-red margin change when ablating direction from h. Linear — exact."""
    d_unit = direction / (direction.norm() + 1e-9)
    with torch.no_grad():
        l_nat  = model.fc2(h.unsqueeze(0))
        m_nat  = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()
        coeff  = (h @ d_unit).item()
        h_abl  = h - coeff * d_unit
        l_abl  = model.fc2(h_abl.unsqueeze(0))
        m_abl  = (l_abl[0, blue_cls] - l_abl[0, red_cls]).item()
    return m_abl - m_nat, m_nat, coeff


def make_random_matched_directions(J_A_unit, h_nat_t0, model_t0,
                                   target_abl_effect, blue_cls, red_cls,
                                   n, seed):
    """
    Construct n random directions, each scaled so that its initial ablation
    effect on h_nat_t0 matches target_abl_effect.

    Method: sample random unit direction d, compute its ablation effect
    (which equals (W_blue-W_red) · (coeff_d * d) by linearity of fc2),
    then scale d so its projection onto h gives the right coefficient.

    Since abl_effect = (W_B - W_R) · (coeff * d_unit)
                     = coeff * cos(d, J_A) * ||J_A||    [when J_A=W_R-W_B]
    we scale coeff = target_abl_effect / (||J_A|| * cos(d, J_A_unit))
    i.e. we scale the direction so its projection onto h gives the right abl.

    Returns list of (direction_tensor, cos_with_JA) pairs.
    """
    rng = torch.Generator().manual_seed(seed + 99999)
    dim = J_A_unit.shape[0]
    W   = model_t0.fc2.weight
    readout_dir = W[blue_cls] - W[red_cls]   # = -J_A direction

    directions = []
    attempts   = 0
    while len(directions) < n and attempts < n * 20:
        attempts += 1
        # Random unit vector
        v = torch.randn(dim, generator=rng)
        v = v / (v.norm() + 1e-9)

        # Compute its natural ablation effect on h at t=0
        eff_v, _, coeff_v = abl_effect(model_t0, h_nat_t0, v, blue_cls, red_cls)

        if abs(eff_v) < 1e-4:
            continue  # degenerate direction, skip

        # Scale v so that its ablation effect equals target_abl_effect.
        # By linearity: abl_effect(alpha*v) = alpha * abl_effect(v)
        # so alpha = target / eff_v
        alpha = target_abl_effect / eff_v
        v_scaled = alpha * v    # NOT unit-norm; scaled to match initial effect

        # Verify (exact by linearity)
        eff_check, _, _ = abl_effect(model_t0, h_nat_t0, v_scaled, blue_cls, red_cls)

        if abs(eff_check - target_abl_effect) > abs(target_abl_effect) * 0.01:
            continue  # scaling failed, skip

        cos_vJA = float(cosine_alignment(v, J_A_unit))
        directions.append((v_scaled, cos_vJA))

    return directions


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    ds_B = PhaseDataset(fm, "B")

    blue_cls = COLOR2ID["blue"]
    red_cls  = COLOR2ID["red"]
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    # Phase A
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"],
                batch_size=CFG["batch_size"], lr=CFG["phase_A_lr"],
                seed=seed, eval_every=CFG["phase_A_steps"])
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    J_A      = jacobian_zor_red_vs_blue(mA, "CTX_RED")
    J_A_unit = J_A / (J_A.norm() + 1e-9)

    # Non-red filler Jacobians at mA (different A-content, same readout)
    non_red_fillers = [o for o in FILLER_OBJECTS if fm[o] != "red"]
    filler_dirs = []
    for obj in non_red_fillers[:6]:
        true_cls = COLOR2ID[fm[obj]]
        j = jacobian_of_margin(mA,
                               torch.tensor([OBJ2ID[obj]], dtype=torch.long),
                               cid, true_cls, blue_cls)
        filler_dirs.append((obj, fm[obj], j / (j.norm() + 1e-9)))

    # Phase B — track trajectory
    mAB = new_model()
    mAB.load_state_dict(copy.deepcopy(mA.state_dict()))
    opt = torch.optim.Adam(mAB.parameters(), lr=CFG["phase_B_lr"])
    rng = np.random.RandomState(seed)

    trajectory = []
    t0_ref = None   # reference ablation effects at t=0 (all matched here)

    for step in range(CFG["phase_B_steps"] + 1):

        if step % RECORD_EVERY == 0 or step == CFG["phase_B_steps"]:

            with torch.no_grad():
                h_nat = mAB.hidden(zid, cid).squeeze(0)
                l_nat = mAB.fc2(h_nat.unsqueeze(0))
                m_nat = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()
                pred_blue = l_nat.argmax(-1).item() == blue_cls

            # ── J_A ablation effect at this checkpoint ─────────────────────
            abl_JA, _, proj_JA = abl_effect(mAB, h_nat, J_A_unit, blue_cls, red_cls)

            # ── B-gradient direction at this checkpoint ────────────────────
            # The direction B-training pushes h — matched to J_A at t=0
            # (because at step 0, B gradient = cross-entropy grad toward blue
            #  = W_blue - W_red = J_A direction)
            mAB.zero_grad()
            h_for_grad = mAB.hidden(zid, cid)
            h_for_grad.retain_grad()
            loss_zor = F.cross_entropy(mAB.fc2(h_for_grad),
                                       torch.tensor([blue_cls]))
            loss_zor.backward()
            grad_h = h_for_grad.grad.squeeze(0).detach().clone()
            mAB.zero_grad()
            D_Bgrad_unit = grad_h / (grad_h.norm() + 1e-9)
            abl_Bgrad, _, proj_Bgrad = abl_effect(mAB, h_nat, D_Bgrad_unit,
                                                   blue_cls, red_cls)
            cos_Bgrad_JA = float(cosine_alignment(D_Bgrad_unit, J_A_unit))

            # ── Filler non-A Jacobian directions ──────────────────────────
            filler_effects = []
            for obj, color, fdir in filler_dirs:
                abl_f, _, proj_f = abl_effect(mAB, h_nat, fdir,
                                               blue_cls, red_cls)
                cos_f_JA = float(cosine_alignment(fdir, J_A_unit))
                filler_effects.append({
                    "obj": obj, "color": color,
                    "abl_delta": round(abl_f, 4),
                    "projection": round(proj_f, 4),
                    "cos_with_JA": round(cos_f_JA, 4),
                })

            # ── At t=0: set reference effects and build matched directions ─
            if step == 0:
                ref_JA     = abs(abl_JA)
                ref_Bgrad  = abs(abl_Bgrad)

                # Build N_RAND_MATCHED directions matched to J_A's initial effect
                target_eff = abl_JA   # signed (negative = disrupts blue)
                rand_dirs  = make_random_matched_directions(
                    J_A_unit, h_nat, mAB, target_eff,
                    blue_cls, red_cls, N_RAND_MATCHED, seed
                )
                t0_ref = {
                    "abl_JA":    abl_JA,
                    "abl_Bgrad": abl_Bgrad,
                    "n_rand_matched": len(rand_dirs),
                }

            # ── Matched random direction effects ──────────────────────────
            rand_abls = []
            rand_projs = []
            rand_cos_JA = []
            for d_r, cos_r in rand_dirs:
                a_r, _, p_r = abl_effect(mAB, h_nat, d_r, blue_cls, red_cls)
                rand_abls.append(a_r)
                rand_projs.append(p_r)
                rand_cos_JA.append(cos_r)

            # frac_remaining for each direction = |abl(t)| / |abl(t0)|
            # For J_A and matched: all start at same ref (t0_ref["abl_JA"])
            ref  = abs(t0_ref["abl_JA"]) if t0_ref else 1.0
            reff = abs(t0_ref["abl_Bgrad"]) if t0_ref else 1.0

            frac_JA    = abs(abl_JA)    / (ref  + 1e-9)
            frac_Bgrad = abs(abl_Bgrad) / (reff + 1e-9)
            frac_rand_mean = float(np.mean([abs(a) for a in rand_abls])) / (ref + 1e-9)
            frac_rand_std  = float(np.std( [abs(a) for a in rand_abls])) / (ref + 1e-9)

            # Filler fracs (using their own t=0 as ref — computed separately)
            filler_fracs = []
            for fe in filler_effects:
                filler_fracs.append(fe["abl_delta"])

            record = {
                "step": step,
                "pred_blue": bool(pred_blue),
                "m_nat": round(m_nat, 4),
                # J_A
                "abl_JA":      round(abl_JA, 4),
                "proj_JA":     round(proj_JA, 4),
                "frac_JA":     round(frac_JA, 4),
                # B-gradient direction
                "abl_Bgrad":   round(abl_Bgrad, 4),
                "proj_Bgrad":  round(proj_Bgrad, 4),
                "frac_Bgrad":  round(frac_Bgrad, 4),
                "cos_Bgrad_JA":round(cos_Bgrad_JA, 4),
                # Matched random
                "mean_abl_rand":  round(float(np.mean(rand_abls)), 4),
                "frac_rand_mean": round(frac_rand_mean, 4),
                "frac_rand_std":  round(frac_rand_std, 4),
                "mean_proj_rand": round(float(np.mean(rand_projs)), 4),
                # Filler directions
                "filler_effects": filler_effects,
            }
            trajectory.append(record)

        # ── B-training step ────────────────────────────────────────────────
        if step < CFG["phase_B_steps"]:
            o, c, l = ds_B.sample_batch(CFG["batch_size"], rng)
            loss = F.cross_entropy(mAB(o, c), l)
            opt.zero_grad(); loss.backward(); opt.step()

    # ── Compute filler frac_remaining using each filler's t=0 abl as ref ──
    if trajectory:
        filler_names = [fe["obj"] for fe in trajectory[0]["filler_effects"]]
        filler_ref   = {fe["obj"]: abs(fe["abl_delta"])
                        for fe in trajectory[0]["filler_effects"]}
        for rec in trajectory:
            for fe in rec["filler_effects"]:
                ref_f = filler_ref.get(fe["obj"], 1.0)
                fe["frac_remaining"] = round(abs(fe["abl_delta"]) / (ref_f + 1e-9), 4)

    # ── Summary statistics ─────────────────────────────────────────────────
    # At the final checkpoint, compare frac_remaining of J_A vs matched controls
    final = trajectory[-1] if trajectory else {}
    t_flip = next((r["step"] for r in trajectory if r["pred_blue"]), None)

    # After-flip trajectory: compare J_A vs matched controls post-flip
    post_flip = [r for r in trajectory if r["step"] >= (t_flip or 0)]

    mean_frac_JA_post    = float(np.mean([r["frac_JA"]        for r in post_flip])) if post_flip else None
    mean_frac_rand_post  = float(np.mean([r["frac_rand_mean"]  for r in post_flip])) if post_flip else None
    mean_frac_Bgrad_post = float(np.mean([r["frac_Bgrad"]      for r in post_flip])) if post_flip else None

    filler_frac_post = {}
    for fe_name in filler_names:
        vals = [fe["frac_remaining"]
                for r in post_flip
                for fe in r["filler_effects"]
                if fe["obj"] == fe_name]
        filler_frac_post[fe_name] = round(float(np.mean(vals)), 4) if vals else None

    return {
        "seed": seed, "status": "OK",
        "t_flip": t_flip,
        "n_rand_matched": t0_ref["n_rand_matched"] if t0_ref else 0,
        "t0_ref": t0_ref,
        # Post-flip means
        "mean_frac_JA_post_flip":    round(mean_frac_JA_post,    4) if mean_frac_JA_post    is not None else None,
        "mean_frac_rand_post_flip":  round(mean_frac_rand_post,  4) if mean_frac_rand_post  is not None else None,
        "mean_frac_Bgrad_post_flip": round(mean_frac_Bgrad_post, 4) if mean_frac_Bgrad_post is not None else None,
        "filler_frac_post_flip": filler_frac_post,
        # Final checkpoint
        "final_frac_JA":    final.get("frac_JA"),
        "final_frac_rand":  final.get("frac_rand_mean"),
        "final_frac_Bgrad": final.get("frac_Bgrad"),
        "final_cos_Bgrad_JA": final.get("cos_Bgrad_JA"),
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Matched Controls Persistence")
    print("Tracking J_A vs property-matched non-A directions through B-training")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(seed)
        all_results.append(r)

        if r["status"] != "OK":
            print(f"  {r['status']}"); continue

        print(f"  t_flip={r['t_flip']}  n_rand={r['n_rand_matched']}")
        print(f"  POST-FLIP mean frac_remaining:")
        print(f"    J_A:              {r['mean_frac_JA_post_flip']:.3f}")
        print(f"    rand (matched):   {r['mean_frac_rand_post_flip']:.3f}")
        print(f"    B-grad (matched): {r['mean_frac_Bgrad_post_flip']:.3f}")
        print(f"  FINAL frac_remaining:")
        print(f"    J_A:              {r['final_frac_JA']:.3f}")
        print(f"    rand (matched):   {r['final_frac_rand']:.3f}")
        print(f"    B-grad:           {r['final_frac_Bgrad']:.3f}  cos(Bgrad,JA)={r['final_cos_Bgrad_JA']:.3f}")
        for obj, frac in r["filler_frac_post_flip"].items():
            print(f"    filler {obj}:     {frac:.3f}")

    # Strip trajectory from JSON output (too large)
    for r in all_results:
        if "trajectory" in r:
            # Keep only key checkpoints
            r["trajectory_summary"] = [
                {k: v for k, v in t.items() if k != "filler_effects"}
                for t in r["trajectory"]
                if t["step"] in {0, 50, 100, 500, 1000, 1500, 2000, 2500, 3000}
            ]
            del r["trajectory"]

    with open("/home/claude/iclr/results/matched_controls_persistence.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/matched_controls_persistence.json")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== AGGREGATE (n={len(ok)}) ===")

    frac_JA   = [r["final_frac_JA"]    for r in ok]
    frac_rand = [r["final_frac_rand"]  for r in ok]
    frac_Bg   = [r["final_frac_Bgrad"] for r in ok]
    gaps_JA_rand = [j - r for j, r in zip(frac_JA, frac_rand)]
    gaps_JA_Bg   = [j - b for j, b in zip(frac_JA, frac_Bg)]

    t_JR, p_JR = stats.ttest_rel(frac_JA, frac_rand)
    t_JB, p_JB = stats.ttest_rel(frac_JA, frac_Bg)
    d_JR = np.mean(gaps_JA_rand) / (np.std(gaps_JA_rand, ddof=1) + 1e-9)
    d_JB = np.mean(gaps_JA_Bg)   / (np.std(gaps_JA_Bg,   ddof=1) + 1e-9)

    np.random.seed(0)
    ci_JR = np.percentile(
        [np.mean(np.random.choice(gaps_JA_rand, len(ok), replace=True)) for _ in range(10000)],
        [2.5, 97.5])
    ci_JB = np.percentile(
        [np.mean(np.random.choice(gaps_JA_Bg, len(ok), replace=True)) for _ in range(10000)],
        [2.5, 97.5])

    print(f"\nFinal frac_remaining (|abl(T)| / |abl(t=0)|):")
    print(f"  J_A:              {np.mean(frac_JA):.3f} +/- {np.std(frac_JA):.3f}")
    print(f"  rand (matched):   {np.mean(frac_rand):.3f} +/- {np.std(frac_rand):.3f}")
    print(f"  B-grad (matched): {np.mean(frac_Bg):.3f} +/- {np.std(frac_Bg):.3f}")
    print(f"\nJ_A vs rand matched:  gap={np.mean(gaps_JA_rand):+.3f}  "
          f"t={t_JR:.3f}  p={p_JR:.6f}  d={d_JR:.3f}  CI=[{ci_JR[0]:.3f},{ci_JR[1]:.3f}]")
    print(f"J_A vs B-grad:        gap={np.mean(gaps_JA_Bg):+.3f}  "
          f"t={t_JB:.3f}  p={p_JB:.6f}  d={d_JB:.3f}  CI=[{ci_JB[0]:.3f},{ci_JB[1]:.3f}]")

    # Filler fracs
    all_filler_fracs = []
    for r in ok:
        all_filler_fracs.extend(r["filler_frac_post_flip"].values())
    frac_filler = [f for f in all_filler_fracs if f is not None]
    gaps_JA_filler = [j - f for j, f in zip(
        [r["mean_frac_JA_post_flip"] for r in ok for _ in r["filler_frac_post_flip"]],
        frac_filler)]
    print(f"  filler directions:{np.mean(frac_filler):.3f} +/- {np.std(frac_filler):.3f}  "
          f"(n={len(frac_filler)})")

    print(f"\nPost-flip mean frac_remaining:")
    print(f"  J_A:              {np.mean([r['mean_frac_JA_post_flip']    for r in ok]):.3f}")
    print(f"  rand (matched):   {np.mean([r['mean_frac_rand_post_flip']  for r in ok]):.3f}")
    print(f"  B-grad:           {np.mean([r['mean_frac_Bgrad_post_flip'] for r in ok]):.3f}")
