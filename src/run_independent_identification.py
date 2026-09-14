"""
Experiment: Genuinely Independent Identification

Addresses the circularity objection:
  "You defined J_A from the red-vs-blue margin at zor, then showed it
   recovers the red-vs-blue margin at zor. That's not identification —
   it's just recovering your own definition."

Fix: identify the direction using SOURCE A (one set of objects/criteria/
features), test whether it recovers behavior on TARGET B (a completely
disjoint set with different objects, different output dimensions, different
measurement). If source-derived direction predicts target behavior, it
carries genuine A-computational information.

Four independent identification strategies, each with a strictly disjoint
source → target split:

STRATEGY 1 — Object split
  Source: derive direction from red-labeled FILLER objects only (Jacobians
          of their red-vs-non-red margins). Never uses zor.
  Target: test whether that direction predicts zor's behavior in mAB.
          Ablating source-derived direction should disrupt zor's B-output.
  Disjoint: source objects ≠ target object (fillers vs zor).

STRATEGY 2 — Output dimension split
  Source: derive direction from the red-vs-GREEN margin at zor (uses zor,
          but a completely different pair of output logits: red vs green,
          not red vs blue). The green dimension was never used in any prior
          experiment.
  Target: test whether that direction predicts the red-vs-BLUE behavior
          in mAB (i.e., the B-output along a different output axis).
  Disjoint: source output pair (red,green) ≠ target output pair (red,blue).

STRATEGY 3 — Context split
  Source: derive direction from zor's hidden state under CTX_BLUE context
          (Jacobian of red-vs-blue at zor,CTX_BLUE — the OTHER context).
  Target: test whether it predicts zor's B-behavior under CTX_RED.
  Disjoint: source context ≠ target context.
  Note: this is the D3 direction from the specificity experiment, now
  used as a pure identification experiment with explicit disjoint testing.

STRATEGY 4 — Phase split (the hardest test)
  Source: derive direction from an EARLY A-training checkpoint (step 100,
          before phase A has converged fully), using only the FILLER
          objects at that checkpoint. Never sees zor; never sees the
          converged A model.
  Target: test whether that early-filler-derived direction predicts zor's
          B-behavior in the final fully-trained mAB.
  Disjoint: source phase (early A), source objects (fillers), source model
            (not converged) are all disjoint from the target (final mAB, zor).

For each strategy, we measure:
  (a) Ablation effect: does ablating source-derived direction from mAB's
      hidden state disrupt B output? (should be ~abl_delta_D1 if independent)
  (b) cos alignment with D1 (the standard J_A): are they pointing the same way?
  (c) Natural participation: h_AB · d_source — is the direction active?
  (d) Specificity: ablation effect vs matched-norm orthogonal control.
  (e) Cross-object generalization (Strategy 1): does filler-derived direction
      predict zor behavior, and does zor-derived direction predict filler behavior?
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
    SPECIAL_OBJECT, SHAM_OBJECT, FILLER_OBJECTS,
    VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import jacobian_of_margin, cosine_alignment

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG = dict(
    hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
    phase_A_steps=600, phase_A_lr=0.01,
    phase_B_steps=3000, phase_B_lr=0.005, batch_size=32,
)
EARLY_CHECKPOINT = 100   # for Strategy 4


def new_model():
    return TinyClassifier(
        VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
        embed_dim=CFG["embed_dim"], ctx_embed_dim=CFG["ctx_embed_dim"],
        hidden_dim=CFG["hidden_dim"],
    )


def get_jacobian(model, obj_name, ctx_name, tgt_cls, ref_cls):
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    return jacobian_of_margin(model, oid, cid, tgt_cls, ref_cls)


def ablation_metrics(model, h_nat, direction, blue_cls, red_cls, seed):
    """
    Core metric bundle: ablate `direction` from h_nat, measure disruption.
    Also compute orthogonal matched-norm control.
    Returns dict.
    """
    d_unit = direction / (direction.norm() + 1e-9)

    with torch.no_grad():
        l_nat   = model.fc2(h_nat.unsqueeze(0))
        m_nat   = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()

        coeff   = (h_nat @ d_unit).item()
        h_abl   = h_nat - coeff * d_unit
        l_abl   = model.fc2(h_abl.unsqueeze(0))
        m_abl   = (l_abl[0, blue_cls] - l_abl[0, red_cls]).item()
        abl_delta = m_abl - m_nat

        # Natural participation
        h_norm     = h_nat.norm().item()
        projection = coeff
        h_par_frac = abs(coeff) / (h_norm + 1e-9)

        # Fraction of natural margin from this direction (linear readout)
        W   = model.fc2.weight
        rd  = W[blue_cls] - W[red_cls]
        frac_margin = (rd @ (coeff * d_unit)).item() / (m_nat + 1e-9)

        # Orthogonal control: same displacement norm, random perpendicular direction
        g     = torch.Generator().manual_seed(seed + 44444)
        v_rnd = torch.randn(h_nat.shape[0], generator=g)
        v_rnd = v_rnd - (v_rnd @ d_unit) * d_unit
        v_rnd = v_rnd / (v_rnd.norm() + 1e-9) * abs(coeff)
        h_ctrl  = h_nat - (h_nat @ (v_rnd / (v_rnd.norm() + 1e-9))).item() * (v_rnd / (v_rnd.norm() + 1e-9))
        l_ctrl  = model.fc2(h_ctrl.unsqueeze(0))
        m_ctrl  = (l_ctrl[0, blue_cls] - l_ctrl[0, red_cls]).item()
        ctrl_delta = m_ctrl - m_nat

    spec = abs(abl_delta) / (abs(ctrl_delta) + 1e-9)
    return {
        "m_nat":       round(m_nat,       4),
        "m_abl":       round(m_abl,       4),
        "abl_delta":   round(abl_delta,   4),
        "ctrl_delta":  round(ctrl_delta,  4),
        "specificity": round(spec,        2),
        "projection":  round(projection,  4),
        "h_par_frac":  round(h_par_frac,  4),
        "frac_margin": round(frac_margin, 4),
    }


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    ds_B = PhaseDataset(fm, "B")

    red_cls   = COLOR2ID["red"]
    blue_cls  = COLOR2ID["blue"]
    green_cls = COLOR2ID["green"]

    # ── Phase A — full training ───────────────────────────────────────────────
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])

    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid_red  = torch.tensor([CTX2ID["CTX_RED"]],  dtype=torch.long)
    cid_blue = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)

    with torch.no_grad():
        if mA(zid, cid_red).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    # ── Phase A — EARLY checkpoint (Strategy 4) ───────────────────────────────
    mA_early = new_model()
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt_early = torch.optim.Adam(mA_early.parameters(), lr=CFG["phase_A_lr"])
    rng_early = np.random.RandomState(seed)
    for step in range(EARLY_CHECKPOINT):
        o, c, l = ds_A.sample_batch(CFG["batch_size"], rng_early)
        loss = F.cross_entropy(mA_early(o, c), l)
        opt_early.zero_grad(); loss.backward(); opt_early.step()

    # ── Phase B ───────────────────────────────────────────────────────────────
    mAB = new_model()
    mAB.load_state_dict(copy.deepcopy(mA.state_dict()))
    opt = torch.optim.Adam(mAB.parameters(), lr=CFG["phase_B_lr"])
    rng = np.random.RandomState(seed)
    for _ in range(CFG["phase_B_steps"]):
        o, c, l = ds_B.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(mAB(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        if mAB(zid, cid_red).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # Standard D1 baseline (used only for alignment comparison, NOT for target)
    D1 = get_jacobian(mA, SPECIAL_OBJECT, "CTX_RED", red_cls, blue_cls)
    D1_unit = D1 / (D1.norm() + 1e-9)

    with torch.no_grad():
        h_AB = mAB.hidden(zid, cid_red).squeeze(0)

    results = {"seed": seed, "status": "OK"}

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 1 — Object split
    # Source: Jacobians at RED-LABELED FILLER objects (never uses zor)
    # Target: zor's B-output in mAB
    # ══════════════════════════════════════════════════════════════════════════
    red_fillers = [o for o in FILLER_OBJECTS if fm[o] == "red"]
    non_red_fillers = [o for o in FILLER_OBJECTS if fm[o] != "red"]

    if red_fillers:
        # Average Jacobian across all red fillers (source = fillers only)
        jacs_source = []
        for obj in red_fillers:
            j = get_jacobian(mA, obj, "CTX_RED", red_cls, blue_cls)
            jacs_source.append(j)
        D_filler_src = torch.stack(jacs_source).mean(0)
        D_filler_src = D_filler_src / (D_filler_src.norm() + 1e-9)

        # Test on TARGET: zor's B-output
        s1_zor = ablation_metrics(mAB, h_AB, D_filler_src, blue_cls, red_cls, seed)
        cos_D1_S1 = float(cosine_alignment(D1_unit, D_filler_src))

        # Cross-object reverse: derive from ZOR, test on a held-out filler
        # Use the FIRST non-red filler as the held-out test object
        if non_red_fillers:
            held_out = non_red_fillers[0]
            held_cls  = COLOR2ID[fm[held_out]]
            held_ref  = blue_cls if held_cls != blue_cls else red_cls
            with torch.no_grad():
                h_held = mAB.hidden(
                    torch.tensor([OBJ2ID[held_out]], dtype=torch.long), cid_red
                ).squeeze(0)
            s1_held = ablation_metrics(mAB, h_held, D1_unit, held_cls, held_ref, seed)
        else:
            s1_held = None

        results["strategy_1"] = {
            "n_red_fillers": len(red_fillers),
            "cos_D1_S1": round(cos_D1_S1, 4),
            # Primary test: filler-derived direction → zor target
            "filler_to_zor": s1_zor,
            # Reverse: zor-derived direction → filler target (cross-object)
            "zor_to_filler": s1_held,
            "held_out_filler": non_red_fillers[0] if non_red_fillers else None,
        }
    else:
        results["strategy_1"] = {"status": "no_red_fillers"}

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 2 — Output dimension split
    # Source: red-vs-GREEN Jacobian at zor (uses zor, but different output dims)
    # Target: red-vs-BLUE recovery in mAB (different output axis entirely)
    # ══════════════════════════════════════════════════════════════════════════
    D_red_green = get_jacobian(mA, SPECIAL_OBJECT, "CTX_RED", red_cls, green_cls)
    D_red_green_unit = D_red_green / (D_red_green.norm() + 1e-9)
    cos_D1_S2 = float(cosine_alignment(D1_unit, D_red_green_unit))

    # Test on TARGET: the red-vs-BLUE margin in mAB (completely different output pair)
    s2 = ablation_metrics(mAB, h_AB, D_red_green_unit, blue_cls, red_cls, seed)

    # Also test the other direction: derive from red-vs-blue, test on red-vs-green
    with torch.no_grad():
        l_nat_full = mAB.fc2(h_AB.unsqueeze(0))
        m_red_green_nat = (l_nat_full[0, red_cls] - l_nat_full[0, green_cls]).item()

        coeff_D1 = (h_AB @ D1_unit).item()
        h_D1_abl = h_AB - coeff_D1 * D1_unit
        l_D1_abl = mAB.fc2(h_D1_abl.unsqueeze(0))
        m_red_green_D1_abl = (l_D1_abl[0, red_cls] - l_D1_abl[0, green_cls]).item()
        cross_dim_delta = m_red_green_D1_abl - m_red_green_nat

    results["strategy_2"] = {
        "cos_D1_S2": round(cos_D1_S2, 4),
        # Primary: red-vs-green Jacobian → blue-vs-red target
        "red_green_to_blue_red": s2,
        # Reverse: red-vs-blue ablation → what happens to red-vs-green?
        "D1_ablation_effect_on_red_green": round(cross_dim_delta, 4),
        "m_red_green_nat": round(m_red_green_nat, 4),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 3 — Context split
    # Source: Jacobian at zor, CTX_BLUE (the OTHER context)
    # Target: zor's B-behavior under CTX_RED
    # ══════════════════════════════════════════════════════════════════════════
    D_ctx_blue = get_jacobian(mA, SPECIAL_OBJECT, "CTX_BLUE", red_cls, blue_cls)
    D_ctx_blue_unit = D_ctx_blue / (D_ctx_blue.norm() + 1e-9)
    cos_D1_S3 = float(cosine_alignment(D1_unit, D_ctx_blue_unit))

    # Test on TARGET: CTX_RED behavior in mAB
    s3 = ablation_metrics(mAB, h_AB, D_ctx_blue_unit, blue_cls, red_cls, seed)

    # Also: test on CTX_BLUE behavior in mAB (same context as source)
    with torch.no_grad():
        h_AB_blue_ctx = mAB.hidden(zid, cid_blue).squeeze(0)
    s3_same_ctx = ablation_metrics(mAB, h_AB_blue_ctx, D_ctx_blue_unit, blue_cls, red_cls, seed)

    # Cross-context reverse: D1 (from CTX_RED) → test on CTX_BLUE
    s3_cross = ablation_metrics(mAB, h_AB_blue_ctx, D1_unit, blue_cls, red_cls, seed)

    results["strategy_3"] = {
        "cos_D1_S3": round(cos_D1_S3, 4),
        # Primary: CTX_BLUE-derived → CTX_RED target (source ≠ target context)
        "ctx_blue_to_ctx_red": s3,
        # Same-context test (sanity): CTX_BLUE-derived → CTX_BLUE target
        "ctx_blue_to_ctx_blue": s3_same_ctx,
        # Reverse cross: CTX_RED-derived → CTX_BLUE target
        "ctx_red_to_ctx_blue": s3_cross,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 4 — Phase split (hardest)
    # Source: Jacobians at FILLER objects from EARLY A checkpoint (step 100)
    # Target: zor's B-behavior in fully-trained FINAL mAB
    # Disjoint: early model (not converged), filler objects, different phase
    # ══════════════════════════════════════════════════════════════════════════
    if red_fillers:
        jacs_early = []
        for obj in red_fillers:
            try:
                j = get_jacobian(mA_early, obj, "CTX_RED", red_cls, blue_cls)
                jacs_early.append(j)
            except Exception:
                pass

        if jacs_early:
            D_early_filler = torch.stack(jacs_early).mean(0)
            D_early_filler = D_early_filler / (D_early_filler.norm() + 1e-9)
            cos_D1_S4 = float(cosine_alignment(D1_unit, D_early_filler))

            # Test on TARGET: zor's final B-behavior in mAB
            s4 = ablation_metrics(mAB, h_AB, D_early_filler, blue_cls, red_cls, seed)

            # Verify early model hasn't converged: check early model's pred for zor
            with torch.no_grad():
                early_pred = mA_early(zid, cid_red).argmax(-1).item()
                early_zor_red_margin = (
                    mA_early(zid, cid_red)[0, red_cls] -
                    mA_early(zid, cid_red)[0, blue_cls]
                ).item()
                final_zor_red_margin = (
                    mA(zid, cid_red)[0, red_cls] -
                    mA(zid, cid_red)[0, blue_cls]
                ).item()

            results["strategy_4"] = {
                "early_checkpoint_step":    EARLY_CHECKPOINT,
                "cos_D1_S4":               round(cos_D1_S4, 4),
                "early_model_predicts_red": bool(early_pred == red_cls),
                "early_zor_red_margin":     round(early_zor_red_margin, 4),
                "final_zor_red_margin":     round(final_zor_red_margin, 4),
                # Primary: early-filler direction → final mAB zor target
                "early_filler_to_final_zor": s4,
                "n_early_jacs": len(jacs_early),
            }
        else:
            results["strategy_4"] = {"status": "no_valid_early_jacs"}
    else:
        results["strategy_4"] = {"status": "no_red_fillers"}

    return results


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Genuinely Independent Identification")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(seed)
        all_results.append(r)

        if r["status"] != "OK":
            print(f"  FAILED: {r['status']}")
            continue

        s1 = r.get("strategy_1", {})
        s2 = r.get("strategy_2", {})
        s3 = r.get("strategy_3", {})
        s4 = r.get("strategy_4", {})

        if "filler_to_zor" in s1:
            print(f"  S1 (obj split):     cos={s1['cos_D1_S1']:+.3f}  "
                  f"abl={s1['filler_to_zor']['abl_delta']:+.3f}  "
                  f"spec={s1['filler_to_zor']['specificity']:.1f}x")
        print(f"  S2 (dim split):     cos={s2['cos_D1_S2']:+.3f}  "
              f"abl={s2['red_green_to_blue_red']['abl_delta']:+.3f}  "
              f"spec={s2['red_green_to_blue_red']['specificity']:.1f}x")
        print(f"  S3 (ctx split):     cos={s3['cos_D1_S3']:+.3f}  "
              f"abl={s3['ctx_blue_to_ctx_red']['abl_delta']:+.3f}  "
              f"spec={s3['ctx_blue_to_ctx_red']['specificity']:.1f}x")
        if "early_filler_to_final_zor" in s4:
            print(f"  S4 (phase split):   cos={s4['cos_D1_S4']:+.3f}  "
                  f"abl={s4['early_filler_to_final_zor']['abl_delta']:+.3f}  "
                  f"spec={s4['early_filler_to_final_zor']['specificity']:.1f}x  "
                  f"early_converged={s4['early_model_predicts_red']}")

    with open("/home/claude/iclr/results/independent_identification.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/independent_identification.json")

    # ── Summary ────────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r.get("status") == "OK"]
    print(f"\n=== AGGREGATE (n={len(ok)}) ===")

    for skey, label, sub_key in [
        ("strategy_1", "S1 filler→zor",    "filler_to_zor"),
        ("strategy_2", "S2 dim-split",     "red_green_to_blue_red"),
        ("strategy_3", "S3 ctx-split",     "ctx_blue_to_ctx_red"),
        ("strategy_4", "S4 phase-split",   "early_filler_to_final_zor"),
    ]:
        vals_abl  = [r[skey][sub_key]["abl_delta"]   for r in ok if sub_key in r.get(skey, {})]
        vals_spec = [r[skey][sub_key]["specificity"] for r in ok if sub_key in r.get(skey, {})]
        vals_cos  = [r[skey].get("cos_D1_S" + skey[-1], float("nan")) for r in ok if skey in r]
        if vals_abl:
            print(f"  {label:20}: abl={np.mean(vals_abl):+.3f}±{np.std(vals_abl):.3f}  "
                  f"spec={np.mean(vals_spec):.1f}x  cos_D1={np.mean(vals_cos):+.3f}±{np.std(vals_cos):.3f}")
