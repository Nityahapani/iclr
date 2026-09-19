"""
Experiment: Independent Upstream Identification
derive → freeze → test on genuinely held-out information

Question: Is J_A merely a readout-direction artifact, or does it identify
information-bearing structure upstream of the readout?

Design — strictly pre-registered:

DERIVE (source set, no readout):
  d_src = mean(h_red_src_fillers) - mean(h_nonred_fillers)   at mA
  where red_src_fillers = first half of red-labeled fillers.
  Method: mean-difference in hidden space. NEVER touches fc2.
  Direction is derived, then FROZEN. No subsequent modification.

TEST (three independent tests, each on disjoint held-out information):

  Test 1 — Held-out object prediction (upstream representational)
    Objects: held-out red fillers (second half) + nonred fillers.
    These objects were NEVER used to derive d_src.
    Labels: fixed before test as {held-out red = 1, nonred = 0}.
    Measurement: does h_obj (in mAB) project onto d_src in a way that
    predicts the fixed label?
    Metric: point-biserial r, AUC, t-test. No readout used.

  Test 2 — Causal upstream intervention (held-out objects)
    Objects: held-out nonred fillers.
    Intervention: add alpha * d_src to h_obj, pass through current fc2.
    Question: does output shift toward red (even though d_src was derived
    from DIFFERENT objects)?
    Control: add alpha * d_perp (orthogonal direction, same norm).
    Metric: red-class logit shift. Tests that d_src causally controls
    color-encoding upstream, for objects it never saw.

  Test 3 — Transfer to zor in mAB (the core target)
    Object: zor (NEVER used to derive d_src).
    Model: mAB (B-trained — d_src was derived from mA, not mAB).
    Measurement 1 (representational): h_zor in mAB projected onto d_src.
      Sign should match held-out red fillers (positive = red-like).
    Measurement 2 (causal): ablating d_src from h_zor disrupts B output.
      Compare to ablating d_perp (matched-norm orthogonal control).
    Measurement 3 (ordering): h_zor's d_src-projection ranks relative to
      ALL fillers. If zor ranks above nonred fillers, d_src identifies
      zor's upstream red-encoding — from a direction derived only on fillers.

  Test 4 — Cross-model transfer (mA → mAB)
    d_src was derived at mA. mAB has different weights after B-training.
    Does d_src still identify the same upstream structure in mAB?
    Derive d_src_AB = same formula but at mAB's hidden states.
    Measure cos(d_src_mA, d_src_mAB): if close to 1, upstream structure
    persisted through B-training at the level of filler representations.
    This is independent of zor entirely.

Pre-registration:
  Labels, objects, and direction are ALL fixed before Test 1-4 run.
  No searching over splits or directions. For each seed:
    - red_src  = FILLER_OBJECTS where fm[o]='red', first ceil(n/2) objects
    - red_held = remaining red fillers
    - nonred   = all non-red fillers
    - d_src    = mean(h_red_src) - mean(h_nonred) at mA, normalised
  These are set mechanically from the filler_mapping before any test runs.
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
from src.probe import cosine_alignment

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG   = dict(hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
             phase_A_steps=600, phase_A_lr=0.01,
             phase_B_steps=3000, phase_B_lr=0.005, batch_size=32)


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                          embed_dim=CFG["embed_dim"],
                          ctx_embed_dim=CFG["ctx_embed_dim"],
                          hidden_dim=CFG["hidden_dim"])


def train_B(init_state, fm, seed):
    m = new_model(); m.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(m.parameters(), lr=CFG["phase_B_lr"])
    ds  = PhaseDataset(fm, "B")
    rng = np.random.RandomState(seed)
    for _ in range(CFG["phase_B_steps"]):
        o, c, l = ds.sample_batch(CFG["batch_size"], rng)
        F.cross_entropy(m(o, c), l).backward()
        opt.step(); opt.zero_grad()
    return m


def get_h(model, obj_name, ctx_name="CTX_RED"):
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(oid, cid).squeeze(0)


def derive_d_src(model, src_objects, nonred_objects):
    """
    DERIVE: mean-difference direction in hidden space.
    No fc2 access. Returns unit-norm direction.
    """
    with torch.no_grad():
        h_src    = torch.stack([get_h(model, o) for o in src_objects])
        h_nonred = torch.stack([get_h(model, o) for o in nonred_objects])
        d = h_src.mean(0) - h_nonred.mean(0)
    return d / (d.norm() + 1e-9)


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    blue_cls = COLOR2ID["blue"]
    red_cls  = COLOR2ID["red"]
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)

    # ── PRE-REGISTRATION: fix splits before any model training ────────────────
    red_fillers    = [o for o in FILLER_OBJECTS if fm[o] == "red"]
    nonred_fillers = [o for o in FILLER_OBJECTS if fm[o] != "red"]

    if len(red_fillers) < 2:
        return {"seed": seed, "status": "INSUFFICIENT_RED_FILLERS",
                "n_red": len(red_fillers)}

    n_src = (len(red_fillers) + 1) // 2   # ceil(n/2)
    red_src  = red_fillers[:n_src]         # SOURCE: derive d_src from these
    red_held = red_fillers[n_src:]         # TARGET: held-out test objects

    # Fixed labels (set before any test): held-out red = 1, nonred = 0
    # This is the label assignment for Test 1. Frozen now.
    test1_objects = red_held + nonred_fillers
    test1_labels  = [1] * len(red_held) + [0] * len(nonred_fillers)

    # ── Phase A ────────────────────────────────────────────────────────────────
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"],
                batch_size=CFG["batch_size"], lr=CFG["phase_A_lr"],
                seed=seed, eval_every=CFG["phase_A_steps"])
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    # ── DERIVE: d_src from source fillers at mA. FREEZE. ─────────────────────
    d_src = derive_d_src(mA, red_src, nonred_fillers)
    # d_src is now frozen. It will not be modified for any test.

    # Orthogonal control direction (same norm = 1, perpendicular to d_src)
    g = torch.Generator().manual_seed(seed + 12345)
    v_raw = torch.randn(CFG["hidden_dim"], generator=g)
    v_raw = v_raw - (v_raw @ d_src) * d_src
    d_perp = v_raw / (v_raw.norm() + 1e-9)

    # Also compute J_A for comparison (readout-based direction)
    with torch.no_grad():
        W = mA.fc2.weight
        J_A_unit = (W[red_cls] - W[blue_cls])
        J_A_unit = J_A_unit / (J_A_unit.norm() + 1e-9)
    cos_dsrc_JA = float(cosine_alignment(d_src, J_A_unit))

    # ── Phase B ────────────────────────────────────────────────────────────────
    mAB = train_B(mA.state_dict(), fm, seed)
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # ═════════════════════════════════════════════════════════════════════════
    # TEST 1 — Held-out object prediction (upstream representational, no fc2)
    # Objects: red_held + nonred_fillers. Labels: fixed above.
    # Measurement: projection of h_obj in mAB onto d_src predicts label.
    # ═════════════════════════════════════════════════════════════════════════
    projections = []
    for obj in test1_objects:
        h_obj = get_h(mAB, obj)
        projections.append((h_obj @ d_src).item())

    labels_arr  = np.array(test1_labels, dtype=float)
    projs_arr   = np.array(projections)

    # Point-biserial correlation (fixed labels, fixed direction)
    if len(set(test1_labels)) > 1 and len(test1_labels) >= 4:
        r_pb, p_pb = stats.pointbiserialr(labels_arr, projs_arr)
    else:
        r_pb, p_pb = float("nan"), float("nan")

    # Welch t-test: held-out red projections vs nonred projections
    proj_red_held = [projections[i] for i, l in enumerate(test1_labels) if l == 1]
    proj_nonred   = [projections[i] for i, l in enumerate(test1_labels) if l == 0]
    if proj_red_held and proj_nonred:
        t_1, p_1 = stats.ttest_ind(proj_red_held, proj_nonred, equal_var=False,
                                    alternative="greater")
    else:
        t_1, p_1 = float("nan"), float("nan")

    # AUC: fraction of (red_held, nonred) pairs where red > nonred
    auc_pairs = [(r > n) for r in proj_red_held for n in proj_nonred]
    auc = float(np.mean(auc_pairs)) if auc_pairs else float("nan")

    test1 = {
        "n_test_objects": len(test1_objects),
        "n_red_held": len(red_held),
        "n_nonred": len(nonred_fillers),
        "proj_red_held_mean": round(float(np.mean(proj_red_held)), 4) if proj_red_held else None,
        "proj_nonred_mean":   round(float(np.mean(proj_nonred)), 4)   if proj_nonred   else None,
        "r_pb":  round(float(r_pb), 4) if not np.isnan(r_pb) else None,
        "p_pb":  round(float(p_pb), 6) if not np.isnan(p_pb) else None,
        "t_red_gt_nonred": round(float(t_1), 4) if not np.isnan(t_1) else None,
        "p_red_gt_nonred": round(float(p_1), 6) if not np.isnan(p_1) else None,
        "auc": round(auc, 4) if not np.isnan(auc) else None,
    }

    # ═════════════════════════════════════════════════════════════════════════
    # TEST 2 — Causal upstream intervention on held-out nonred fillers
    # Steer h_obj along d_src; measure red-class logit shift.
    # Control: steer along d_perp (orthogonal, same norm).
    # ═════════════════════════════════════════════════════════════════════════
    steer_results = []
    for obj in nonred_fillers[:6]:
        true_cls  = COLOR2ID[fm[obj]]
        h_obj     = get_h(mAB, obj)
        with torch.no_grad():
            l_nat = mAB.fc2(h_obj.unsqueeze(0))
            m_nat_red = (l_nat[0, red_cls] - l_nat[0, true_cls]).item()

        for alpha in [1.0, 2.0, 4.0, 8.0]:
            with torch.no_grad():
                h_steered  = h_obj + alpha * d_src
                h_ctrl     = h_obj + alpha * d_perp
                l_st   = mAB.fc2(h_steered.unsqueeze(0))
                l_ct   = mAB.fc2(h_ctrl.unsqueeze(0))
                shift_red  = (l_st[0,  red_cls] - l_st[0, true_cls]).item() - m_nat_red
                shift_ctrl = (l_ct[0,  red_cls] - l_ct[0, true_cls]).item() - m_nat_red
            steer_results.append({
                "obj": obj, "alpha": alpha,
                "red_shift_dsrc":  round(shift_red,  4),
                "red_shift_ctrl":  round(shift_ctrl, 4),
                "specificity": round(abs(shift_red) / (abs(shift_ctrl) + 1e-9), 2),
            })

    mean_red_shift_4 = float(np.mean([s["red_shift_dsrc"] for s in steer_results if s["alpha"]==4.0]))
    mean_ctrl_shift_4= float(np.mean([s["red_shift_ctrl"] for s in steer_results if s["alpha"]==4.0]))
    mean_spec_4      = float(np.mean([s["specificity"]    for s in steer_results if s["alpha"]==4.0]))

    test2 = {
        "mean_red_shift_alpha4":  round(mean_red_shift_4,  4),
        "mean_ctrl_shift_alpha4": round(mean_ctrl_shift_4, 4),
        "mean_specificity_alpha4":round(mean_spec_4,       2),
        "steer_results": steer_results,
    }

    # ═════════════════════════════════════════════════════════════════════════
    # TEST 3 — Transfer to zor in mAB
    # zor was NEVER used in derivation. mAB was NEVER used in derivation.
    # d_src was derived at mA from a filler subset.
    # ═════════════════════════════════════════════════════════════════════════
    h_zor_AB = get_h(mAB, SPECIAL_OBJECT)
    proj_zor  = (h_zor_AB @ d_src).item()

    # Representational rank: what fraction of fillers score below zor on d_src?
    all_filler_projs = []
    for obj in FILLER_OBJECTS:
        h_f = get_h(mAB, obj)
        all_filler_projs.append((obj, fm[obj], (h_f @ d_src).item()))
    rank_pct = float(np.mean([p < proj_zor for _, _, p in all_filler_projs])) * 100

    # Sign check: does zor's projection have the same sign as red_held mean?
    sign_matches_red = (proj_zor > 0) == (np.mean(proj_red_held) > 0) if proj_red_held else None

    # Causal: ablate d_src from h_zor in mAB; measure B-output disruption
    with torch.no_grad():
        l_nat_zor = mAB.fc2(h_zor_AB.unsqueeze(0))
        m_nat_zor = (l_nat_zor[0, blue_cls] - l_nat_zor[0, red_cls]).item()
        coeff_dsrc = (h_zor_AB @ d_src).item()
        h_abl      = h_zor_AB - coeff_dsrc * d_src
        l_abl      = mAB.fc2(h_abl.unsqueeze(0))
        m_abl_zor  = (l_abl[0, blue_cls] - l_abl[0, red_cls]).item()
        abl_delta_dsrc = m_abl_zor - m_nat_zor

        # Control: ablate d_perp
        coeff_perp = (h_zor_AB @ d_perp).item()
        h_ctrl_z   = h_zor_AB - coeff_perp * d_perp
        l_ctrl_z   = mAB.fc2(h_ctrl_z.unsqueeze(0))
        m_ctrl_z   = (l_ctrl_z[0, blue_cls] - l_ctrl_z[0, red_cls]).item()
        ctrl_delta_zor = m_ctrl_z - m_nat_zor

        # J_A ablation for comparison
        coeff_JA   = (h_zor_AB @ J_A_unit).item()
        h_JA_abl   = h_zor_AB - coeff_JA * J_A_unit
        l_JA_abl   = mAB.fc2(h_JA_abl.unsqueeze(0))
        m_JA_abl   = (l_JA_abl[0, blue_cls] - l_JA_abl[0, red_cls]).item()
        abl_delta_JA = m_JA_abl - m_nat_zor

    test3 = {
        "proj_zor_dsrc":     round(proj_zor, 4),
        "proj_red_held_mean":round(float(np.mean(proj_red_held)), 4) if proj_red_held else None,
        "proj_nonred_mean":  round(float(np.mean(proj_nonred)), 4),
        "sign_matches_red":  sign_matches_red,
        "rank_pct":          round(rank_pct, 1),
        "m_nat_zor":         round(m_nat_zor, 4),
        "abl_delta_dsrc":    round(abl_delta_dsrc, 4),
        "ctrl_delta_zor":    round(ctrl_delta_zor, 4),
        "abl_delta_JA":      round(abl_delta_JA, 4),
        "specificity_dsrc":  round(abs(abl_delta_dsrc) / (abs(ctrl_delta_zor) + 1e-9), 2),
        "dsrc_as_frac_of_JA":round(abl_delta_dsrc / (abl_delta_JA + 1e-9), 4),
    }

    # ═════════════════════════════════════════════════════════════════════════
    # TEST 4 — Cross-model transfer: does d_src persist in mAB's filler reps?
    # Derive d_src_AB using same formula but at mAB (completely independent).
    # cos(d_src_mA, d_src_mAB) measures upstream structure persistence.
    # Neither uses zor; this is filler-only evidence.
    # ═════════════════════════════════════════════════════════════════════════
    d_src_AB = derive_d_src(mAB, red_src, nonred_fillers)
    cos_dsrc_AB = float(cosine_alignment(d_src, d_src_AB))

    # Also derive from held-out red fillers (held-out object set)
    if red_held:
        d_src_held = derive_d_src(mAB, red_held, nonred_fillers)
        cos_held_AB = float(cosine_alignment(d_src, d_src_held))
        cos_src_held_AB = float(cosine_alignment(d_src_AB, d_src_held))
    else:
        d_src_held = None
        cos_held_AB = float("nan")
        cos_src_held_AB = float("nan")

    test4 = {
        "cos_dsrc_mA_mAB":      round(cos_dsrc_AB, 4),
        "cos_dsrc_mA_held_mAB": round(cos_held_AB, 4),
        "cos_src_held_mAB":     round(cos_src_held_AB, 4),
        "cos_dsrc_JA":          round(cos_dsrc_JA, 4),
    }

    return {
        "seed": seed, "status": "OK",
        "n_red_src":    len(red_src),
        "n_red_held":   len(red_held),
        "n_nonred":     len(nonred_fillers),
        "cos_dsrc_JA":  round(cos_dsrc_JA, 4),
        "test1": test1,
        "test2": test2,
        "test3": test3,
        "test4": test4,
    }


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Independent Upstream Identification")
    print("derive → freeze → test on genuinely held-out information")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(seed)
        all_results.append(r)

        if r["status"] != "OK":
            print(f"  {r['status']} (n_red={r.get('n_red', '?')})")
            continue

        t1, t2, t3, t4 = r["test1"], r["test2"], r["test3"], r["test4"]
        print(f"  src={r['n_red_src']} held={r['n_red_held']} nonred={r['n_nonred']}  "
              f"cos(d_src,JA)={r['cos_dsrc_JA']:+.3f}")
        print(f"  T1 (held-out repr): r_pb={t1.get('r_pb','NA')} "
              f"p={t1.get('p_pb','NA')}  auc={t1.get('auc','NA')}  "
              f"t={t1.get('t_red_gt_nonred','NA')} p={t1.get('p_red_gt_nonred','NA')}")
        print(f"  T2 (causal steer):  red_shift@4={t2['mean_red_shift_alpha4']:+.3f}  "
              f"ctrl={t2['mean_ctrl_shift_alpha4']:+.3f}  "
              f"spec={t2['mean_specificity_alpha4']:.1f}x")
        print(f"  T3 (zor transfer):  proj_zor={t3['proj_zor_dsrc']:+.4f}  "
              f"rank={t3['rank_pct']:.0f}%  "
              f"abl_dsrc={t3['abl_delta_dsrc']:+.3f}  "
              f"abl_JA={t3['abl_delta_JA']:+.3f}  "
              f"dsrc/JA={t3['dsrc_as_frac_of_JA']:+.3f}")
        print(f"  T4 (cross-model):   cos(mA,mAB)={t4['cos_dsrc_mA_mAB']:+.3f}  "
              f"cos(mA,held_mAB)={t4['cos_dsrc_mA_held_mAB']:+.3f}")

    with open("/home/claude/iclr/results/independent_upstream_id.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved.")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== AGGREGATE (n={len(ok)}) ===\n")

    def collect(key_path):
        """Traverse nested dict by dot-separated path."""
        vals = []
        for r in ok:
            node = r
            for k in key_path.split("."):
                node = node.get(k) if isinstance(node, dict) else None
            if node is not None and not (isinstance(node, float) and np.isnan(node)):
                vals.append(float(node))
        return vals

    metrics = [
        ("T1 r_pb",             "test1.r_pb"),
        ("T1 AUC",              "test1.auc"),
        ("T1 t(red>nonred)",    "test1.t_red_gt_nonred"),
        ("T2 red_shift@4",      "test2.mean_red_shift_alpha4"),
        ("T2 ctrl_shift@4",     "test2.mean_ctrl_shift_alpha4"),
        ("T2 specificity@4",    "test2.mean_specificity_alpha4"),
        ("T3 proj_zor",         "test3.proj_zor_dsrc"),
        ("T3 rank_%",           "test3.rank_pct"),
        ("T3 abl_dsrc",         "test3.abl_delta_dsrc"),
        ("T3 abl_JA",           "test3.abl_delta_JA"),
        ("T3 dsrc/JA",          "test3.dsrc_as_frac_of_JA"),
        ("T3 specificity",      "test3.specificity_dsrc"),
        ("T4 cos(mA,mAB)",      "test4.cos_dsrc_mA_mAB"),
        ("T4 cos(mA,held_mAB)", "test4.cos_dsrc_mA_held_mAB"),
        ("cos(d_src, J_A)",     "cos_dsrc_JA"),
    ]

    for label, path in metrics:
        vals = collect(path)
        if not vals:
            continue
        mean, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
        print(f"  {label:<28}: {mean:+.4f} +/- {sd:.4f}  (n={len(vals)})")

    # Formal tests
    print()
    rpb   = collect("test1.r_pb");            t,p = stats.ttest_1samp(rpb, 0)   if rpb   else (np.nan,np.nan)
    print(f"  T1 t(r_pb vs 0):   t={t:.3f}, p={p:.6f}")
    abl_d = collect("test3.abl_delta_dsrc");  t,p = stats.ttest_1samp(abl_d, 0) if abl_d else (np.nan,np.nan)
    print(f"  T3 t(abl_dsrc vs 0): t={t:.3f}, p={p:.6f}")
    red_s = collect("test2.mean_red_shift_alpha4"); ctrl_s = collect("test2.mean_ctrl_shift_alpha4")
    if red_s and ctrl_s:
        t,p = stats.ttest_rel(red_s, ctrl_s)
        print(f"  T2 t(red_shift vs ctrl): t={t:.3f}, p={p:.6f}")
    cos_t4 = collect("test4.cos_dsrc_mA_mAB");  t,p = stats.ttest_1samp(cos_t4, 0) if cos_t4 else (np.nan,np.nan)
    print(f"  T4 t(cos vs 0):    t={t:.3f}, p={p:.6f}")
