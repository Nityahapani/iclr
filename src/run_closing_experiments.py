"""
Three experiments closing the remaining gaps:

EXP A — Genuinely independent causal identification, not from the readout
  Derives a direction from INTERNAL representational structure only
  (linear probe on filler hidden states, no access to fc2 weights),
  then tests it causally on a disjoint target and disjoint measurement.
  Source: probe fitted on filler hiddens at mA to classify red vs non-red.
          Never sees zor; never touches fc2.
  Target measurement 1: does ablating the probe direction from h_AB disrupt
          the B output? (readout-dependent, but direction is readout-independent)
  Target measurement 2: ACTIVATION STEERING — does adding the probe direction
          to a NON-RED object's hidden state steer its output toward red?
          Completely disjoint from zor and from the ablation measurement.
  Target measurement 3: within-mAB, does h_zor have a significantly larger
          projection onto the probe direction than h_non-red-fillers?
          Purely internal representational test, no readout involved.

EXP B — Seed-level statistical analysis of cross-lineage experiment
  Re-runs the core cross-lineage rescue experiment (patch J_A from lineage X
  into model from lineage Y, ask whether B output is restored) with:
    - Full per-seed results for own-lineage vs cross-lineage
    - One-sample t-test: own-lineage rescue_delta vs 0
    - Paired t-test: own-lineage vs cross-lineage rescue_delta
    - Permutation test: shuffle lineage assignments, recompute gap,
      compare to observed gap (p-value under null of no lineage specificity)
    - Cohen's d effect size for the own vs cross gap
    - Confidence intervals (bootstrap, 10k resamples)

EXP C — Persistent computation vs persistent alignment/parameterisation
  The key distinction: does mAB compute the red-encoding as a live
  intermediate variable, or does it merely retain parameter structure
  that would enable such computation if called upon?

  Three independent probes of the same computational variable
  (the internal red-encoding of zor), using no shared method:

  C1. REPRESENTATIONAL PROBE (no readout, no J_A):
      Fit a linear classifier on filler hidden states at mA to detect
      red-encodings (direction P). Test whether h_zor in mAB has a
      significantly larger P-projection than h_non-red objects.
      This is purely internal: no fc2, no J_A, no output margin.

  C2. CAUSAL STEERING (no J_A, no output margin for direction):
      Derive a "red steering vector" = mean(h_red_fillers) - mean(h_nonred_fillers)
      at mAB itself (computed on FILLER objects, never on zor).
      Add this vector to a non-red filler's hidden state and measure output shift.
      Then check: does h_zor already have a large projection onto this steering
      vector, relative to non-red fillers? Purely internal convergent evidence.

  C3. CROSS-METHOD CONVERGENCE (the kill shot):
      Build directions from three completely independent methods:
        d1 = J_A (readout Jacobian at mA)
        d2 = probe direction P (linear classifier on filler hiddens at mA)
        d3 = activation difference (mean red filler hiddens minus mean nonred, at mAB)
      Test: do all three agree on the SIGN and RANK of zor's projection
      relative to non-red fillers?
      If all three converge — same sign, similar rank, independent of each
      other's construction — the conclusion that mAB carries an internal
      red-encoding of zor is overdetermined across methods, making
      "it's just readout alignment" very difficult to argue.
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
    SPECIAL_OBJECT, SHAM_OBJECT, CONTROL_OBJECT, FILLER_OBJECTS,
    VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (
    jacobian_zor_red_vs_blue, jacobian_of_margin, cosine_alignment,
    build_v_A_diff_in_diff, activation_patch_from_theta_A,
)

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG = dict(
    hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
    phase_A_steps=600, phase_A_lr=0.01,
    phase_B_steps=3000, phase_B_lr=0.005, batch_size=32,
)


def new_model():
    return TinyClassifier(
        VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
        embed_dim=CFG["embed_dim"], ctx_embed_dim=CFG["ctx_embed_dim"],
        hidden_dim=CFG["hidden_dim"],
    )


def train_B(init_state, fm, seed):
    m = new_model()
    m.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(m.parameters(), lr=CFG["phase_B_lr"])
    ds  = PhaseDataset(fm, "B")
    rng = np.random.RandomState(seed)
    for _ in range(CFG["phase_B_steps"]):
        o, c, l = ds.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(m(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def fit_probe_direction(model, fm, ctx_name="CTX_RED"):
    """
    Fit a linear probe on filler hidden states to classify red vs non-red.
    Never sees zor, never touches fc2.
    Returns unit-norm probe weight vector.
    """
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    h_list, y_list = [], []
    with torch.no_grad():
        for obj in FILLER_OBJECTS:
            oid = torch.tensor([OBJ2ID[obj]], dtype=torch.long)
            h = model.hidden(oid, ctx_id).squeeze(0)
            h_list.append(h)
            y_list.append(1.0 if fm[obj] == "red" else 0.0)
    H = torch.stack(h_list)            # [n_fillers, hidden_dim]
    Y = torch.tensor(y_list)           # [n_fillers]

    # Logistic regression via gradient descent
    w = torch.zeros(H.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(500):
        logits = H.detach() @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()

    w_unit = w.detach() / (w.detach().norm() + 1e-9)
    # probe accuracy
    with torch.no_grad():
        preds = (H @ w + b > 0).float()
        acc = (preds == Y).float().mean().item()
    return w_unit, acc


def get_filler_projections(model, fm, direction, ctx_name="CTX_RED"):
    """Project each filler's hidden state onto direction. Returns dict obj->projection."""
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    projs = {}
    with torch.no_grad():
        for obj in FILLER_OBJECTS:
            oid = torch.tensor([OBJ2ID[obj]], dtype=torch.long)
            h = model.hidden(oid, ctx_id).squeeze(0)
            projs[obj] = (h @ direction).item()
    return projs


def ablation_delta(model, obj_name, ctx_name, direction, cls_a, cls_b):
    """Ablate direction from h, return margin change (cls_a - cls_b)."""
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    d_unit = direction / (direction.norm() + 1e-9)
    with torch.no_grad():
        h = model.hidden(oid, cid).squeeze(0)
        l_nat = model.fc2(h.unsqueeze(0))
        m_nat = (l_nat[0, cls_a] - l_nat[0, cls_b]).item()
        coeff = (h @ d_unit).item()
        h_abl = h - coeff * d_unit
        l_abl = model.fc2(h_abl.unsqueeze(0))
        m_abl = (l_abl[0, cls_a] - l_abl[0, cls_b]).item()
    return m_abl - m_nat, m_nat


# ═══════════════════════════════════════════════════════════════════════════════
# EXP A — Independent readout-free causal identification
# ═══════════════════════════════════════════════════════════════════════════════

def exp_A(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    red_cls  = COLOR2ID["red"]
    blue_cls = COLOR2ID["blue"]

    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    mAB = train_B(mA.state_dict(), fm, seed)
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # Standard J_A (for comparison only — not used to construct test directions)
    J_A = jacobian_zor_red_vs_blue(mA, "CTX_RED")
    J_A_unit = J_A / (J_A.norm() + 1e-9)

    # ── Probe direction: fit on filler hiddens at mA, never touches fc2 or zor ──
    P, probe_acc_A = fit_probe_direction(mA, fm, "CTX_RED")
    cos_P_JA = cosine_alignment(P, J_A_unit)

    # Probe accuracy on mAB fillers (does the probe still work after B-training?)
    _, probe_acc_AB = fit_probe_direction(mAB, fm, "CTX_RED")

    # ── Target measurement 1: ablate P from h_zor in mAB, check B disruption ──
    # (direction is readout-independent; target measurement uses readout)
    abl_P_delta, m_nat = ablation_delta(mAB, SPECIAL_OBJECT, "CTX_RED",
                                         P, blue_cls, red_cls)

    # Orthogonal control: matched norm, perpendicular to P
    g = torch.Generator().manual_seed(seed + 33333)
    v_ctrl = torch.randn(CFG["hidden_dim"], generator=g)
    v_ctrl = v_ctrl - (v_ctrl @ P) * P
    v_ctrl = v_ctrl / (v_ctrl.norm() + 1e-9)
    abl_ctrl_delta, _ = ablation_delta(mAB, SPECIAL_OBJECT, "CTX_RED",
                                        v_ctrl, blue_cls, red_cls)

    # ── Target measurement 2: activation steering on NON-ZOR objects ──
    # Add P to a non-red filler's hidden state; check if output shifts toward red.
    # Completely disjoint from zor and from any measurement involving zor.
    nonred_fillers = [o for o in FILLER_OBJECTS if fm[o] != "red"]
    steering_results = []
    for obj in nonred_fillers[:4]:   # test on first 4 non-red fillers
        obj_id = torch.tensor([OBJ2ID[obj]], dtype=torch.long)
        true_cls = COLOR2ID[fm[obj]]
        with torch.no_grad():
            h_obj = mAB.hidden(obj_id, cid).squeeze(0)
            l_nat_obj = mAB.fc2(h_obj.unsqueeze(0))
            m_nat_obj = (l_nat_obj[0, red_cls] - l_nat_obj[0, true_cls]).item()

            # Steer toward P direction: h_steered = h + alpha * P
            for alpha in [1.0, 2.0, 4.0]:
                h_steered = h_obj + alpha * P
                l_steer = mAB.fc2(h_steered.unsqueeze(0))
                m_steered = (l_steer[0, red_cls] - l_steer[0, true_cls]).item()
                steering_results.append({
                    "obj": obj, "alpha": alpha,
                    "m_nat": round(m_nat_obj, 4),
                    "m_steered": round(m_steered, 4),
                    "red_shift": round(m_steered - m_nat_obj, 4),
                    "pred_steered": l_steer.argmax(-1).item(),
                })

    # ── Target measurement 3: representational rank test ──
    # Does h_zor have a larger P-projection than non-red fillers in mAB?
    # No readout involved at all.
    projs_AB = get_filler_projections(mAB, fm, P)
    with torch.no_grad():
        h_zor_AB = mAB.hidden(zid, cid).squeeze(0)
        proj_zor_AB = (h_zor_AB @ P).item()

    proj_red_fillers    = [projs_AB[o] for o in FILLER_OBJECTS if fm[o] == "red"]
    proj_nonred_fillers = [projs_AB[o] for o in FILLER_OBJECTS if fm[o] != "red"]

    # One-sided t-test: is zor's projection significantly larger than non-red fillers?
    # H0: proj_zor <= mean(proj_nonred); H1: proj_zor > mean(proj_nonred)
    t_stat, p_val_2sided = stats.ttest_1samp(proj_nonred_fillers,
                                               proj_zor_AB)  # tests if nonred mean == proj_zor
    # note: ttest_1samp(data, popmean) tests H0: mean(data)==popmean
    # we want H0: proj_zor == mean(nonred), so use:
    t_stat2, p_val_2sided2 = stats.ttest_1samp(proj_nonred_fillers, proj_zor_AB)
    p_one_sided = p_val_2sided2 / 2  # one-sided

    # Point-biserial: does fm color (red=1, nonred=0) predict P-projection across fillers?
    filler_colors = np.array([1.0 if fm[o] == "red" else 0.0 for o in FILLER_OBJECTS])
    filler_projs  = np.array([projs_AB[o] for o in FILLER_OBJECTS])
    r_pb, p_pb = stats.pointbiserialr(filler_colors, filler_projs)

    return {
        "seed": seed, "status": "OK",
        "probe_acc_A": round(probe_acc_A, 4),
        "probe_acc_AB": round(probe_acc_AB, 4),
        "cos_P_JA": round(cos_P_JA, 4),
        # Measurement 1: causal ablation of probe direction
        "m_nat_blue_red": round(m_nat, 4),
        "abl_P_delta": round(abl_P_delta, 4),
        "abl_ctrl_delta": round(abl_ctrl_delta, 4),
        "specificity": round(abs(abl_P_delta) / (abs(abl_ctrl_delta) + 1e-9), 2),
        # Measurement 2: steering on non-zor objects (sample)
        "mean_red_shift_alpha1": round(np.mean([r["red_shift"] for r in steering_results
                                                 if r["alpha"] == 1.0]), 4),
        "mean_red_shift_alpha4": round(np.mean([r["red_shift"] for r in steering_results
                                                 if r["alpha"] == 4.0]), 4),
        "steering_results": steering_results,
        # Measurement 3: representational rank
        "proj_zor_AB": round(proj_zor_AB, 4),
        "mean_proj_red_fillers": round(np.mean(proj_red_fillers), 4) if proj_red_fillers else None,
        "mean_proj_nonred_fillers": round(np.mean(proj_nonred_fillers), 4),
        "t_zor_vs_nonred": round(t_stat2, 4),
        "p_one_sided": round(p_one_sided, 6),
        "r_pb_color_proj": round(r_pb, 4),
        "p_pb": round(p_pb, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXP B — Seed-level statistical analysis of cross-lineage experiment
# ═══════════════════════════════════════════════════════════════════════════════

def exp_B(all_seeds):
    """
    Cross-lineage rescue: patch J_A from lineage X into mAB from lineage Y.
    Own-lineage: patch mA_seed_i's J_A into mAB_seed_i (same filler mapping).
    Cross-lineage: patch mA_seed_j's J_A into mAB_seed_i (j != i).

    Rescue_delta = m_rescued - m_ablated.
    H0 (cross-lineage): rescue_delta = 0.
    H1 (own-lineage > cross-lineage): the rescue is lineage-specific.
    """
    blue_cls = COLOR2ID["blue"]
    red_cls  = COLOR2ID["red"]

    # Train all seeds first, collect states
    models = {}
    for seed in all_seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        fm   = make_filler_mapping(seed=seed)
        ds_A = PhaseDataset(fm, "A")
        mA   = new_model()
        train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                    lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
        zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
        cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
        with torch.no_grad():
            if mA(zid, cid).argmax(-1).item() != red_cls:
                models[seed] = None; continue
        mAB = train_B(mA.state_dict(), fm, seed)
        with torch.no_grad():
            if mAB(zid, cid).argmax(-1).item() != blue_cls:
                models[seed] = None; continue
        J_A = jacobian_zor_red_vs_blue(mA, "CTX_RED")
        models[seed] = {"mA": mA, "mAB": mAB, "J_A": J_A, "fm": fm}

    valid_seeds = [s for s in all_seeds if models[s] is not None]
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    per_seed_results = []

    for seed_T in valid_seeds:
        mAB = models[seed_T]["mAB"]
        J_A_own = models[seed_T]["J_A"]
        J_A_own_unit = J_A_own / (J_A_own.norm() + 1e-9)

        with torch.no_grad():
            h_nat = mAB.hidden(zid, cid).squeeze(0)
            l_nat = mAB.fc2(h_nat.unsqueeze(0))
            m_nat = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()

        # Own-lineage: ablate then restore with own J_A
        with torch.no_grad():
            coeff_own = (h_nat @ J_A_own_unit).item()
            h_abl_own  = h_nat - coeff_own * J_A_own_unit
            h_resc_own = h_abl_own + coeff_own * J_A_own_unit  # == h_nat
            l_abl_own  = mAB.fc2(h_abl_own.unsqueeze(0))
            m_abl_own  = (l_abl_own[0, blue_cls] - l_abl_own[0, red_cls]).item()
            # restore: project out own from ablated state, add back
            coeff_own_from_abl = (h_abl_own @ J_A_own_unit).item()
            h_resc_own = h_abl_own + coeff_own_from_abl * J_A_own_unit
            l_resc_own = mAB.fc2(h_resc_own.unsqueeze(0))
            m_resc_own = (l_resc_own[0, blue_cls] - l_resc_own[0, red_cls]).item()

        own_rescue_delta = m_resc_own - m_abl_own
        own_abl_delta    = m_abl_own  - m_nat

        # Cross-lineage: for each other seed, patch THAT seed's J_A into THIS mAB
        cross_rescues = []
        for seed_S in valid_seeds:
            if seed_S == seed_T:
                continue
            J_A_cross      = models[seed_S]["J_A"]
            J_A_cross_unit = J_A_cross / (J_A_cross.norm() + 1e-9)
            with torch.no_grad():
                coeff_cross = (h_nat @ J_A_cross_unit).item()
                h_abl_cross = h_nat - coeff_cross * J_A_cross_unit
                l_abl_cross = mAB.fc2(h_abl_cross.unsqueeze(0))
                m_abl_cross = (l_abl_cross[0, blue_cls] - l_abl_cross[0, red_cls]).item()

                # Rescue: restore the cross-lineage J_A component from ablated state
                coeff_cross_from_abl = (h_abl_cross @ J_A_cross_unit).item()
                h_resc_cross = h_abl_cross + coeff_cross_from_abl * J_A_cross_unit
                l_resc_cross = mAB.fc2(h_resc_cross.unsqueeze(0))
                m_resc_cross = (l_resc_cross[0, blue_cls] - l_resc_cross[0, red_cls]).item()

                cos_own_cross = cosine_alignment(J_A_own_unit, J_A_cross_unit)

            cross_rescues.append({
                "source_seed": seed_S,
                "cos_own_cross": round(float(cos_own_cross), 4),
                "abl_delta": round(m_abl_cross - m_nat, 4),
                "rescue_delta": round(m_resc_cross - m_abl_cross, 4),
            })

        mean_cross_rescue = float(np.mean([r["rescue_delta"] for r in cross_rescues]))
        mean_cross_abl    = float(np.mean([r["abl_delta"]    for r in cross_rescues]))

        per_seed_results.append({
            "seed_T": seed_T,
            "m_nat": round(m_nat, 4),
            "own_abl_delta":    round(own_abl_delta, 4),
            "own_rescue_delta": round(own_rescue_delta, 4),
            "mean_cross_abl_delta":    round(mean_cross_abl, 4),
            "mean_cross_rescue_delta": round(mean_cross_rescue, 4),
            "own_minus_cross_rescue":  round(own_rescue_delta - mean_cross_rescue, 4),
            "cross_rescues": cross_rescues,
        })

    # ── Statistical tests ──────────────────────────────────────────────────────
    own_rescues   = [r["own_rescue_delta"]        for r in per_seed_results]
    cross_rescues_mean = [r["mean_cross_rescue_delta"] for r in per_seed_results]
    gaps          = [r["own_minus_cross_rescue"]   for r in per_seed_results]

    # 1. One-sample t-test: own_rescue_delta vs 0
    t_own, p_own = stats.ttest_1samp(own_rescues, 0)
    # 2. Paired t-test: own vs cross
    t_pair, p_pair = stats.ttest_rel(own_rescues, cross_rescues_mean)
    # 3. Cohen's d for the gap
    d_gap = np.mean(gaps) / (np.std(gaps, ddof=1) + 1e-9)
    # 4. Bootstrap CI for the gap (10k resamples)
    np.random.seed(0)
    boot_gaps = [np.mean(np.random.choice(gaps, len(gaps), replace=True))
                 for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
    # 5. Permutation test: shuffle own/cross labels, recompute mean gap
    all_rescue_vals = own_rescues + cross_rescues_mean
    observed_gap = np.mean(gaps)
    np.random.seed(42)
    perm_gaps = []
    for _ in range(10000):
        perm = np.random.permutation(len(per_seed_results) * 2)
        g1   = [all_rescue_vals[perm[i]] for i in range(len(per_seed_results))]
        g2   = [all_rescue_vals[perm[i]] for i in range(len(per_seed_results),
                                                          len(per_seed_results) * 2)]
        perm_gaps.append(np.mean(np.array(g1) - np.array(g2)))
    p_perm = np.mean(np.array(perm_gaps) >= observed_gap)

    return {
        "per_seed": per_seed_results,
        "n_valid_seeds": len(valid_seeds),
        "mean_own_rescue":   round(float(np.mean(own_rescues)),        4),
        "std_own_rescue":    round(float(np.std(own_rescues, ddof=1)), 4),
        "mean_cross_rescue": round(float(np.mean(cross_rescues_mean)),        4),
        "std_cross_rescue":  round(float(np.std(cross_rescues_mean, ddof=1)), 4),
        "mean_gap":          round(float(np.mean(gaps)),        4),
        "std_gap":           round(float(np.std(gaps, ddof=1)), 4),
        "t_own_vs_0":        round(float(t_own),  4),
        "p_own_vs_0":        round(float(p_own),  6),
        "t_own_vs_cross":    round(float(t_pair), 4),
        "p_own_vs_cross":    round(float(p_pair), 6),
        "cohens_d_gap":      round(float(d_gap),  4),
        "ci_95_lo":          round(float(ci_lo),  4),
        "ci_95_hi":          round(float(ci_hi),  4),
        "p_permutation":     round(float(p_perm), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXP C — Persistent computation vs persistent alignment/parameterisation
# ═══════════════════════════════════════════════════════════════════════════════

def exp_C(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    red_cls  = COLOR2ID["red"]
    blue_cls = COLOR2ID["blue"]

    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    mAB = train_B(mA.state_dict(), fm, seed)
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # ── Build three independent directions ────────────────────────────────────
    # d1: J_A (readout Jacobian at mA) — the standard direction
    d1_JA = jacobian_zor_red_vs_blue(mA, "CTX_RED")
    d1_unit = d1_JA / (d1_JA.norm() + 1e-9)

    # d2: Linear probe on filler hiddens at mA (no fc2, no zor)
    d2_probe, probe_acc = fit_probe_direction(mA, fm, "CTX_RED")

    # d3: Mean-difference steering vector computed at mAB (no mA, no J_A, no readout)
    #     mean(h_red_fillers in mAB) - mean(h_nonred_fillers in mAB)
    red_fillers    = [o for o in FILLER_OBJECTS if fm[o] == "red"]
    nonred_fillers = [o for o in FILLER_OBJECTS if fm[o] != "red"]
    with torch.no_grad():
        h_red_list = [mAB.hidden(torch.tensor([OBJ2ID[o]], dtype=torch.long), cid).squeeze(0)
                      for o in red_fillers]
        h_nonred_list = [mAB.hidden(torch.tensor([OBJ2ID[o]], dtype=torch.long), cid).squeeze(0)
                         for o in nonred_fillers]
        if h_red_list and h_nonred_list:
            d3_raw = torch.stack(h_red_list).mean(0) - torch.stack(h_nonred_list).mean(0)
            d3_unit = d3_raw / (d3_raw.norm() + 1e-9)
        else:
            d3_unit = None

    dirs = {"d1_JA": d1_unit, "d2_probe": d2_probe, "d3_steer": d3_unit}

    # ── C1: Representational probe — does h_zor in mAB have larger d2 projection ──
    # than non-red objects? No readout involved.
    with torch.no_grad():
        h_zor_AB = mAB.hidden(zid, cid).squeeze(0)

    filler_projs_d2 = get_filler_projections(mAB, fm, d2_probe)
    proj_zor_d2 = (h_zor_AB @ d2_probe).item()
    proj_red_d2    = [filler_projs_d2[o] for o in red_fillers]
    proj_nonred_d2 = [filler_projs_d2[o] for o in nonred_fillers]

    # Is zor's projection above the red-filler mean?
    zor_vs_red_gap    = proj_zor_d2 - np.mean(proj_red_d2) if proj_red_d2 else None
    zor_vs_nonred_gap = proj_zor_d2 - np.mean(proj_nonred_d2)

    # Rank: what fraction of all fillers have smaller d2-projection than zor?
    all_filler_projs = list(filler_projs_d2.values())
    rank_pct = np.mean([v < proj_zor_d2 for v in all_filler_projs]) * 100

    # ── C2: Causal steering — add d3 to non-red fillers, measure red shift ──
    # Completely disjoint from zor and from J_A.
    steer_results = []
    if d3_unit is not None:
        for obj in nonred_fillers[:4]:
            oid = torch.tensor([OBJ2ID[obj]], dtype=torch.long)
            true_cls = COLOR2ID[fm[obj]]
            with torch.no_grad():
                h_obj = mAB.hidden(oid, cid).squeeze(0)
                l_nat_obj = mAB.fc2(h_obj.unsqueeze(0))
                m_nat_obj = (l_nat_obj[0, red_cls] - l_nat_obj[0, true_cls]).item()
                for alpha in [2.0, 5.0, 10.0]:
                    h_s = h_obj + alpha * d3_unit
                    l_s = mAB.fc2(h_s.unsqueeze(0))
                    m_s = (l_s[0, red_cls] - l_s[0, true_cls]).item()
                    steer_results.append({
                        "obj": obj, "alpha": alpha,
                        "red_shift": round(m_s - m_nat_obj, 4),
                        "pred": l_s.argmax(-1).item(),
                    })

    # Does h_zor_AB have a large d3-projection relative to non-red fillers?
    proj_zor_d3 = (h_zor_AB @ d3_unit).item() if d3_unit is not None else None
    filler_projs_d3 = get_filler_projections(mAB, fm, d3_unit) if d3_unit is not None else {}
    proj_red_d3    = [filler_projs_d3[o] for o in red_fillers]    if filler_projs_d3 else []
    proj_nonred_d3 = [filler_projs_d3[o] for o in nonred_fillers] if filler_projs_d3 else []

    # ── C3: Cross-method convergence ──────────────────────────────────────────
    # Do all three directions agree on sign and rank of zor's projection?
    convergence = {}
    for name, d in dirs.items():
        if d is None:
            convergence[name] = None; continue
        proj_zor = (h_zor_AB @ d).item()
        filler_p = get_filler_projections(mAB, fm, d)
        rank_pct_d = np.mean([v < proj_zor for v in filler_p.values()]) * 100
        proj_red_d    = [filler_p[o] for o in red_fillers]
        proj_nonred_d = [filler_p[o] for o in nonred_fillers]
        if proj_red_d and proj_nonred_d:
            t_rn, p_rn = stats.ttest_ind(proj_red_d, proj_nonred_d, alternative="greater")
        else:
            t_rn, p_rn = float("nan"), float("nan")

        # Ablation effect of this direction on mAB B-output (for cross-method comparison)
        d_abl, m_nat_d = ablation_delta(mAB, SPECIAL_OBJECT, "CTX_RED", d, blue_cls, red_cls)

        convergence[name] = {
            "proj_zor":        round(proj_zor, 4),
            "rank_pct":        round(rank_pct_d, 1),
            "mean_proj_red":   round(np.mean(proj_red_d), 4) if proj_red_d else None,
            "mean_proj_nonred":round(np.mean(proj_nonred_d), 4) if proj_nonred_d else None,
            "t_red_gt_nonred": round(float(t_rn), 4),
            "p_red_gt_nonred": round(float(p_rn), 6),
            "abl_delta_on_B":  round(d_abl, 4),
            "cos_with_d1":     round(cosine_alignment(d, d1_unit), 4),
        }

    # Pairwise cos between all three directions
    pairwise = {}
    dir_items = [(k, v) for k, v in dirs.items() if v is not None]
    for i, (n1, v1) in enumerate(dir_items):
        for j, (n2, v2) in enumerate(dir_items):
            if j <= i: continue
            pairwise[f"{n1}|{n2}"] = round(cosine_alignment(v1, v2), 4)

    return {
        "seed": seed, "status": "OK",
        "probe_acc_A": round(probe_acc, 4),
        # C1: representational probe
        "proj_zor_d2":        round(proj_zor_d2, 4),
        "mean_proj_red_d2":   round(np.mean(proj_red_d2), 4) if proj_red_d2 else None,
        "mean_proj_nonred_d2":round(np.mean(proj_nonred_d2), 4),
        "zor_vs_red_gap_d2":  round(zor_vs_red_gap, 4) if zor_vs_red_gap is not None else None,
        "zor_vs_nonred_gap_d2": round(zor_vs_nonred_gap, 4),
        "rank_pct_d2": round(rank_pct, 1),
        # C2: causal steering
        "proj_zor_d3": round(proj_zor_d3, 4) if proj_zor_d3 is not None else None,
        "mean_proj_red_d3":    round(np.mean(proj_red_d3), 4) if proj_red_d3 else None,
        "mean_proj_nonred_d3": round(np.mean(proj_nonred_d3), 4) if proj_nonred_d3 else None,
        "mean_red_shift_alpha5": round(np.mean([r["red_shift"] for r in steer_results
                                                 if r["alpha"] == 5.0]), 4) if steer_results else None,
        "steer_results": steer_results,
        # C3: cross-method convergence
        "convergence": convergence,
        "pairwise_cos": pairwise,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    results_A, results_C = [], []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        rA = exp_A(seed)
        results_A.append(rA)
        if rA["status"] == "OK":
            print(f"  A: probe_acc={rA['probe_acc_A']:.2f} cos_P_JA={rA['cos_P_JA']:+.3f} "
                  f"abl_P={rA['abl_P_delta']:+.3f} ctrl={rA['abl_ctrl_delta']:+.3f} "
                  f"spec={rA['specificity']:.1f}x  r_pb={rA['r_pb_color_proj']:+.3f} "
                  f"p_pb={rA['p_pb']:.4f}", flush=True)
        rC = exp_C(seed)
        results_C.append(rC)
        if rC["status"] == "OK":
            conv = rC["convergence"]
            print(f"  C: rank_d2={rC['rank_pct_d2']:.0f}%  "
                  f"abl_d1={conv['d1_JA']['abl_delta_on_B']:+.3f}  "
                  f"abl_d2={conv['d2_probe']['abl_delta_on_B']:+.3f}  "
                  f"abl_d3={conv['d3_steer']['abl_delta_on_B']:+.3f}  "
                  f"cos(d1,d2)={rC['pairwise_cos'].get('d1_JA|d2_probe', float('nan')):+.3f}  "
                  f"cos(d1,d3)={rC['pairwise_cos'].get('d1_JA|d3_steer', float('nan')):+.3f}", flush=True)

    print("\n--- Running Exp B (cross-lineage, all seeds) ---", flush=True)
    result_B = exp_B(SEEDS)
    print(f"  B: own_rescue={result_B['mean_own_rescue']:+.3f}±{result_B['std_own_rescue']:.3f}  "
          f"cross_rescue={result_B['mean_cross_rescue']:+.3f}±{result_B['std_cross_rescue']:.3f}  "
          f"gap={result_B['mean_gap']:+.3f}  "
          f"t={result_B['t_own_vs_cross']:.2f}  p={result_B['p_own_vs_cross']:.4f}  "
          f"d={result_B['cohens_d_gap']:.2f}  p_perm={result_B['p_permutation']:.4f}")

    with open("/home/claude/iclr/results/closing_experiments.json", "w") as f:
        json.dump({"exp_A": results_A, "exp_B": result_B, "exp_C": results_C}, f, indent=2)
    print("\nSaved to results/closing_experiments.json")
