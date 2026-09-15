"""
Shared Substrate Experiments — Upstream of the Readout

The gap: current evidence shows J_A (= W_red - W_blue at fc2) remains
causally potent in mAB. This is partially explained by readout geometry
preservation. To establish shared substrate we need evidence that does
NOT reduce to readout alignment.

Four experiments, all operating on fc1/embed — upstream of fc2:

SS1 — Weight-space substrate test
  The A-direction in the WEIGHT space of fc1 is:
    w_A = fc1.weight @ embed.weight[zor]  [32-dim vector]
  This is the column of fc1 activated by zor's embedding — the pre-tanh
  pre-activation direction for zor. It exists entirely within fc1's weight
  space and does not involve fc2 at all.
  Compare w_A in mA vs mAB vs mB_scratch:
    - cos(w_A_mA, w_A_mAB): does B-training preserve fc1's zor-column?
    - cos(w_A_mA, w_A_scratch): does scratch training produce a different column?
  If mAB preserves w_A and scratch does not, the substrate is in fc1.
  Then: ablate w_A's direction from the pre-activation (before tanh),
  observe how much this disrupts h and the B output — entirely upstream
  of fc2.

SS2 — Pre-readout steering
  Construct the A-direction entirely from fc1 + embed, without any fc2 access:
    pre_A = fc1.weight @ embed.weight[zor]   [pre-activation direction]
  At mAB, steer the PRE-ACTIVATION (before tanh) toward/away from pre_A:
    z_steered = z_nat + alpha * pre_A / ||pre_A||
    h_steered = tanh(z_steered)
  Then pass h_steered through fc2 and measure A-output recovery.
  If steering the pre-activation toward the A-direction (constructed with
  NO fc2 access) recovers A confidence, the A-substrate is in fc1, not fc2.
  Control: steer with a random direction of the same norm.

SS3 — Readout transplant test
  Take mAB's fc1 + embed weights, freeze them.
  Train a FRESH fc2 on B-only data (so the readout has no memory of A).
  Then test: does ablating the fc1-derived A-direction from h still
  disrupt the fresh-fc2 B output?
  Repeat with mB_scratch's fc1 + embed + fresh fc2.
  If mAB's upstream weights cause residual A-accessibility even through a
  fresh readout, the substrate is genuinely in fc1/embed, not fc2.
  mB_scratch should show no such effect.

SS4 — Causal mediation decomposition through fc1
  Total causal effect of changing embed[zor]:
    TCE = output(embed_A) - output(embed_B)
  Decompose into:
    Direct effect via fc2 alone (hold h fixed, swap readout): 0 by design
    Mediated effect via fc1→h→fc2:
      NDE = output change when embed[zor] changes but h is held at mAB value
      NIE = output change when only h changes (embed fixed, h set to mA value)
  Compare NIE in mAB vs mB_scratch: if mAB has larger NIE (h carries more
  A-specific information through the fc1 pathway), the substrate is in fc1.
"""

import copy
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, "/home/claude/iclr")

from src.task import (
    make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
    SPECIAL_OBJECT, FILLER_OBJECTS, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import jacobian_zor_red_vs_blue, cosine_alignment

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG = dict(hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
           phase_A_steps=600, phase_A_lr=0.01,
           phase_B_steps=3000, phase_B_lr=0.005, batch_size=32)


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                          embed_dim=CFG["embed_dim"], ctx_embed_dim=CFG["ctx_embed_dim"],
                          hidden_dim=CFG["hidden_dim"])


def train_loop(model, dataset, steps, lr, seed, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.RandomState(seed)
    for _ in range(steps):
        o, c, l = dataset.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(model(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()


def train_B(init_state, fm, seed):
    m = new_model(); m.load_state_dict(copy.deepcopy(init_state))
    train_loop(m, PhaseDataset(fm, "B"), CFG["phase_B_steps"], CFG["phase_B_lr"], seed)
    return m


def train_B_only_scratch(fm, seed):
    """Train from random init on B-only distribution (twice as long for convergence)."""
    m = new_model()
    torch.manual_seed(seed + 9000); np.random.seed(seed + 9000)
    train_loop(m, PhaseDataset(fm, "B_only"), CFG["phase_B_steps"] * 2,
               CFG["phase_B_lr"], seed + 9000)
    return m


def get_pre_activation(model, obj_name, ctx_name):
    """Return the pre-tanh activation z = fc1(cat(embed, ctx_embed)) for (obj, ctx)."""
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        e = model.embed(oid)
        c = model.ctx_embed(cid)
        z = model.fc1(torch.cat([e, c], dim=-1))   # pre-tanh, [1, hidden_dim]
    return z.squeeze(0)


def get_fc1_zor_column(model):
    """fc1.weight @ embed[zor] — the pre-activation direction for zor, shape [hidden_dim]."""
    with torch.no_grad():
        e_zor = model.embed.weight[OBJ2ID[SPECIAL_OBJECT]]   # [embed_dim]
        # fc1 input is [embed; ctx_embed]; embed occupies first embed_dim cols
        W_embed = model.fc1.weight[:, :CFG["embed_dim"]]      # [hidden_dim, embed_dim]
        col = W_embed @ e_zor                                  # [hidden_dim]
    return col


def train_fresh_fc2(frozen_model, fm, seed, steps=2000):
    """
    Freeze fc1 + embeddings of frozen_model, train only a fresh fc2 on B-only data.
    Returns a new model with frozen_model's upstream weights and a freshly trained fc2.
    """
    m = new_model()
    m.load_state_dict(copy.deepcopy(frozen_model.state_dict()))
    # Freeze everything except fc2
    for name, p in m.named_parameters():
        if "fc2" not in name:
            p.requires_grad_(False)
    # Re-initialise fc2
    nn.init.xavier_uniform_(m.fc2.weight)
    nn.init.zeros_(m.fc2.bias)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()),
                           lr=0.01)
    ds = PhaseDataset(fm, "B_only")
    rng = np.random.RandomState(seed + 7777)
    for _ in range(steps):
        o, c, l = ds.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(m(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()
    # Unfreeze
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    red_cls  = COLOR2ID["red"]
    blue_cls = COLOR2ID["blue"]
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    # Phase A
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    # Phase B
    mAB = train_B(mA.state_dict(), fm, seed)
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # B-from-scratch (no phase A)
    torch.manual_seed(seed + 9000); np.random.seed(seed + 9000)
    mB_scratch = train_B_only_scratch(fm, seed)
    with torch.no_grad():
        scratch_blue = mB_scratch(zid, cid).argmax(-1).item() == blue_cls

    # Standard J_A for reference
    J_A     = jacobian_zor_red_vs_blue(mA, "CTX_RED")
    J_A_unit = J_A / (J_A.norm() + 1e-9)

    result = {"seed": seed, "status": "OK", "scratch_blue": bool(scratch_blue)}

    # ══════════════════════════════════════════════════════════════════════════
    # SS1 — Weight-space substrate test
    # ══════════════════════════════════════════════════════════════════════════

    # fc1 zor-column in each model (pre-activation direction for zor, no fc2)
    col_A   = get_fc1_zor_column(mA)
    col_AB  = get_fc1_zor_column(mAB)
    col_scr = get_fc1_zor_column(mB_scratch)

    col_A_u   = col_A   / (col_A.norm()   + 1e-9)
    col_AB_u  = col_AB  / (col_AB.norm()  + 1e-9)
    col_scr_u = col_scr / (col_scr.norm() + 1e-9)

    cos_A_AB  = float(cosine_alignment(col_A_u,   col_AB_u))
    cos_A_scr = float(cosine_alignment(col_A_u,   col_scr_u))
    cos_AB_scr= float(cosine_alignment(col_AB_u,  col_scr_u))

    # Ablate the fc1 zor-column from the PRE-ACTIVATION (before tanh), upstream of fc2
    with torch.no_grad():
        z_nat_AB = get_pre_activation(mAB, SPECIAL_OBJECT, "CTX_RED")
        h_nat_AB = torch.tanh(z_nat_AB)
        l_nat    = mAB.fc2(h_nat_AB.unsqueeze(0))
        m_nat    = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()

        # Ablate col_A from pre-activation
        coeff_z = (z_nat_AB @ col_A_u).item()
        z_abl   = z_nat_AB - coeff_z * col_A_u
        h_abl   = torch.tanh(z_abl)
        l_abl   = mAB.fc2(h_abl.unsqueeze(0))
        m_abl_preact = (l_abl[0, blue_cls] - l_abl[0, red_cls]).item()
        preact_abl_delta = m_abl_preact - m_nat

        # Control: ablate random direction from pre-activation
        g = torch.Generator().manual_seed(seed + 11111)
        v_ctrl = torch.randn(CFG["hidden_dim"], generator=g)
        v_ctrl = v_ctrl - (v_ctrl @ col_A_u) * col_A_u
        v_ctrl = v_ctrl / (v_ctrl.norm() + 1e-9)
        coeff_ctrl = (z_nat_AB @ v_ctrl).item()
        z_ctrl_abl = z_nat_AB - coeff_ctrl * v_ctrl
        h_ctrl_abl = torch.tanh(z_ctrl_abl)
        l_ctrl_abl = mAB.fc2(h_ctrl_abl.unsqueeze(0))
        m_ctrl_preact = (l_ctrl_abl[0, blue_cls] - l_ctrl_abl[0, red_cls]).item()
        ctrl_preact_delta = m_ctrl_preact - m_nat

    # Same on scratch
    with torch.no_grad():
        z_nat_scr = get_pre_activation(mB_scratch, SPECIAL_OBJECT, "CTX_RED")
        h_nat_scr = torch.tanh(z_nat_scr)
        l_nat_scr = mB_scratch.fc2(h_nat_scr.unsqueeze(0))
        m_nat_scr = (l_nat_scr[0, blue_cls] - l_nat_scr[0, red_cls]).item()
        coeff_z_scr = (z_nat_scr @ col_A_u).item()
        z_abl_scr   = z_nat_scr - coeff_z_scr * col_A_u
        h_abl_scr   = torch.tanh(z_abl_scr)
        l_abl_scr   = mB_scratch.fc2(h_abl_scr.unsqueeze(0))
        m_abl_scr   = (l_abl_scr[0, blue_cls] - l_abl_scr[0, red_cls]).item()
        preact_abl_delta_scr = m_abl_scr - m_nat_scr

    result["SS1"] = {
        "cos_colA_colAB":  round(cos_A_AB,   4),
        "cos_colA_colScr": round(cos_A_scr,  4),
        "cos_colAB_colScr":round(cos_AB_scr, 4),
        "preact_abl_delta_AB":   round(preact_abl_delta,     4),
        "preact_ctrl_delta_AB":  round(ctrl_preact_delta,    4),
        "preact_abl_delta_scratch": round(preact_abl_delta_scr, 4),
        "specificity_AB":  round(abs(preact_abl_delta)     / (abs(ctrl_preact_delta)    + 1e-9), 2),
        "AB_vs_scratch_ratio": round(abs(preact_abl_delta) / (abs(preact_abl_delta_scr) + 1e-9), 2),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # SS2 — Pre-readout steering from fc1 only
    # ══════════════════════════════════════════════════════════════════════════
    # Steer the PRE-ACTIVATION toward col_A (constructed with no fc2 access).
    # Measure how much this shifts output toward red in mAB vs mB_scratch.

    steering_results = {}
    for model_name, model in [("mAB", mAB), ("scratch", mB_scratch)]:
        with torch.no_grad():
            z_nat = get_pre_activation(model, SPECIAL_OBJECT, "CTX_RED")
            h_nat_m = torch.tanh(z_nat)
            l_nat_m = model.fc2(h_nat_m.unsqueeze(0))
            m_nat_m = (l_nat_m[0, red_cls] - l_nat_m[0, blue_cls]).item()  # red vs blue

        steer_curve = []
        for alpha in [-4, -2, 0, 2, 4, 8, 16]:
            with torch.no_grad():
                z_s = z_nat + alpha * col_A_u
                h_s = torch.tanh(z_s)
                l_s = model.fc2(h_s.unsqueeze(0))
                m_s = (l_s[0, red_cls] - l_s[0, blue_cls]).item()
                pred_s = l_s.argmax(-1).item()
            steer_curve.append({"alpha": alpha, "m_red_blue": round(m_s, 4),
                                 "shift": round(m_s - m_nat_m, 4),
                                 "pred_red": pred_s == red_cls})

        # Control: steer with random orthogonal direction
        ctrl_curve = []
        for alpha in [-4, -2, 0, 2, 4, 8, 16]:
            with torch.no_grad():
                z_s = z_nat + alpha * v_ctrl
                h_s = torch.tanh(z_s)
                l_s = model.fc2(h_s.unsqueeze(0))
                m_s = (l_s[0, red_cls] - l_s[0, blue_cls]).item()
            ctrl_curve.append({"alpha": alpha, "shift": round(m_s - m_nat_m, 4)})

        steering_results[model_name] = {
            "m_nat_red_blue": round(m_nat_m, 4),
            "steer_curve": steer_curve,
            "ctrl_curve": ctrl_curve,
            "shift_at_alpha8":  next(s["shift"] for s in steer_curve if s["alpha"] == 8),
            "ctrl_at_alpha8":   next(s["shift"] for s in ctrl_curve  if s["alpha"] == 8),
            "recovers_red_at8": next(s["pred_red"] for s in steer_curve if s["alpha"] == 8),
        }

    result["SS2"] = steering_results

    # ══════════════════════════════════════════════════════════════════════════
    # SS3 — Readout transplant test
    # ══════════════════════════════════════════════════════════════════════════
    # Freeze mAB's fc1+embed, train a fresh fc2 on B-only.
    # Then ablate col_A from the pre-activation of this transplanted model.
    # Repeat with mB_scratch's fc1+embed + fresh fc2.
    # If mAB's upstream retains A-substrate, ablation disrupts even the fresh readout.

    mAB_transplant = train_fresh_fc2(mAB, fm, seed, steps=2000)
    mSCR_transplant = train_fresh_fc2(mB_scratch, fm, seed, steps=2000)

    transplant_results = {}
    for model_name, model in [("mAB_transplant", mAB_transplant),
                               ("mSCR_transplant", mSCR_transplant)]:
        with torch.no_grad():
            # Verify transplant predicts blue
            pred_b = model(zid, cid).argmax(-1).item() == blue_cls
            z_nat = get_pre_activation(model, SPECIAL_OBJECT, "CTX_RED")
            h_nat_m = torch.tanh(z_nat)
            l_nat_m = model.fc2(h_nat_m.unsqueeze(0))
            m_nat_m = (l_nat_m[0, blue_cls] - l_nat_m[0, red_cls]).item()

            # Ablate col_A from pre-activation
            coeff_z_m = (z_nat @ col_A_u).item()
            z_abl_m   = z_nat - coeff_z_m * col_A_u
            h_abl_m   = torch.tanh(z_abl_m)
            l_abl_m   = model.fc2(h_abl_m.unsqueeze(0))
            m_abl_m   = (l_abl_m[0, blue_cls] - l_abl_m[0, red_cls]).item()

            # Control
            coeff_ctrl_m = (z_nat @ v_ctrl).item()
            z_ctrl_m     = z_nat - coeff_ctrl_m * v_ctrl
            h_ctrl_m     = torch.tanh(z_ctrl_m)
            l_ctrl_m     = model.fc2(h_ctrl_m.unsqueeze(0))
            m_ctrl_m     = (l_ctrl_m[0, blue_cls] - l_ctrl_m[0, red_cls]).item()

        transplant_results[model_name] = {
            "pred_blue": bool(pred_b),
            "m_nat": round(m_nat_m, 4),
            "abl_delta": round(m_abl_m - m_nat_m, 4),
            "ctrl_delta": round(m_ctrl_m - m_nat_m, 4),
            "specificity": round(abs(m_abl_m - m_nat_m) / (abs(m_ctrl_m - m_nat_m) + 1e-9), 2),
        }

    result["SS3"] = transplant_results

    # ══════════════════════════════════════════════════════════════════════════
    # SS4 — Causal mediation through fc1 (NIE decomposition)
    # ══════════════════════════════════════════════════════════════════════════
    # NIE (Natural Indirect Effect): fix embed[zor] at mAB value,
    # substitute h with mA's hidden state → measures how much A's INTERMEDIATE
    # COMPUTATION (h from mA) shifts the output when read through mAB's fc2.
    #
    # This is the critical test: if A's h carries a different red/blue signal
    # than mAB's h even through the CURRENT readout, there is substrate-level
    # computation preserved in the upstream weights.

    with torch.no_grad():
        h_mA  = mA.hidden(zid, cid).squeeze(0)
        h_mAB = mAB.hidden(zid, cid).squeeze(0)
        h_scr = mB_scratch.hidden(zid, cid).squeeze(0)

        # What does mAB's fc2 say about each hidden state?
        l_mA_through_mAB  = mAB.fc2(h_mA.unsqueeze(0))
        l_mAB_through_mAB = mAB.fc2(h_mAB.unsqueeze(0))
        l_scr_through_mAB = mAB.fc2(h_scr.unsqueeze(0))

        # Red-vs-blue margin when reading different h through mAB's fc2
        m_mA_thru_AB  = (l_mA_through_mAB[0,  red_cls] - l_mA_through_mAB[0,  blue_cls]).item()
        m_mAB_thru_AB = (l_mAB_through_mAB[0, red_cls] - l_mAB_through_mAB[0, blue_cls]).item()
        m_scr_thru_AB = (l_scr_through_mAB[0, red_cls] - l_scr_through_mAB[0, blue_cls]).item()

        # NIE_mA: substituting mA's h into mAB's readout → red shift
        NIE_mA  = m_mA_thru_AB  - m_mAB_thru_AB   # > 0 means mA's h is more red
        NIE_scr = m_scr_thru_AB - m_mAB_thru_AB   # scratch h has no A history

        # Proportion of the A-B behavioral gap explained by h substitution
        # A-B gap = m_mA_thru_mA_fc2 - m_mAB_thru_AB (not directly comparable)
        # More useful: just NIE_mA vs NIE_scr shows A-specific h structure

        # Also: what does mA's fc2 say about mAB's hidden state?
        # (Tests whether mAB's h is readable by mA's readout)
        l_mAB_thru_mA = mA.fc2(h_mAB.unsqueeze(0))
        m_mAB_thru_mA = (l_mAB_thru_mA[0, red_cls] - l_mAB_thru_mA[0, blue_cls]).item()

        # Cross-readout matrix: 2x2 table
        # mA_h/mA_fc2 (original A behavior)
        l_mA_thru_mA  = mA.fc2(h_mA.unsqueeze(0))
        m_mA_thru_mA  = (l_mA_thru_mA[0, red_cls] - l_mA_thru_mA[0, blue_cls]).item()
        # mAB_h/mAB_fc2 (current B behavior)
        # mA_h/mAB_fc2 = NIE test
        # mAB_h/mA_fc2 = tests if current h is still "readable" by A's decoder

        # Also test scratch h through mA's fc2
        l_scr_thru_mA = mA.fc2(h_scr.unsqueeze(0))
        m_scr_thru_mA = (l_scr_thru_mA[0, red_cls] - l_scr_thru_mA[0, blue_cls]).item()

    result["SS4"] = {
        # NIE: how much does substituting A's h (or scratch's h) into mAB's fc2
        # shift the red-vs-blue margin?
        "NIE_mA":   round(NIE_mA,  4),   # > 0 = mA's h is more red under mAB's readout
        "NIE_scr":  round(NIE_scr, 4),   # control: scratch h has no A history
        "NIE_gap":  round(NIE_mA - NIE_scr, 4),  # > 0 = A's h is specifically more red
        # Cross-readout table
        "m_mA_thru_mA_fc2":   round(m_mA_thru_mA,  4),  # should be large positive (A correct)
        "m_mAB_thru_mAB_fc2": round(m_mAB_thru_AB, 4),  # near zero or negative (B correct)
        "m_mA_thru_mAB_fc2":  round(m_mA_thru_AB,  4),  # KEY: mA's h through B's decoder
        "m_mAB_thru_mA_fc2":  round(m_mAB_thru_mA, 4),  # mAB's h through A's decoder
        "m_scr_thru_mAB_fc2": round(m_scr_thru_AB, 4),  # scratch h through B's decoder (ctrl)
        "m_scr_thru_mA_fc2":  round(m_scr_thru_mA, 4),  # scratch h through A's decoder (ctrl)
    }

    return result


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    print("SHARED SUBSTRATE EXPERIMENTS — Upstream of the Readout")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(seed)
        all_results.append(r)

        if r["status"] != "OK":
            print(f"  {r['status']}")
            continue

        ss1 = r["SS1"]
        ss2_AB = r["SS2"]["mAB"]
        ss3_AB = r["SS3"]["mAB_transplant"]
        ss3_SC = r["SS3"]["mSCR_transplant"]
        ss4 = r["SS4"]

        print(f"  SS1: cos(A,AB)={ss1['cos_colA_colAB']:+.3f}  "
              f"cos(A,scr)={ss1['cos_colA_colScr']:+.3f}  "
              f"preact_abl_AB={ss1['preact_abl_delta_AB']:+.3f}  "
              f"preact_abl_scr={ss1['preact_abl_delta_scratch']:+.3f}  "
              f"spec={ss1['specificity_AB']:.1f}x")
        print(f"  SS2: steer@8_AB={ss2_AB['shift_at_alpha8']:+.3f}  "
              f"ctrl@8_AB={ss2_AB['ctrl_at_alpha8']:+.3f}  "
              f"recovers_red={ss2_AB['recovers_red_at8']}")
        print(f"  SS3: abl_AB_transplant={ss3_AB['abl_delta']:+.3f}(spec={ss3_AB['specificity']:.1f}x)  "
              f"abl_SCR_transplant={ss3_SC['abl_delta']:+.3f}(spec={ss3_SC['specificity']:.1f}x)")
        print(f"  SS4: NIE_mA={ss4['NIE_mA']:+.3f}  NIE_scr={ss4['NIE_scr']:+.3f}  "
              f"gap={ss4['NIE_gap']:+.3f}  "
              f"mA_h/mAB_fc2={ss4['m_mA_thru_mAB_fc2']:+.3f}  "
              f"mAB_h/mA_fc2={ss4['m_mAB_thru_mA_fc2']:+.3f}")

    with open("/home/claude/iclr/results/shared_substrate_upstream.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/shared_substrate_upstream.json")

    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== AGGREGATE (n={len(ok)}) ===")

    # SS1
    print("\nSS1 — Weight-space substrate (fc1 zor-column):")
    print(f"  cos(colA, colAB):  {np.mean([r['SS1']['cos_colA_colAB']  for r in ok]):+.3f} "
          f"+/- {np.std([r['SS1']['cos_colA_colAB']  for r in ok]):.3f}")
    print(f"  cos(colA, colScr): {np.mean([r['SS1']['cos_colA_colScr'] for r in ok]):+.3f} "
          f"+/- {np.std([r['SS1']['cos_colA_colScr'] for r in ok]):.3f}")
    print(f"  preact_abl_AB:     {np.mean([r['SS1']['preact_abl_delta_AB']       for r in ok]):+.3f} "
          f"+/- {np.std([r['SS1']['preact_abl_delta_AB']       for r in ok]):.3f}")
    print(f"  preact_abl_scratch:{np.mean([r['SS1']['preact_abl_delta_scratch']   for r in ok]):+.3f} "
          f"+/- {np.std([r['SS1']['preact_abl_delta_scratch']   for r in ok]):.3f}")
    print(f"  specificity_AB:    {np.mean([r['SS1']['specificity_AB']            for r in ok]):.1f}x")

    # SS2
    print("\nSS2 — Pre-readout steering from fc1 col_A:")
    print(f"  shift@alpha=8, mAB:     {np.mean([r['SS2']['mAB']['shift_at_alpha8']     for r in ok]):+.3f}")
    print(f"  ctrl@alpha=8,  mAB:     {np.mean([r['SS2']['mAB']['ctrl_at_alpha8']      for r in ok]):+.3f}")
    print(f"  shift@alpha=8, scratch: {np.mean([r['SS2']['scratch']['shift_at_alpha8'] for r in ok]):+.3f}")
    print(f"  recovers_red@8, mAB:    {sum(r['SS2']['mAB']['recovers_red_at8'] for r in ok)}/{len(ok)}")

    # SS3
    print("\nSS3 — Fresh readout transplant:")
    print(f"  abl_delta mAB_transplant:  {np.mean([r['SS3']['mAB_transplant']['abl_delta']  for r in ok]):+.3f} "
          f"+/- {np.std([r['SS3']['mAB_transplant']['abl_delta']  for r in ok]):.3f}")
    print(f"  abl_delta SCR_transplant:  {np.mean([r['SS3']['mSCR_transplant']['abl_delta'] for r in ok]):+.3f} "
          f"+/- {np.std([r['SS3']['mSCR_transplant']['abl_delta'] for r in ok]):.3f}")

    # SS4
    print("\nSS4 — Causal mediation (NIE decomposition):")
    print(f"  NIE_mA:              {np.mean([r['SS4']['NIE_mA']  for r in ok]):+.3f} +/- {np.std([r['SS4']['NIE_mA']  for r in ok]):.3f}")
    print(f"  NIE_scr:             {np.mean([r['SS4']['NIE_scr'] for r in ok]):+.3f} +/- {np.std([r['SS4']['NIE_scr'] for r in ok]):.3f}")
    print(f"  NIE_gap (mA - scr):  {np.mean([r['SS4']['NIE_gap'] for r in ok]):+.3f} +/- {np.std([r['SS4']['NIE_gap'] for r in ok]):.3f}")
    print(f"  mA_h / mAB_fc2:      {np.mean([r['SS4']['m_mA_thru_mAB_fc2']  for r in ok]):+.3f}")
    print(f"  mAB_h / mA_fc2:      {np.mean([r['SS4']['m_mAB_thru_mA_fc2']  for r in ok]):+.3f}")
    print(f"  scr_h / mAB_fc2:     {np.mean([r['SS4']['m_scr_thru_mAB_fc2'] for r in ok]):+.3f}")
    print(f"  scr_h / mA_fc2:      {np.mean([r['SS4']['m_scr_thru_mA_fc2']  for r in ok]):+.3f}")

    from scipy import stats as st
    NIE_gaps = [r['SS4']['NIE_gap'] for r in ok]
    t, p = st.ttest_1samp(NIE_gaps, 0)
    print(f"  t(NIE_gap vs 0):     t={t:.3f}, p={p:.6f}")
    AB_vs_scr_SS3 = ([r['SS3']['mAB_transplant']['abl_delta'] for r in ok],
                     [r['SS3']['mSCR_transplant']['abl_delta'] for r in ok])
    t3, p3 = st.ttest_rel(*AB_vs_scr_SS3)
    print(f"\nSS3 paired t(AB vs SCR transplant): t={t3:.3f}, p={p3:.6f}")
    SS1_pair = ([r['SS1']['preact_abl_delta_AB'] for r in ok],
                [r['SS1']['preact_abl_delta_scratch'] for r in ok])
    t1, p1 = st.ttest_rel(*SS1_pair)
    print(f"SS1 paired t(AB vs SCR preact_abl): t={t1:.3f}, p={p1:.6f}")
