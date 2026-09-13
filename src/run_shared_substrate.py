"""
Experiments: Shared Substrate

Four experiments that move from "A-direction is causally present during B"
to "B's computation is implemented *through* the A-substrate".

Exp 1 — Content Transfer Test
  If B reuses A's substrate, then the *content* of A should bleed into B,
  not just the direction. Concretely:
    zor:  A-phase teaches red, B overwrites to blue  → model_AB_zor
    fenn: A-phase teaches green, B overwrites to blue → model_AB_fenn
  Both end up predicting blue. If they share substrate, the J_A direction
  for zor (red-aligned) vs fenn (green-aligned) should be distinguishable
  inside mAB, and patching J_A(zor) into mAB_fenn should shift output
  toward red (not green), and vice versa.

  Killer version: cross-patch hidden states.
    h_zor from mAB_zor, read through mAB_fenn's readout → predicts red or green?
    h_fenn from mAB_fenn, read through mAB_zor's readout → predicts red or green?
  If substrate is shared and content-specific, the cross-readout predicts
  the OTHER model's A-content (red for fenn model, green for zor model).

Exp 2 — B-from-Scratch Subspace Comparison
  Train mB_scratch: phase B from random init (no phase A).
  mAB: the usual A→B model.
  Both learn to predict blue for zor. Do they use the same hidden subspace?
  If B reuses A's substrate, mAB and mB_scratch should have different
  hidden-subspace geometries (mAB uses A's subspace; mB_scratch builds fresh).
  If B ignores A, mAB and mB_scratch should converge to the same subspace.
  Metric: cos(J_AB, J_scratch) — do they use the same direction?
  Also: does J_A (from mA) predict mAB's behavior better than mB_scratch's?

Exp 3 — Targeted A-Weight Modification
  Directly edit the embedding weight for zor (the most A-specific parameter)
  and test whether B's output shifts in a content-specific direction.
  If substrate is shared:
    - zeroing embed[zor] should disrupt B
    - replacing embed[zor] with embed[fenn] should shift B toward fenn's color
    - replacing embed[zor] with a random vector should scramble B
  If B has built an independent substrate:
    - embed[zor] modifications should leave B unchanged (B doesn't use it)
  This is the cleanest weight-level test.

Exp 4 — Gradient Alignment During B-Training
  Track whether B-training gradients are aligned with or orthogonal to J_A.
  If B reuses A's substrate: dL_B/dh should be aligned with J_A throughout
  (B is updating along the same axis A used).
  If B routes around A: dL_B/dh should be orthogonal to J_A after an
  initial transient.
  Metric: cos(grad_B_at_h, J_A) tracked across all B-training steps.
  Also track the gradient w.r.t. the embedding for zor specifically.
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
from src.probe import jacobian_zor_red_vs_blue, jacobian_of_margin, cosine_alignment

SEEDS   = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG = dict(hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
           phase_A_steps=600, phase_A_lr=0.01,
           phase_B_steps=3000, phase_B_lr=0.005, batch_size=32)


def new_model():
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                          embed_dim=CFG["embed_dim"],
                          ctx_embed_dim=CFG["ctx_embed_dim"],
                          hidden_dim=CFG["hidden_dim"])


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


def get_jacobian(model, obj_name, ctx_name, tgt_cls, ref_cls):
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    return jacobian_of_margin(model, oid, cid, tgt_cls, ref_cls)


def margin(model, obj_name, ctx_name, cls_a, cls_b):
    with torch.no_grad():
        oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
        cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
        l   = model(oid, cid)
        return (l[0, cls_a] - l[0, cls_b]).item(), l.argmax(-1).item()


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 1 — Content Transfer
# ═══════════════════════════════════════════════════════════════════════════════

def exp1_content_transfer(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    red_cls   = COLOR2ID["red"]
    blue_cls  = COLOR2ID["blue"]
    green_cls = COLOR2ID["green"]

    # Train mA (teaches zor→red AND fenn→green simultaneously)
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])

    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    fid = torch.tensor([OBJ2ID[SHAM_OBJECT]],   dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]],      dtype=torch.long)

    with torch.no_grad():
        zor_A_pred  = mA(zid, cid).argmax(-1).item() == red_cls
        fenn_A_pred = mA(fid, cid).argmax(-1).item() == green_cls
    if not (zor_A_pred and fenn_A_pred):
        return {"seed": seed, "status": "FAILED_A"}

    # Train mAB (same model, B phase overwrites both zor and fenn to blue)
    mAB = train_B(mA.state_dict(), fm, seed)

    with torch.no_grad():
        zor_B_pred  = mAB(zid, cid).argmax(-1).item() == blue_cls
        fenn_B_pred = mAB(fid, cid).argmax(-1).item() == blue_cls

    # J_A for zor (red vs blue direction) and fenn (green vs blue direction)
    J_zor  = get_jacobian(mA, SPECIAL_OBJECT, "CTX_RED", red_cls,   blue_cls)
    J_fenn = get_jacobian(mA, SHAM_OBJECT,    "CTX_RED", green_cls, blue_cls)
    J_zor_unit  = J_zor  / (J_zor.norm()  + 1e-9)
    J_fenn_unit = J_fenn / (J_fenn.norm() + 1e-9)

    # Alignment between the two A-directions
    cos_zor_fenn = cosine_alignment(J_zor, J_fenn)

    with torch.no_grad():
        h_zor  = mAB.hidden(zid, cid).squeeze(0)
        h_fenn = mAB.hidden(fid, cid).squeeze(0)

        # Natural B margins
        l_zor_nat  = mAB.fc2(h_zor.unsqueeze(0))
        l_fenn_nat = mAB.fc2(h_fenn.unsqueeze(0))
        m_zor_nat  = (l_zor_nat[0,  blue_cls] - l_zor_nat[0,  red_cls]).item()
        m_fenn_nat = (l_fenn_nat[0, blue_cls] - l_fenn_nat[0, green_cls]).item()

        # ── Core test: cross-patch hidden states ─────────────────────────────
        # Read h_zor through mAB's own readout → how much red vs blue?
        # Read h_fenn through mAB's own readout → how much green vs blue?
        red_from_zor_h  = (l_zor_nat[0,  red_cls]   - l_zor_nat[0,  green_cls]).item()
        green_from_fenn_h=(l_fenn_nat[0, green_cls]  - l_fenn_nat[0, red_cls]).item()

        # Cross-read: h_zor through mAB, ask red vs green (content discrimination)
        # "Does the hidden state of zor carry a red-vs-green signal?"
        # This tests content specificity: if substrate is shared, h_zor should
        # carry red > green; h_fenn should carry green > red
        red_vs_green_zor  = (l_zor_nat[0,  red_cls]   - l_zor_nat[0, green_cls]).item()
        green_vs_red_fenn = (l_fenn_nat[0, green_cls]  - l_fenn_nat[0, red_cls]).item()

        # ── Ablation specificity: does removing J_zor disrupt zor but not fenn? ──
        coeff_zor_on_zor  = (h_zor  @ J_zor_unit).item()
        coeff_zor_on_fenn = (h_fenn @ J_zor_unit).item()
        h_zor_abl  = h_zor  - coeff_zor_on_zor  * J_zor_unit
        h_fenn_abl = h_fenn - coeff_zor_on_fenn * J_zor_unit

        l_zor_abl  = mAB.fc2(h_zor_abl.unsqueeze(0))
        l_fenn_abl = mAB.fc2(h_fenn_abl.unsqueeze(0))
        m_zor_abl  = (l_zor_abl[0,  blue_cls] - l_zor_abl[0,  red_cls]).item()
        m_fenn_abl = (l_fenn_abl[0, blue_cls] - l_fenn_abl[0, green_cls]).item()
        zor_abl_delta  = m_zor_abl  - m_zor_nat
        fenn_abl_delta = m_fenn_abl - m_fenn_nat

        # Symmetric: remove J_fenn from both
        coeff_fenn_on_fenn = (h_fenn @ J_fenn_unit).item()
        coeff_fenn_on_zor  = (h_zor  @ J_fenn_unit).item()
        h_fenn_abl2 = h_fenn - coeff_fenn_on_fenn * J_fenn_unit
        h_zor_abl2  = h_zor  - coeff_fenn_on_zor  * J_fenn_unit
        l_fenn_abl2 = mAB.fc2(h_fenn_abl2.unsqueeze(0))
        l_zor_abl2  = mAB.fc2(h_zor_abl2.unsqueeze(0))
        m_fenn_abl2 = (l_fenn_abl2[0, blue_cls] - l_fenn_abl2[0, green_cls]).item()
        m_zor_abl2  = (l_zor_abl2[0,  blue_cls] - l_zor_abl2[0,  red_cls]).item()
        fenn_abl2_delta = m_fenn_abl2 - m_fenn_nat
        zor_abl2_delta  = m_zor_abl2  - m_zor_nat

        # ── Content probe: project h onto (J_zor - J_fenn) ───────────────────
        # This direction discriminates the A-content of zor vs fenn.
        # If B reuses A substrate with content, h_zor should have + projection,
        # h_fenn should have - projection.
        J_diff = J_zor_unit - J_fenn_unit
        J_diff_unit = J_diff / (J_diff.norm() + 1e-9)
        content_proj_zor  = (h_zor  @ J_diff_unit).item()
        content_proj_fenn = (h_fenn @ J_diff_unit).item()
        content_gap = content_proj_zor - content_proj_fenn  # >0 = content-specific

        # ── h_zor projected through J_fenn space and vice versa ──────────────
        # "Does h_zor carry green information? Does h_fenn carry red?"
        # Compute the J_fenn-parallel component of h_zor and ask mAB readout
        coeff_jf_on_zor  = (h_zor @ J_fenn_unit).item()
        h_zor_in_fenn_space = coeff_jf_on_zor * J_fenn_unit
        l_zor_fenn_space = mAB.fc2(h_zor_in_fenn_space.unsqueeze(0))
        green_signal_in_zor = (l_zor_fenn_space[0, green_cls] - l_zor_fenn_space[0, red_cls]).item()

        coeff_jz_on_fenn = (h_fenn @ J_zor_unit).item()
        h_fenn_in_zor_space = coeff_jz_on_fenn * J_zor_unit
        l_fenn_zor_space = mAB.fc2(h_fenn_in_zor_space.unsqueeze(0))
        red_signal_in_fenn = (l_fenn_zor_space[0, red_cls] - l_fenn_zor_space[0, green_cls]).item()

    return {
        "seed": seed, "status": "OK",
        "zor_B_correct": bool(zor_B_pred),
        "fenn_B_correct": bool(fenn_B_pred),
        "cos_JA_zor_fenn": round(float(cos_zor_fenn), 4),
        # Natural B margins
        "m_zor_nat":  round(m_zor_nat,  4),
        "m_fenn_nat": round(m_fenn_nat, 4),
        # Content discrimination in natural hidden states
        "red_vs_green_in_zor_h":  round(red_vs_green_zor,   4),
        "green_vs_red_in_fenn_h": round(green_vs_red_fenn,  4),
        "content_proj_zor":       round(content_proj_zor,   4),
        "content_proj_fenn":      round(content_proj_fenn,  4),
        "content_gap":            round(content_gap,        4),
        # Cross-space projection signals
        "green_signal_in_zor_h":  round(green_signal_in_zor, 4),
        "red_signal_in_fenn_h":   round(red_signal_in_fenn,  4),
        # Ablation specificity: J_zor ablation hits zor more than fenn?
        "zor_abl_delta_from_J_zor":  round(zor_abl_delta,   4),
        "fenn_abl_delta_from_J_zor": round(fenn_abl_delta,  4),
        "specificity_J_zor": round(abs(zor_abl_delta) / (abs(fenn_abl_delta) + 1e-9), 2),
        # Symmetric: J_fenn ablation
        "fenn_abl_delta_from_J_fenn": round(fenn_abl2_delta, 4),
        "zor_abl_delta_from_J_fenn":  round(zor_abl2_delta,  4),
        "specificity_J_fenn": round(abs(fenn_abl2_delta) / (abs(zor_abl2_delta) + 1e-9), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 2 — B-from-Scratch Subspace Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def exp2_scratch_comparison(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    ds_B_only = PhaseDataset(fm, "B_only")

    blue_cls = COLOR2ID["blue"]
    red_cls  = COLOR2ID["red"]

    # mA
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != COLOR2ID["red"]:
            return {"seed": seed, "status": "FAILED_A"}

    J_A = get_jacobian(mA, SPECIAL_OBJECT, "CTX_RED", red_cls, blue_cls)

    # mAB: initialized from mA, then B-trained
    mAB = train_B(mA.state_dict(), fm, seed)

    # mB_scratch: B-trained from RANDOM INIT (no phase A)
    torch.manual_seed(seed + 9000); np.random.seed(seed + 9000)
    mB_scratch = new_model()
    opt = torch.optim.Adam(mB_scratch.parameters(), lr=CFG["phase_B_lr"])
    rng = np.random.RandomState(seed + 9000)
    # Train twice as long to ensure convergence (no A warmup)
    for _ in range(CFG["phase_B_steps"] * 2):
        o, c, l = ds_B_only.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(mB_scratch(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        mAB_blue    = mAB(zid,      cid).argmax(-1).item() == blue_cls
        scratch_blue= mB_scratch(zid, cid).argmax(-1).item() == blue_cls

    # J direction in each final model (how each model reads out blue vs red for zor)
    J_AB      = get_jacobian(mAB,      SPECIAL_OBJECT, "CTX_RED", blue_cls, red_cls)
    J_scratch = get_jacobian(mB_scratch, SPECIAL_OBJECT, "CTX_RED", blue_cls, red_cls)

    # Alignment
    cos_JA_JAB     = cosine_alignment(J_A,  J_AB)
    cos_JA_Jscr    = cosine_alignment(J_A,  J_scratch)
    cos_JAB_Jscr   = cosine_alignment(J_AB, J_scratch)

    with torch.no_grad():
        h_AB     = mAB.hidden(zid,       cid).squeeze(0)
        h_scratch= mB_scratch.hidden(zid, cid).squeeze(0)

        # Natural B margins for each
        l_AB_nat     = mAB(zid, cid)
        l_scr_nat    = mB_scratch(zid, cid)
        m_AB_nat     = (l_AB_nat[0, blue_cls]    - l_AB_nat[0, red_cls]).item()
        m_scr_nat    = (l_scr_nat[0, blue_cls]   - l_scr_nat[0, red_cls]).item()

        # J_A ablation effect on mAB vs mB_scratch
        J_A_unit = J_A / (J_A.norm() + 1e-9)

        # mAB
        coeff_AB = (h_AB @ J_A_unit).item()
        h_AB_abl = h_AB - coeff_AB * J_A_unit
        l_AB_abl = mAB.fc2(h_AB_abl.unsqueeze(0))
        m_AB_abl = (l_AB_abl[0, blue_cls] - l_AB_abl[0, red_cls]).item()
        abl_delta_AB = m_AB_abl - m_AB_nat

        # mB_scratch: ablate J_A (from mA!) from scratch model's hidden state
        coeff_scr = (h_scratch @ J_A_unit).item()
        h_scr_abl = h_scratch - coeff_scr * J_A_unit
        l_scr_abl = mB_scratch.fc2(h_scr_abl.unsqueeze(0))
        m_scr_abl = (l_scr_abl[0, blue_cls] - l_scr_abl[0, red_cls]).item()
        abl_delta_scr = m_scr_abl - m_scr_nat

        # Hidden-state geometry: how similar are h_AB and h_scratch?
        cos_h = cosine_alignment(h_AB, h_scratch)

        # J_A projection in each hidden state
        JA_proj_AB  = coeff_AB
        JA_proj_scr = coeff_scr

        # Frac of natural B margin explained by J_A component in each model
        W = mAB.fc2.weight
        rd_AB = W[blue_cls] - W[red_cls]
        frac_AB = (rd_AB @ (coeff_AB * J_A_unit)).item() / (m_AB_nat + 1e-9)

        W_scr = mB_scratch.fc2.weight
        rd_scr = W_scr[blue_cls] - W_scr[red_cls]
        frac_scr = (rd_scr @ (coeff_scr * J_A_unit)).item() / (m_scr_nat + 1e-9)

    return {
        "seed": seed, "status": "OK",
        "mAB_blue": bool(mAB_blue), "scratch_blue": bool(scratch_blue),
        # Jacobian alignment
        "cos_JA_JAB":   round(float(cos_JA_JAB),   4),
        "cos_JA_Jscr":  round(float(cos_JA_Jscr),  4),
        "cos_JAB_Jscr": round(float(cos_JAB_Jscr), 4),
        # Natural margins
        "m_AB_nat":  round(m_AB_nat,  4),
        "m_scr_nat": round(m_scr_nat, 4),
        # J_A ablation effect
        "abl_delta_AB":  round(abl_delta_AB,  4),
        "abl_delta_scr": round(abl_delta_scr, 4),
        # J_A projection in each hidden state
        "JA_proj_AB":  round(JA_proj_AB,  4),
        "JA_proj_scr": round(JA_proj_scr, 4),
        # Fraction of natural B margin from J_A component
        "frac_margin_AB":  round(float(frac_AB),  4),
        "frac_margin_scr": round(float(frac_scr), 4),
        # Hidden-state cosine similarity
        "cos_h_AB_scratch": round(float(cos_h), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 3 — Targeted A-Weight Modification
# ═══════════════════════════════════════════════════════════════════════════════

def exp3_weight_modification(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    blue_cls  = COLOR2ID["blue"]
    red_cls   = COLOR2ID["red"]
    green_cls = COLOR2ID["green"]

    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    fid = torch.tensor([OBJ2ID[SHAM_OBJECT]],   dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    mAB = train_B(mA.state_dict(), fm, seed)

    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

        # Natural B output
        l_nat = mAB(zid, cid)
        m_nat_blue_red   = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()
        m_nat_blue_green = (l_nat[0, blue_cls] - l_nat[0, green_cls]).item()

        # Store original embed[zor]
        embed_zor_orig  = mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]].clone()
        embed_fenn_orig = mAB.embed.weight[OBJ2ID[SHAM_OBJECT]].clone()

        # ── Modification 1: zero out embed[zor] ──────────────────────────────
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = torch.zeros_like(embed_zor_orig)
        l_zero = mAB(zid, cid)
        m_zero_blue_red = (l_zero[0, blue_cls] - l_zero[0, red_cls]).item()
        pred_zero = l_zero.argmax(-1).item()
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = embed_zor_orig.clone()

        # ── Modification 2: replace embed[zor] with embed[fenn] ──────────────
        # If substrate is shared: output should shift toward green (fenn's A content)
        # If substrate is independent: output should not shift content-specifically
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = embed_fenn_orig.clone()
        l_fenn_emb = mAB(zid, cid)
        m_fenn_emb_blue_red   = (l_fenn_emb[0, blue_cls]  - l_fenn_emb[0, red_cls]).item()
        m_fenn_emb_green_red  = (l_fenn_emb[0, green_cls] - l_fenn_emb[0, red_cls]).item()
        pred_fenn_emb = l_fenn_emb.argmax(-1).item()
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = embed_zor_orig.clone()

        # ── Modification 3: replace embed[zor] with random vector ────────────
        g = torch.Generator().manual_seed(seed + 55555)
        rand_emb = torch.randn(embed_zor_orig.shape, generator=g)
        rand_emb = rand_emb / rand_emb.norm() * embed_zor_orig.norm()
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = rand_emb
        l_rand = mAB(zid, cid)
        m_rand_blue_red = (l_rand[0, blue_cls] - l_rand[0, red_cls]).item()
        pred_rand = l_rand.argmax(-1).item()
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = embed_zor_orig.clone()

        # ── Modification 4: interpolate embed[zor] → embed[fenn] ─────────────
        # Track how the blue-vs-red AND blue-vs-green margins shift continuously
        interp_curve = []
        for lam in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            interp_emb = (1 - lam) * embed_zor_orig + lam * embed_fenn_orig
            mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = interp_emb
            l_interp = mAB(zid, cid)
            interp_curve.append({
                "lam": lam,
                "m_blue_red":   round((l_interp[0, blue_cls]  - l_interp[0, red_cls]).item(),   4),
                "m_blue_green": round((l_interp[0, blue_cls]  - l_interp[0, green_cls]).item(), 4),
                "m_green_red":  round((l_interp[0, green_cls] - l_interp[0, red_cls]).item(),   4),
                "pred":         l_interp.argmax(-1).item(),
            })
        mAB.embed.weight[OBJ2ID[SPECIAL_OBJECT]] = embed_zor_orig.clone()

    return {
        "seed": seed, "status": "OK",
        # Natural
        "m_nat_blue_red":   round(m_nat_blue_red,   4),
        "m_nat_blue_green": round(m_nat_blue_green, 4),
        # Zero embed
        "m_zero_blue_red": round(m_zero_blue_red, 4),
        "pred_zero":       pred_zero,
        "zero_delta":      round(m_zero_blue_red - m_nat_blue_red, 4),
        # Fenn embed swap
        "m_fenn_emb_blue_red":  round(m_fenn_emb_blue_red,  4),
        "m_fenn_emb_green_red": round(m_fenn_emb_green_red, 4),
        "pred_fenn_emb": pred_fenn_emb,
        "fenn_swap_blue_delta":  round(m_fenn_emb_blue_red  - m_nat_blue_red,   4),
        "fenn_swap_green_shift": round(m_fenn_emb_green_red, 4),  # >0 = shifted toward green
        # Random embed
        "m_rand_blue_red": round(m_rand_blue_red, 4),
        "pred_rand":       pred_rand,
        "rand_delta":      round(m_rand_blue_red - m_nat_blue_red, 4),
        # Interpolation curve
        "interp_curve": interp_curve,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 4 — Gradient Alignment During B-Training
# ═══════════════════════════════════════════════════════════════════════════════

def exp4_gradient_alignment(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    ds_B = PhaseDataset(fm, "B")

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

    J_A = get_jacobian(mA, SPECIAL_OBJECT, "CTX_RED", red_cls, blue_cls)
    J_A_unit = J_A / (J_A.norm() + 1e-9)

    # B-training with gradient tracking at every eval step
    mAB = new_model()
    mAB.load_state_dict(copy.deepcopy(mA.state_dict()))
    opt = torch.optim.Adam(mAB.parameters(), lr=CFG["phase_B_lr"])
    rng = np.random.RandomState(seed)

    RECORD_EVERY = 50
    trajectory = []

    for step in range(CFG["phase_B_steps"]):
        o, c, l = ds_B.sample_batch(CFG["batch_size"], rng)
        logits = mAB(o, c)
        loss   = F.cross_entropy(logits, l)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % RECORD_EVERY == 0 or step == CFG["phase_B_steps"] - 1:
            with torch.no_grad():
                pred_B = mAB(zid, cid).argmax(-1).item() == blue_cls
                h_nat  = mAB.hidden(zid, cid).squeeze(0)
                l_nat  = mAB.fc2(h_nat.unsqueeze(0))
                m_nat  = (l_nat[0, blue_cls] - l_nat[0, red_cls]).item()
                JA_proj = (h_nat @ J_A_unit).item()

            # Gradient of loss w.r.t. hidden state h at (zor, CTX_RED)
            mAB.zero_grad()
            h_for_grad = mAB.hidden(zid, cid)
            h_for_grad.retain_grad()
            l_grad = mAB.fc2(h_for_grad)
            # Use cross-entropy loss toward blue for zor
            loss_zor = F.cross_entropy(l_grad, torch.tensor([blue_cls]))
            loss_zor.backward()
            grad_h = h_for_grad.grad.squeeze(0).detach()

            # Alignment between gradient and J_A
            cos_grad_JA = cosine_alignment(grad_h, J_A_unit)

            # Gradient of embed[zor] lives in embed_dim=16 space.
            # Project into hidden_dim=32 via fc1.weight[:, :embed_dim]
            embed_grad = mAB.embed.weight.grad
            if embed_grad is not None:
                grad_e = embed_grad[OBJ2ID[SPECIAL_OBJECT]].detach()
                W_fc1_e = mAB.fc1.weight[:, :CFG["embed_dim"]].detach()
                grad_h_from_embed = W_fc1_e @ grad_e
                cos_embed_grad_JA = cosine_alignment(grad_h_from_embed, J_A_unit)
            else:
                cos_embed_grad_JA = float("nan")

            mAB.zero_grad()

            trajectory.append({
                "step":           step,
                "pred_B":         bool(pred_B),
                "m_nat":          round(m_nat, 4),
                "JA_proj":        round(JA_proj, 4),
                "cos_grad_h_JA":  round(float(cos_grad_JA), 4),
                "cos_embed_JA":   round(float(cos_embed_grad_JA), 4),
            })

    # Summary stats: mean alignment before vs after t_flip
    t_flip = next((t["step"] for t in trajectory if t["pred_B"]), None)
    before = [t for t in trajectory if t_flip is None or t["step"] <  t_flip]
    after  = [t for t in trajectory if t_flip is not None and t["step"] >= t_flip]

    def safe_mean(lst, key):
        vals = [x[key] for x in lst if not np.isnan(x[key])]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    return {
        "seed": seed, "status": "OK",
        "t_flip": t_flip,
        "mean_cos_grad_JA_before": safe_mean(before, "cos_grad_h_JA"),
        "mean_cos_grad_JA_after":  safe_mean(after,  "cos_grad_h_JA"),
        "mean_cos_embed_JA_before": safe_mean(before, "cos_embed_JA"),
        "mean_cos_embed_JA_after":  safe_mean(after,  "cos_embed_JA"),
        "final_JA_proj":            safe_mean([trajectory[-1]], "JA_proj"),
        "trajectory": trajectory,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)
    all_results = {"exp1": [], "exp2": [], "exp3": [], "exp4": []}

    print("=" * 70)
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        print("  Exp1: content transfer...", flush=True)
        r1 = exp1_content_transfer(seed)
        all_results["exp1"].append(r1)
        if r1["status"] == "OK":
            print(f"    cos(JA_zor,JA_fenn)={r1['cos_JA_zor_fenn']:.3f}  "
                  f"content_gap={r1['content_gap']:+.3f}  "
                  f"spec_Jzor={r1['specificity_J_zor']:.2f}  "
                  f"spec_Jfenn={r1['specificity_J_fenn']:.2f}")

        print("  Exp2: scratch comparison...", flush=True)
        r2 = exp2_scratch_comparison(seed)
        all_results["exp2"].append(r2)
        if r2["status"] == "OK":
            print(f"    cos(JA,JAB)={r2['cos_JA_JAB']:.3f}  "
                  f"cos(JA,Jscr)={r2['cos_JA_Jscr']:.3f}  "
                  f"cos(JAB,Jscr)={r2['cos_JAB_Jscr']:.3f}  "
                  f"abl_AB={r2['abl_delta_AB']:+.3f}  "
                  f"abl_scr={r2['abl_delta_scr']:+.3f}")

        print("  Exp3: weight modification...", flush=True)
        r3 = exp3_weight_modification(seed)
        all_results["exp3"].append(r3)
        if r3["status"] == "OK":
            print(f"    zero_delta={r3['zero_delta']:+.3f}  "
                  f"fenn_swap_delta={r3['fenn_swap_blue_delta']:+.3f}  "
                  f"green_shift={r3['fenn_swap_green_shift']:+.3f}  "
                  f"rand_delta={r3['rand_delta']:+.3f}")

        print("  Exp4: gradient alignment...", flush=True)
        r4 = exp4_gradient_alignment(seed)
        all_results["exp4"].append(r4)
        if r4["status"] == "OK":
            print(f"    t_flip={r4['t_flip']}  "
                  f"cos_grad_JA before={r4['mean_cos_grad_JA_before']:+.3f}  "
                  f"after={r4['mean_cos_grad_JA_after']:+.3f}  "
                  f"final_JA_proj={r4['final_JA_proj']:+.3f}")

    with open("/home/claude/iclr/results/shared_substrate.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/shared_substrate.json")
