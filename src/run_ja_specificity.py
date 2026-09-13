"""
Experiment: J_A Carries A-Specific Computational Information, Not Merely Output Direction

Addresses the single biggest remaining conceptual vulnerability: is J_A just
an artifact of looking at zor's output gradient, or does it reflect something
genuinely A-specific?

Design: construct six independently derived directions — varying which object,
which context, which mathematical object, and whether the readout is involved
at all — and test whether they:
  (a) converge on the same subspace (pairwise |cos|)
  (b) identify the same naturally active component of h_AB
  (c) have the same causal effect on the B output when ablated
  (d) all rescue the B output when their shared D1 component is restored

Directions:
  D1: margin Jacobian at zor, CTX_RED           (baseline: target obj + target ctx + readout)
  D2: avg margin Jacobian over red-labeled fillers, CTX_RED  (excludes zor entirely)
  D3: margin Jacobian at zor, CTX_BLUE          (same obj, different ctx)
  D4: cross-entropy loss gradient toward red at zor  (different mathematical object)
  D5: diff-in-diff activation direction (h_zor_red - h_zor_blue) - (h_vex_red - h_vex_blue)
  D6: linear probe on filler hidden states only  (never sees zor, no explicit readout)

Additional test (D7): cross-model activation patch — read mA_indep's hidden
  state for zor through mAB's CURRENT readout, to test whether independently
  trained models learn representations compatible with mAB's decoder.

The killer result: if D1=D2=D3 (convergence), it shows the direction is not
an artifact of the specific object or context used to derive it. If D4 also
converges, the mathematical object doesn't matter either. If ablating D2 or D3
disrupts B output as much as D1, the causal claim is object/context-invariant.

Run: python3 -m src.run_ja_specificity
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
    SPECIAL_OBJECT, CONTROL_OBJECT, FILLER_OBJECTS,
    VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (
    jacobian_zor_red_vs_blue, jacobian_of_margin, cosine_alignment,
    build_v_A_diff_in_diff, direction_B_behavioral_loss_grad,
    direction_D_disjoint_inputs,
)

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG = {
    "hidden_dim": 32, "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005, "batch_size": 32,
}


def new_model():
    return TinyClassifier(
        VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
        embed_dim=CFG["embed_dim"], ctx_embed_dim=CFG["ctx_embed_dim"],
        hidden_dim=CFG["hidden_dim"],
    )


def train_phase_B(mA_state, fm, seed):
    mAB = new_model()
    mAB.load_state_dict(copy.deepcopy(mA_state))
    opt = torch.optim.Adam(mAB.parameters(), lr=CFG["phase_B_lr"])
    ds_B = PhaseDataset(fm, "B")
    rng = np.random.RandomState(seed)
    for _ in range(CFG["phase_B_steps"]):
        o, c, l = ds_B.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(mAB(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()
    return mAB


def construct_directions(mA, fm):
    """
    Build all six directions in mA's coordinate system (== mAB's, since
    mAB is initialized from mA). Returns a dict of {name: tensor}.
    """
    zid     = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]],  dtype=torch.long)
    ctx_blue= torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)

    # D1: margin Jacobian at zor, CTX_RED (baseline)
    D1 = jacobian_zor_red_vs_blue(mA, "CTX_RED")

    # D2: average margin Jacobian over red-labeled fillers (excludes zor)
    red_fillers = [o for o in FILLER_OBJECTS if fm[o] == "red"]
    if red_fillers:
        jacs = [jacobian_of_margin(mA,
                                   torch.tensor([OBJ2ID[o]], dtype=torch.long),
                                   ctx_red, COLOR2ID["red"], COLOR2ID["blue"])
                for o in red_fillers]
        D2_raw = torch.stack(jacs).mean(0)
        D2 = D2_raw / (D2_raw.norm() + 1e-9)
    else:
        D2 = None

    # D3: margin Jacobian at zor, CTX_BLUE (different context, same object)
    D3_raw = jacobian_of_margin(mA, zid, ctx_blue,
                                COLOR2ID["red"], COLOR2ID["blue"])
    D3 = D3_raw / (D3_raw.norm() + 1e-9)

    # D4: cross-entropy loss gradient toward red (different mathematical object)
    D4_raw = direction_B_behavioral_loss_grad(mA, SPECIAL_OBJECT, "CTX_RED")
    D4 = D4_raw / (D4_raw.norm() + 1e-9)

    # D5: diff-in-diff activation direction (no readout involved)
    D5, _, _ = build_v_A_diff_in_diff(mA)   # already unit-norm

    # D6: linear probe on filler hidden states only (never sees zor or readout)
    D6_raw = direction_D_disjoint_inputs(mA, "CTX_RED")
    D6 = (D6_raw / (D6_raw.norm() + 1e-9)) if D6_raw is not None else None

    return {
        "D1_target_jacobian":  D1,
        "D2_heldout_fillers":  D2,
        "D3_alt_context":      D3,
        "D4_loss_gradient":    D4,
        "D5_diff_in_diff":     D5,
        "D6_probe_fillers":    D6,
    }


def run_one_seed(seed: int, verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    fm  = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    # Phase A (primary)
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    rid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        if mA(zid, rid).argmax(-1).item() != COLOR2ID["red"]:
            return {"seed": seed, "status": "FAILED_A"}

    # Independent A model (different random init, same task)
    torch.manual_seed(seed + 5000)
    np.random.seed(seed + 5000)
    mA_indep = new_model()
    train_phase(mA_indep, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed + 5000, eval_every=CFG["phase_A_steps"])
    with torch.no_grad():
        indep_ok = mA_indep(zid, rid).argmax(-1).item() == COLOR2ID["red"]

    # Phase B
    torch.manual_seed(seed)
    np.random.seed(seed)
    mAB = train_phase_B(mA.state_dict(), fm, seed)
    bi, ri_idx = COLOR2ID["blue"], COLOR2ID["red"]
    with torch.no_grad():
        if mAB(zid, rid).argmax(-1).item() != COLOR2ID["blue"]:
            return {"seed": seed, "status": "FAILED_B"}

    # ── Build directions (all in mA's coordinate system) ─────────────────────
    dirs = construct_directions(mA, fm)

    # ── Natural hidden state during B behavior ────────────────────────────────
    with torch.no_grad():
        h_AB  = mAB.hidden(zid, rid).squeeze(0)
        l_nat = mAB.fc2(h_AB.unsqueeze(0))
        m_nat = (l_nat[0, bi] - l_nat[0, ri_idx]).item()
    h_norm = h_AB.norm().item()

    # ── Pairwise |cos| between all valid directions ───────────────────────────
    valid = {k: v for k, v in dirs.items() if v is not None}
    names = list(valid.keys())
    cos_matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            cos_matrix[f"{n1}|{n2}"] = round(
                abs(cosine_alignment(valid[n1], valid[n2])), 4
            )

    # ── Natural projection of h_AB onto each direction ────────────────────────
    nat_proj = {}
    for name, d in dirs.items():
        if d is None:
            nat_proj[name] = None
            continue
        d_unit = d / (d.norm() + 1e-9)
        proj   = (h_AB @ d_unit).item()
        nat_proj[name] = {
            "projection":     round(proj, 4),
            "h_parallel_frac": round(abs(proj) / h_norm, 4),
        }

    # ── Ablation effect: remove each direction from h, measure B disruption ───
    abl = {}
    for name, d in dirs.items():
        if d is None:
            abl[name] = None
            continue
        d_unit = d / (d.norm() + 1e-9)
        coeff  = (h_AB @ d_unit).item()
        h_a    = h_AB - coeff * d_unit
        with torch.no_grad():
            l_a = mAB.fc2(h_a.unsqueeze(0))
            m_a = (l_a[0, bi] - l_a[0, ri_idx]).item()
        abl[name] = {"m_abl": round(m_a, 4), "delta": round(m_a - m_nat, 4)}

    # Express each direction's effect as % of D1's
    d1_delta = abl["D1_target_jacobian"]["delta"]
    for name in abl:
        if abl[name] is not None and d1_delta != 0:
            abl[name]["pct_of_D1"] = round(
                abl[name]["delta"] / d1_delta * 100, 1
            )

    # ── Subspace rescue: ablate Di, restore D1 component, check B recovery ────
    D1_unit = dirs["D1_target_jacobian"] / (dirs["D1_target_jacobian"].norm() + 1e-9)
    rescue = {}
    for name, d in dirs.items():
        if d is None or name == "D1_target_jacobian":
            continue
        d_unit    = d / (d.norm() + 1e-9)
        coeff_i   = (h_AB @ d_unit).item()
        h_abl_i   = h_AB - coeff_i * d_unit
        coeff_1   = (h_abl_i @ D1_unit).item()
        h_resc    = h_abl_i + coeff_1 * D1_unit
        with torch.no_grad():
            l_ai  = mAB.fc2(h_abl_i.unsqueeze(0))
            l_rs  = mAB.fc2(h_resc.unsqueeze(0))
            m_ai  = (l_ai[0, bi] - l_ai[0, ri_idx]).item()
            m_rs  = (l_rs[0, bi] - l_rs[0, ri_idx]).item()
        rescue[name] = {
            "m_abl_i":   round(m_ai, 4),
            "m_rescued": round(m_rs, 4),
            "rescue_delta": round(m_rs - m_ai, 4),
            "pct_of_D1_effect": round(
                (m_rs - m_ai) / abs(d1_delta) * 100, 1
            ) if d1_delta != 0 else None,
        }

    # ── D7: cross-model activation patch ─────────────────────────────────────
    # Read mA_indep's hidden state for zor through mAB's current readout.
    # Tests whether an independently trained A model's representation is
    # compatible with mAB's decoder — a different coordinate system entirely.
    m_indep_cross      = None
    pred_indep_cross_blue = None
    if indep_ok:
        with torch.no_grad():
            h_indep = mA_indep.hidden(zid, rid).squeeze(0)
            l_cross = mAB.fc2(h_indep.unsqueeze(0))
            m_indep_cross = (l_cross[0, bi] - l_cross[0, ri_idx]).item()
            pred_indep_cross_blue = l_cross.argmax(-1).item() == bi

    result = {
        "seed":     seed,
        "status":   "OK",
        "indep_ok": indep_ok,
        "m_natural": round(m_nat, 4),
        "m_indep_cross": round(m_indep_cross, 4) if m_indep_cross is not None else None,
        "pred_indep_cross_blue": bool(pred_indep_cross_blue) if pred_indep_cross_blue is not None else None,
        "n_red_fillers": len([o for o in FILLER_OBJECTS if fm[o] == "red"]),
        "cos_matrix": cos_matrix,
        "nat_proj":   nat_proj,
        "abl":        abl,
        "rescue":     rescue,
    }

    if verbose:
        d = lambda k: f"{abl[k]['delta']:+.3f}" if abl.get(k) else "NA"
        print(f"  seed={seed}: "
              f"D1={d('D1_target_jacobian')}  D2={d('D2_heldout_fillers')}  "
              f"D3={d('D3_alt_context')}  D4={d('D4_loss_gradient')}  "
              f"D5={d('D5_diff_in_diff')}  D6={d('D6_probe_fillers')}  "
              f"cross={round(m_indep_cross,3) if m_indep_cross is not None else 'NA'}")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT: J_A Specificity — independently derived directions")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        all_results.append(r)

    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)
    with open("/home/claude/iclr/results/ja_specificity.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/ja_specificity.json")

    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== SUMMARY (n={len(ok)}) ===")
    print("\nMean ablation deltas:")
    for key in ["D1_target_jacobian", "D2_heldout_fillers", "D3_alt_context",
                "D4_loss_gradient", "D5_diff_in_diff", "D6_probe_fillers"]:
        vals = [r["abl"][key]["delta"] for r in ok if r["abl"].get(key)]
        if vals:
            print(f"  {key:30}: {np.mean(vals):+.3f} +/- {np.std(vals):.3f}  (n={len(vals)})")

    print("\nMean pairwise |cos| (averaged across seeds):")
    from collections import defaultdict
    all_pairs = defaultdict(list)
    for r in ok:
        for k, v in r["cos_matrix"].items():
            all_pairs[k].append(v)
    for k in sorted(all_pairs):
        vals = all_pairs[k]
        print(f"  {k:55}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
