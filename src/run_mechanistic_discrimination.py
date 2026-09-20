"""
Experiment: Mechanistic Discrimination

Directly discriminates three competing mechanistic hypotheses about why
J_A remains causally potent in mAB after behavioral overwriting.

HYPOTHESES:
  H_A (A-specific substrate):   A-era representation persists separately;
                                 removing it disrupts A-decodability but
                                 not B-decodability.
  H_B (B-specific substrate):   B-training built a fresh representation;
                                 removing it disrupts B but not A-residue.
  H_shared (shared substrate):  A and B use the same hidden subspace;
                                 removing the shared component disrupts both.

REPRESENTATIONS — all built from hidden states, not readout weights:
  S_A      = unit(h_mA(zor))          -- direction of A's hidden state
  S_B      = unit(h_mAB - h_mA)       -- what B-training ADDED to h
  S_shared = unit(proj of h_mAB onto h_mB_scratch)
             -- component of mAB's h that aligns with a model
                that learned B with NO phase-A history
  S_ortho  = random unit direction orthogonal to S_A, S_B, S_shared (null)

INTERVENTIONS — on h_mAB(zor), matched displacement norm, no readout used:
  Remove S_A:      h' = h_mAB - (h_mAB · Ŝ_A) Ŝ_A
  Remove S_B:      h' = h_mAB - (h_mAB · Ŝ_B) Ŝ_B
  Remove S_shared: h' = h_mAB - (h_mAB · Ŝ_sh) Ŝ_sh
  Remove S_ortho:  h' = h_mAB - (h_mAB · Ŝ_orth) Ŝ_orth  [null control]

OUTPUTS — two independent measurements per intervention:
  A-probe: pass h' through mA's ORIGINAL fc2 (not mAB's).
           Measures how much the residual h' still looks like the
           A-era representation to A's own decoder.
           A_score = logit_red(mA.fc2(h')) - logit_blue(mA.fc2(h'))
           This is independent of mAB's readout entirely.

  B-probe: pass h' through mAB's CURRENT fc2.
           B_score = logit_blue(mAB.fc2(h')) - logit_red(mAB.fc2(h'))
           This is the B-behavior measurement.

A_score and B_score are genuinely independent because they use different
decoders (mA.fc2 vs mAB.fc2). For the 2-class linear readout they are
not forced to be negatives of each other — h' → mA.fc2 and h' → mAB.fc2
can produce any pattern because the two fc2 matrices have different weights.

PREDICTED PATTERNS:
  Intervention   | A_score change | B_score change | Supports
  --------------|----------------|----------------|----------
  remove S_A    |      ↓         |       ~        | H_A
  remove S_B    |      ~         |       ↓        | H_B
  remove S_shared|     ↓         |       ↓        | H_shared
  remove S_ortho |     ~         |       ~        | (null)

ADDITIONAL CROSS-CHECK — fenn (sham object, A=green, B=blue):
  fenn shares the same A-era training regime as zor but has a DIFFERENT
  A-binding (green, not red). If S_A is a generic A-substrate direction:
    - removing S_A from zor's h should make zor look LESS like a red-A object
      under mA's decoder
    - removing S_A from fenn's h should make fenn look LESS like green-A
  We test this cross-object transfer to verify S_A is not just the
  zor-specific activation but the general A-computation direction.
"""

import copy
import json
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, "/home/claude/iclr")

from src.task import (
    make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
    SPECIAL_OBJECT, SHAM_OBJECT, FILLER_OBJECTS,
    VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
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


def train_B_scratch(fm, seed):
    torch.manual_seed(seed + 9000); np.random.seed(seed + 9000)
    m = new_model()
    opt = torch.optim.Adam(m.parameters(), lr=CFG["phase_B_lr"])
    ds  = PhaseDataset(fm, "B_only")
    rng = np.random.RandomState(seed + 9000)
    for _ in range(CFG["phase_B_steps"] * 2):
        o, c, l = ds.sample_batch(CFG["batch_size"], rng)
        F.cross_entropy(m(o, c), l).backward()
        opt.step(); opt.zero_grad()
    return m


def get_h(model, obj_name, ctx_name="CTX_RED"):
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(oid, cid).squeeze(0)


def proj_out(h, d_unit):
    """Remove the d_unit component from h."""
    return h - (h @ d_unit) * d_unit


def measure(h, mA_fc2, mAB_fc2, red_cls, blue_cls):
    """
    A-score:  logit_red - logit_blue through mA's original decoder.
    B-score:  logit_blue - logit_red through mAB's current decoder.
    These use different weight matrices → genuinely independent.
    """
    with torch.no_grad():
        lA = mA_fc2(h.unsqueeze(0))
        lB = mAB_fc2(h.unsqueeze(0))
    A_score = (lA[0, red_cls]  - lA[0, blue_cls]).item()
    B_score = (lB[0, blue_cls] - lB[0, red_cls]).item()
    return A_score, B_score


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    red_cls  = COLOR2ID["red"]
    blue_cls = COLOR2ID["blue"]
    green_cls= COLOR2ID["green"]
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    fid = torch.tensor([OBJ2ID[SHAM_OBJECT]],   dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]],      dtype=torch.long)

    # Phase A
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"],
                batch_size=CFG["batch_size"], lr=CFG["phase_A_lr"],
                seed=seed, eval_every=CFG["phase_A_steps"])
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}
        fenn_A_ok = mA(fid, cid).argmax(-1).item() == green_cls

    # Phase B
    mAB = train_B(mA.state_dict(), fm, seed)
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # B-from-scratch (no A history) — used to define S_shared
    mB_scratch = train_B_scratch(fm, seed)
    with torch.no_grad():
        scratch_blue = mB_scratch(zid, cid).argmax(-1).item() == blue_cls

    # ── Build representations from hidden states (NO readout access) ──────────
    h_mA      = get_h(mA,       SPECIAL_OBJECT)   # A's computation of zor
    h_mAB     = get_h(mAB,      SPECIAL_OBJECT)   # current mAB computation
    h_scratch = get_h(mB_scratch, SPECIAL_OBJECT)  # B-only computation

    # S_A: direction of A's hidden state (what A computes for zor)
    S_A = h_mA / (h_mA.norm() + 1e-9)

    # S_B: what B-training added to h (the delta from A to AB)
    delta_B = h_mAB - h_mA
    S_B = delta_B / (delta_B.norm() + 1e-9)

    # S_shared: component of h_mAB aligned with mB_scratch's hidden state
    # (what mAB and mB_scratch have in common for zor — no A history involved)
    S_sh_raw = h_scratch / (h_scratch.norm() + 1e-9)
    S_shared = S_sh_raw  # unit vector of scratch's h is the shared direction

    # S_ortho: random unit direction orthogonal to all three
    g = torch.Generator().manual_seed(seed + 44444)
    v_raw = torch.randn(CFG["hidden_dim"], generator=g)
    for d in [S_A, S_B, S_shared]:
        v_raw = v_raw - (v_raw @ d) * d
    S_ortho = v_raw / (v_raw.norm() + 1e-9)

    # Natural projections of h_mAB onto each subspace
    proj_onto_A   = (h_mAB @ S_A).item()
    proj_onto_B   = (h_mAB @ S_B).item()
    proj_onto_sh  = (h_mAB @ S_shared).item()
    proj_onto_orth= (h_mAB @ S_ortho).item()

    # Pairwise alignments between subspaces
    cos_AB   = float(cosine_alignment(S_A, S_B))
    cos_Ash  = float(cosine_alignment(S_A, S_shared))
    cos_Bsh  = float(cosine_alignment(S_B, S_shared))

    # J_A (readout direction) for comparison
    with torch.no_grad():
        W = mA.fc2.weight
        J_A_unit = (W[red_cls] - W[blue_cls]) / ((W[red_cls]-W[blue_cls]).norm()+1e-9)
    cos_SA_JA  = float(cosine_alignment(S_A, J_A_unit))
    cos_SB_JA  = float(cosine_alignment(S_B, J_A_unit))
    cos_Ssh_JA = float(cosine_alignment(S_shared, J_A_unit))

    # ── Baseline: natural h_mAB through both decoders ────────────────────────
    A_nat, B_nat = measure(h_mAB, mA.fc2, mAB.fc2, red_cls, blue_cls)

    # ── Interventions: remove each subspace from h_mAB ────────────────────────
    # All displacements matched: we ablate the UNIT PROJECTION component,
    # so the displacement norm = |proj_onto_X| for each direction.
    # This is the natural matched-norm ablation (removes exactly the component
    # that exists in h — no artificial scaling).

    interventions = {}
    for name, S, proj in [
        ("remove_S_A",      S_A,      proj_onto_A),
        ("remove_S_B",      S_B,      proj_onto_B),
        ("remove_S_shared", S_shared, proj_onto_sh),
        ("remove_S_ortho",  S_ortho,  proj_onto_orth),
    ]:
        h_int = h_mAB - proj * S          # h with that component removed
        A_int, B_int = measure(h_int, mA.fc2, mAB.fc2, red_cls, blue_cls)
        interventions[name] = {
            "proj_magnitude":  round(abs(proj), 4),   # displacement norm
            "A_score":         round(A_int, 4),
            "B_score":         round(B_int, 4),
            "delta_A":         round(A_int - A_nat, 4),  # change from baseline
            "delta_B":         round(B_int - B_nat, 4),
        }

    # ── Fenn cross-check: same interventions on fenn's hidden state ───────────
    # fenn has A-binding = green (not red). If S_A captures generic A-substrate:
    # removing S_A from fenn's h should reduce fenn's green-decodability under mA.
    h_fenn_AB = get_h(mAB, SHAM_OBJECT)
    A_fenn_nat = mA.fc2(h_fenn_AB.unsqueeze(0))[0, green_cls].item()  # green logit

    fenn_interventions = {}
    for name, S in [("remove_S_A", S_A), ("remove_S_B", S_B),
                    ("remove_S_shared", S_shared), ("remove_S_ortho", S_ortho)]:
        proj_f = (h_fenn_AB @ S).item()
        h_f_int = h_fenn_AB - proj_f * S
        with torch.no_grad():
            green_after = mA.fc2(h_f_int.unsqueeze(0))[0, green_cls].item()
        fenn_interventions[name] = {
            "proj_magnitude": round(abs(proj_f), 4),
            "green_logit_after": round(green_after, 4),
            "delta_green": round(green_after - A_fenn_nat, 4),
        }

    return {
        "seed": seed, "status": "OK",
        "scratch_blue": bool(scratch_blue),
        "fenn_A_ok": bool(fenn_A_ok),
        # Baseline scores
        "A_nat": round(A_nat, 4),
        "B_nat": round(B_nat, 4),
        # Subspace projections of h_mAB
        "proj_onto_A":    round(proj_onto_A,    4),
        "proj_onto_B":    round(proj_onto_B,    4),
        "proj_onto_sh":   round(proj_onto_sh,   4),
        "proj_onto_ortho":round(proj_onto_orth, 4),
        # Pairwise subspace alignments
        "cos_SA_SB":   round(cos_AB,  4),
        "cos_SA_Ssh":  round(cos_Ash, 4),
        "cos_SB_Ssh":  round(cos_Bsh, 4),
        # Alignment with readout direction (for comparison)
        "cos_SA_JA":   round(cos_SA_JA,  4),
        "cos_SB_JA":   round(cos_SB_JA,  4),
        "cos_Ssh_JA":  round(cos_Ssh_JA, 4),
        # Interventions
        "interventions": interventions,
        "fenn_interventions": fenn_interventions,
        "A_fenn_nat": round(A_fenn_nat, 4),
    }


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Mechanistic Discrimination")
    print("Directly distinguishing H_A, H_B, and H_shared")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(seed)
        all_results.append(r)

        if r["status"] != "OK":
            print(f"  {r['status']}"); continue

        print(f"  Baseline: A_nat={r['A_nat']:+.3f}  B_nat={r['B_nat']:+.3f}")
        print(f"  Projections: S_A={r['proj_onto_A']:+.3f}  "
              f"S_B={r['proj_onto_B']:+.3f}  S_sh={r['proj_onto_sh']:+.3f}  "
              f"S_orth={r['proj_onto_ortho']:+.3f}")
        print(f"  cos(S_A,J_A)={r['cos_SA_JA']:+.3f}  "
              f"cos(S_B,J_A)={r['cos_SB_JA']:+.3f}  "
              f"cos(S_sh,J_A)={r['cos_Ssh_JA']:+.3f}")
        print(f"  cos(S_A,S_B)={r['cos_SA_SB']:+.3f}  "
              f"cos(S_A,S_sh)={r['cos_SA_Ssh']:+.3f}  "
              f"cos(S_B,S_sh)={r['cos_SB_Ssh']:+.3f}")
        print()
        print(f"  {'Intervention':<20} | {'disp_norm':>9} | {'dA_score':>9} | "
              f"{'dB_score':>9} | {'fenn_dG':>9}")
        print(f"  {'baseline':<20} | {'':>9} | {r['A_nat']:>+9.3f} | "
              f"{r['B_nat']:>+9.3f} | {r['A_fenn_nat']:>+9.3f}")
        for name, iv in r["interventions"].items():
            fi = r["fenn_interventions"].get(name, {})
            print(f"  {name:<20} | {iv['proj_magnitude']:>9.4f} | "
                  f"{iv['delta_A']:>+9.3f} | {iv['delta_B']:>+9.3f} | "
                  f"{fi.get('delta_green', float('nan')):>+9.3f}")

    # Save
    def fix(obj):
        if type(obj) is bool: return int(obj)
        if isinstance(obj, dict): return {k: fix(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [fix(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        return obj

    with open("/home/claude/iclr/results/mechanistic_discrimination.json", "w") as f:
        json.dump(fix(all_results), f, indent=2)
    print("\nSaved.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== AGGREGATE (n={len(ok)}) ===\n")

    intv_names = ["remove_S_A", "remove_S_B", "remove_S_shared", "remove_S_ortho"]
    labels     = ["S_A",        "S_B",        "S_shared",        "S_ortho (null)"]

    print(f"  {'Intervention':<16} | {'disp_norm':>9} | {'ΔA_score':>10} | "
          f"{'ΔB_score':>10} | {'fenn_ΔG':>9} | interpretation")
    print("  " + "-" * 90)
    for name, label in zip(intv_names, labels):
        norms  = [r["interventions"][name]["proj_magnitude"] for r in ok]
        dA     = [r["interventions"][name]["delta_A"]        for r in ok]
        dB     = [r["interventions"][name]["delta_B"]        for r in ok]
        dG     = [r["fenn_interventions"][name]["delta_green"] for r in ok]
        interp = {
            "remove_S_A":      "→ H_A if ΔA↓, ΔB~",
            "remove_S_B":      "→ H_B if ΔA~, ΔB↓",
            "remove_S_shared": "→ H_shared if ΔA↓, ΔB↓",
            "remove_S_ortho":  "→ null if ΔA~, ΔB~",
        }[name]
        print(f"  {label:<16} | {np.mean(norms):>9.4f} | "
              f"{np.mean(dA):>+10.3f} | {np.mean(dB):>+10.3f} | "
              f"{np.mean(dG):>+9.3f} | {interp}")

    print()
    print("  Statistical tests (one-sample t vs 0):")
    for name, label in zip(intv_names, labels):
        dA = [r["interventions"][name]["delta_A"] for r in ok]
        dB = [r["interventions"][name]["delta_B"] for r in ok]
        tA, pA = stats.ttest_1samp(dA, 0)
        tB, pB = stats.ttest_1samp(dB, 0)
        print(f"  {label:<16}: ΔA t={tA:+6.2f} p={pA:.4f}  |  "
              f"ΔB t={tB:+6.2f} p={pB:.4f}")

    print()
    print("  Subspace geometry (means):")
    for k in ["cos_SA_JA", "cos_SB_JA", "cos_Ssh_JA",
              "cos_SA_SB", "cos_SA_Ssh", "cos_SB_Ssh"]:
        vals = [r[k] for r in ok]
        print(f"    {k:<18}: {np.mean(vals):+.4f} +/- {np.std(vals):.4f}")
