"""
Experiment: Readout Reparameterization

Separates:
    HISTORICAL COMPUTATIONAL CONTENT  (in fc1/embed)
    from
    COMPATIBILITY WITH THE CURRENT READOUT  (in fc2)

The core objection to SS4: mA's h reads as red through mAB's fc2 (+8.10)
partly because mAB's fc2 was trained on data that includes zor, so the
fc2 may have re-encoded some A-compatibility. The NIE gap vs scratch is
real (p<0.00001) but the readout is not truly independent of A-history.

Fix: replace fc2 with three independently trained readouts, none of which
has ever seen zor in either phase A or phase B:

READOUT R1 — Filler-only neutral readout
    Trained ONLY on filler color discrimination (no zor, no sham, no control).
    Sees the same filler→color mapping as A and B training, but never zor.
    Objective: can this neutral readout detect more red-content in mAB's
    zor-hidden-state than in mB_scratch's?
    If yes: the content is in h (mAB's fc1 output), not in fc2 compatibility.

READOUT R2 — Permuted-label readout
    Trained on filler discrimination with a RANDOMLY PERMUTED color assignment
    (so red→green, blue→yellow, etc. — a completely different label space).
    This readout has no concept of red vs blue as we defined them.
    After training, we ask: in R2's label space, does mAB's zor-h land
    in the bucket that corresponds to A's true color (red) more than scratch?
    Operationally: find R2's 'red-equivalent' class (the class that
    red-fillers mostly predict), then check if mAB's zor-h lands there.

READOUT R3 — Cross-seed readout
    Trained on a DIFFERENT seed's filler mapping (different color assignments
    to the same objects). The readout geometry is entirely from a different
    task instance — no shared A-history with the model being tested.
    After calibration on filler hiddens only, test zor-content detection.

For each readout, the key comparison is:
    score(mAB's zor-h under Ri) vs score(mB_scratch's zor-h under Ri)
    where 'score' = the readout's red-class logit for zor-h.

If mAB's zor-h consistently scores higher on the red class across all three
independently parameterized readouts — none of which knows about A or B for
zor — that is historical computational content, not readout compatibility.

Control: use SHAM_OBJECT (fenn, A-phase green) to verify the readout
correctly distinguishes red-encoded vs green-encoded objects, confirming
it is a functional red/color detector, not just noise.
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
    SPECIAL_OBJECT, SHAM_OBJECT, FILLER_OBJECTS,
    VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import jacobian_zor_red_vs_blue, cosine_alignment

SEEDS   = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
CFG     = dict(hidden_dim=32, embed_dim=16, ctx_embed_dim=8,
               phase_A_steps=600, phase_A_lr=0.01,
               phase_B_steps=3000, phase_B_lr=0.005, batch_size=32)
R_STEPS = 3000   # steps to train each readout


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
    m   = new_model()
    opt = torch.optim.Adam(m.parameters(), lr=CFG["phase_B_lr"])
    ds  = PhaseDataset(fm, "B_only")
    rng = np.random.RandomState(seed + 9000)
    for _ in range(CFG["phase_B_steps"] * 2):
        o, c, l = ds.sample_batch(CFG["batch_size"], rng)
        F.cross_entropy(m(o, c), l).backward()
        opt.step(); opt.zero_grad()
    return m


def train_readout_on_frozen_upstream(
        upstream_model,    # provides frozen fc1 + embed
        filler_mapping,    # color assignments for fillers ONLY
        seed,
        permute_labels=False,
        foreign_filler_mapping=None,  # if provided, use this mapping instead
        steps=R_STEPS,
):
    """
    Freeze upstream_model's embed + fc1. Train a FRESH fc2 (readout)
    on filler color discrimination only — zor is NEVER included.

    permute_labels: randomly permutes the color→class mapping so the
        readout learns a different label space entirely.
    foreign_filler_mapping: if given, train on a different filler→color
        mapping (cross-seed readout).

    Returns: the model with frozen upstream + trained readout,
             plus the label permutation used (identity if not permuted).
    """
    # Build a fresh fc2 on top of the frozen upstream
    m = new_model()
    m.load_state_dict(copy.deepcopy(upstream_model.state_dict()))
    for name, p in m.named_parameters():
        if "fc2" not in name:
            p.requires_grad_(False)
    nn.init.xavier_uniform_(m.fc2.weight)
    nn.init.zeros_(m.fc2.bias)

    fm_use = foreign_filler_mapping if foreign_filler_mapping else filler_mapping

    # Optional label permutation
    rng_perm = np.random.RandomState(seed + 55555)
    perm = list(range(NUM_CLASSES))
    if permute_labels:
        rng_perm.shuffle(perm)
    perm_tensor = torch.tensor(perm, dtype=torch.long)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, m.parameters()), lr=0.01
    )
    ctx_id = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    rng    = np.random.RandomState(seed + 8888)

    for _ in range(steps):
        # Sample ONLY from fillers — never zor, sham, or control
        obj   = FILLER_OBJECTS[rng.randint(len(FILLER_OBJECTS))]
        color = fm_use[obj]
        label_orig = COLOR2ID[color]
        label      = perm_tensor[label_orig].item()
        oid        = torch.tensor([OBJ2ID[obj]], dtype=torch.long)
        logits     = m(oid, ctx_id)
        loss       = F.cross_entropy(logits, torch.tensor([label]))
        loss.backward(); opt.step(); opt.zero_grad()

    for p in m.parameters():
        p.requires_grad_(True)

    return m, perm


def read_score(model, obj_name, ctx_name, target_cls):
    """Logit of target_cls for (obj, ctx) through model."""
    oid = torch.tensor([OBJ2ID[obj_name]], dtype=torch.long)
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(oid, cid)
    return logits[0, target_cls].item(), logits[0].argmax().item()


def filler_readout_acc(model, filler_mapping, perm, ctx_name="CTX_RED"):
    """Accuracy of the model's readout on fillers under label permutation."""
    correct = 0
    cid = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    perm_t = torch.tensor(perm, dtype=torch.long)
    for obj in FILLER_OBJECTS:
        oid   = torch.tensor([OBJ2ID[obj]], dtype=torch.long)
        label = perm_t[COLOR2ID[filler_mapping[obj]]].item()
        with torch.no_grad():
            pred = model(oid, cid).argmax(-1).item()
        correct += int(pred == label)
    return correct / len(FILLER_OBJECTS)


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fm   = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")

    red_cls   = COLOR2ID["red"]
    blue_cls  = COLOR2ID["blue"]
    green_cls = COLOR2ID["green"]
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    fid = torch.tensor([OBJ2ID[SHAM_OBJECT]],   dtype=torch.long)   # fenn → green
    cid = torch.tensor([CTX2ID["CTX_RED"]],      dtype=torch.long)

    # ── Phase A ───────────────────────────────────────────────────────────────
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"],
                batch_size=CFG["batch_size"], lr=CFG["phase_A_lr"],
                seed=seed, eval_every=CFG["phase_A_steps"])
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != red_cls:
            return {"seed": seed, "status": "FAILED_A"}

    # ── Phase B ───────────────────────────────────────────────────────────────
    mAB = train_B(mA.state_dict(), fm, seed)
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != blue_cls:
            return {"seed": seed, "status": "FAILED_B"}

    # ── B from scratch ────────────────────────────────────────────────────────
    mB_scratch = train_B_scratch(fm, seed)
    with torch.no_grad():
        scratch_blue = mB_scratch(zid, cid).argmax(-1).item() == blue_cls

    # Foreign filler mapping (different seed) for R3
    fm_foreign = make_filler_mapping(seed=(seed + 3) % 7 + 1234)

    results = {"seed": seed, "status": "OK", "scratch_blue": bool(scratch_blue)}

    # ══════════════════════════════════════════════════════════════════════════
    # Train the three independent readouts on top of each upstream model
    # ══════════════════════════════════════════════════════════════════════════

    readout_configs = [
        # (label, permute, foreign_fm, readout_seed_offset)
        ("R1_neutral",   False, None,       0),
        ("R2_permuted",  True,  None,       1),
        ("R3_foreign",   False, fm_foreign, 2),
    ]

    # For each readout type, train on top of mAB and mB_scratch
    for rname, permute, ffm, roff in readout_configs:
        results[rname] = {}

        for mname, upstream in [("mAB", mAB), ("scratch", mB_scratch)]:
            torch.manual_seed(seed + 20000 + roff * 100)
            m_r, perm = train_readout_on_frozen_upstream(
                upstream, fm, seed + roff * 100,
                permute_labels=permute,
                foreign_filler_mapping=ffm,
                steps=R_STEPS,
            )

            # Readout accuracy on fillers (sanity check)
            acc = filler_readout_acc(m_r, ffm if ffm else fm, perm)

            # Find which class the readout assigned to "red" content
            # under permutation/foreign mapping.
            # Strategy: ask the readout what it predicts for a known red filler.
            red_fillers = [o for o in FILLER_OBJECTS if (ffm if ffm else fm)[o] == "red"]

            if red_fillers:
                oid_rf = torch.tensor([OBJ2ID[red_fillers[0]]], dtype=torch.long)
                with torch.no_grad():
                    logits_rf = m_r(oid_rf, cid)
                # The predicted class for a red filler IS the 'red class' in
                # this readout's label space (regardless of permutation).
                red_class_in_readout = logits_rf.argmax(-1).item()
            else:
                # No red fillers for this seed — fall back to permuted red_cls
                perm_t = torch.tensor(perm, dtype=torch.long)
                red_class_in_readout = perm_t[red_cls].item()

            # KEY MEASUREMENT: zor-h's score on the red class in this readout
            zor_logit, zor_pred = read_score(m_r, SPECIAL_OBJECT, "CTX_RED",
                                              red_class_in_readout)

            # Sham (fenn → green) for contrast: should score LOW on red class
            fenn_logit, fenn_pred = read_score(m_r, SHAM_OBJECT, "CTX_RED",
                                                red_class_in_readout)

            # Control: blue class in this readout
            blue_fillers = [o for o in FILLER_OBJECTS if (ffm if ffm else fm)[o] == "blue"]
            if blue_fillers:
                oid_bf = torch.tensor([OBJ2ID[blue_fillers[0]]], dtype=torch.long)
                with torch.no_grad():
                    logits_bf = m_r(oid_bf, cid)
                blue_class_in_readout = logits_bf.argmax(-1).item()
            else:
                perm_t = torch.tensor(perm, dtype=torch.long)
                blue_class_in_readout = perm_t[blue_cls].item()

            zor_blue_logit, _ = read_score(m_r, SPECIAL_OBJECT, "CTX_RED",
                                            blue_class_in_readout)

            # Net red-vs-blue score for zor under this readout
            zor_net = zor_logit - zor_blue_logit

            # Average red logit for red fillers (readout calibration check)
            red_logits_fillers = []
            for o in red_fillers[:4]:
                l, _ = read_score(m_r, o, "CTX_RED", red_class_in_readout)
                red_logits_fillers.append(l)
            mean_red_filler_logit = float(np.mean(red_logits_fillers)) if red_logits_fillers else None

            results[rname][mname] = {
                "filler_acc":             round(acc, 4),
                "red_class_in_readout":   red_class_in_readout,
                "zor_logit_red_class":    round(zor_logit,   4),
                "fenn_logit_red_class":   round(fenn_logit,  4),
                "zor_net_red_minus_blue": round(zor_net,     4),
                "mean_red_filler_logit":  round(mean_red_filler_logit, 4) if mean_red_filler_logit else None,
                "zor_pred_is_red":        bool(zor_pred == red_class_in_readout),
            }

        # Key comparison: mAB vs scratch under this readout
        mAB_net = results[rname]["mAB"]["zor_net_red_minus_blue"]
        scr_net = results[rname]["scratch"]["zor_net_red_minus_blue"]
        results[rname]["gap_mAB_minus_scratch"] = round(mAB_net - scr_net, 4)
        results[rname]["mAB_filler_acc"]      = results[rname]["mAB"]["filler_acc"]
        results[rname]["scratch_filler_acc"]  = results[rname]["scratch"]["filler_acc"]

    return results


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Readout Reparameterization")
    print("Separating historical computational content from readout compatibility")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(seed)
        all_results.append(r)

        if r["status"] != "OK":
            print(f"  {r['status']}")
            continue

        for rname in ["R1_neutral", "R2_permuted", "R3_foreign"]:
            rr = r[rname]
            mAB_net = rr["mAB"]["zor_net_red_minus_blue"]
            scr_net = rr["scratch"]["zor_net_red_minus_blue"]
            acc_AB  = rr["mAB"]["filler_acc"]
            acc_scr = rr["scratch"]["filler_acc"]
            print(f"  {rname}: mAB_net={mAB_net:+.3f}  scr_net={scr_net:+.3f}  "
                  f"gap={mAB_net-scr_net:+.3f}  "
                  f"acc_AB={acc_AB:.2f}  acc_scr={acc_scr:.2f}", flush=True)

    with open("/home/claude/iclr/results/readout_reparameterization.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/readout_reparameterization.json")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== AGGREGATE (n={len(ok)}) ===")

    for rname, label in [("R1_neutral",  "R1 Neutral filler readout"),
                          ("R2_permuted", "R2 Permuted-label readout"),
                          ("R3_foreign",  "R3 Foreign-seed readout")]:
        mAB_nets = [r[rname]["mAB"]["zor_net_red_minus_blue"]     for r in ok]
        scr_nets = [r[rname]["scratch"]["zor_net_red_minus_blue"]  for r in ok]
        gaps     = [r[rname]["gap_mAB_minus_scratch"]              for r in ok]
        accs_AB  = [r[rname]["mAB"]["filler_acc"]                  for r in ok]
        accs_scr = [r[rname]["scratch"]["filler_acc"]              for r in ok]

        t_mAB, p_mAB   = stats.ttest_1samp(mAB_nets, 0)
        t_gap,  p_gap   = stats.ttest_1samp(gaps, 0)
        t_pair, p_pair  = stats.ttest_rel(mAB_nets, scr_nets)
        d_gap           = np.mean(gaps) / (np.std(gaps, ddof=1) + 1e-9)
        np.random.seed(42)
        boot = [np.mean(np.random.choice(gaps, len(gaps), replace=True))
                for _ in range(10000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        print(f"\n  {label}:")
        print(f"    filler_acc mAB:    {np.mean(accs_AB):.3f}  scratch: {np.mean(accs_scr):.3f}")
        print(f"    mAB  zor_net:      {np.mean(mAB_nets):+.3f} +/- {np.std(mAB_nets):.3f}")
        print(f"    scr  zor_net:      {np.mean(scr_nets):+.3f} +/- {np.std(scr_nets):.3f}")
        print(f"    gap (mAB - scr):   {np.mean(gaps):+.3f} +/- {np.std(gaps):.3f}")
        print(f"    t(mAB_net vs 0):   t={t_mAB:.3f}, p={p_mAB:.6f}")
        print(f"    t(gap vs 0):       t={t_gap:.3f},  p={p_gap:.6f}")
        print(f"    t(paired mAB/scr): t={t_pair:.3f}, p={p_pair:.6f}")
        print(f"    Cohen d (gap):     {d_gap:.3f}")
        print(f"    95% CI gap:        [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"    mAB pred_red:      {sum(r[rname]['mAB']['zor_pred_is_red'] for r in ok)}/{len(ok)}")
        print(f"    scr pred_red:      {sum(r[rname]['scratch']['zor_pred_is_red'] for r in ok)}/{len(ok)}")
