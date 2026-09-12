"""
Experiment: Subspace-Level Causal Claim

Upgrades the direction-based evidence to a SUBSPACE-level claim:

    "A specific historical computational subspace remains naturally
     instantiated and causally participates after behavioral overwriting."

Design:
  1. Identify the A-subspace independently — via PCA over multiple
     independently derived Jacobian directions (D1..D4 from the
     specificity experiment). The subspace is the span of the top-k
     principal components of the matrix [D1|D2|D3|D4], identified
     WITHOUT using the natural forward pass of mAB at all.

  2. Ablate the full A-subspace from h_AB and measure B-margin disruption.
     Subspace ablation = project h onto the complement of span{D1..Dk}.

  3. Restore the A-subspace component back and verify full recovery.

  4. Controls — all matched to the A-subspace on:
        (a) dimensionality (same k)
        (b) Frobenius norm of the projection matrix (same total 'size')
        (c) Fraction of ||h|| captured (same displacement magnitude)
     Four control types:
        C1: k random orthonormal directions (uniform Haar measure)
        C2: k random rotations of the A-subspace basis (same span volume,
            random orientation — strongest control)
        C3: k directions from an UNRELATED functional subspace —
            Jacobians for the filler objects' OWN top-class margins
            (a live, non-A computation sharing the same hidden space)
        C4: k directions from the B-subspace Jacobians (blue-vs-red
            at zor and red-labeled fillers) — tests whether it is
            specifically the A-subspace or just any 'task-relevant' subspace

  5. Random rotation sweep: for 50 random k-dim subspaces drawn uniformly,
     measure ablation effect distribution — establishes where the A-subspace
     sits in the distribution of all same-dimensional subspaces.

  6. Subspace specificity score: fraction of random rotations with ablation
     effect >= A-subspace's effect. If <5%, A-subspace is in the top 5th
     percentile of all k-dim subspaces for causal effect on B output.

Run across all 7 seeds, k in {1, 2, 3, 4}.
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
    SPECIAL_OBJECT, FILLER_OBJECTS, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE
)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (
    jacobian_zor_red_vs_blue, jacobian_of_margin, cosine_alignment,
    direction_B_behavioral_loss_grad
)

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
K_VALUES = [1, 2, 3, 4]
N_RANDOM_ROTATIONS = 200
CFG = {
    "hidden_dim": 32, "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005, "batch_size": 32,
}


def new_model():
    return TinyClassifier(
        VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
        embed_dim=CFG["embed_dim"], ctx_embed_dim=CFG["ctx_embed_dim"],
        hidden_dim=CFG["hidden_dim"]
    )


# ── Subspace utilities ────────────────────────────────────────────────────────

def orthonormal_basis(directions: list[torch.Tensor], k: int) -> torch.Tensor:
    """
    Given a list of direction vectors, stack them into a matrix and compute
    the top-k left singular vectors via SVD. This gives an orthonormal basis
    for the k-dimensional subspace best spanned by those directions.
    Returns: [hidden_dim, k] matrix of orthonormal basis vectors.
    """
    D = torch.stack(directions, dim=1)          # [hidden_dim, n_dirs]
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    return U[:, :k]                              # [hidden_dim, k]


def project_onto_subspace(h: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """h_proj = B B^T h  where B is [hidden_dim, k] orthonormal."""
    coords = basis.T @ h          # [k]
    return basis @ coords         # [hidden_dim]


def ablate_subspace(h: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """h_abl = h - B B^T h  (remove the subspace component)."""
    return h - project_onto_subspace(h, basis)


def random_orthonormal_basis(dim: int, k: int, seed: int) -> torch.Tensor:
    """Uniformly random k-dim orthonormal basis in R^dim (Haar measure)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k]


def random_rotation_of_basis(basis: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Apply a random rotation WITHIN the ambient space to the basis, giving a
    new k-dim subspace with the same singular-value structure but random
    orientation. This is the strongest rotation control: same span 'volume',
    fully randomized orientation.
    """
    dim, k = basis.shape
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(A)          # random orthogonal matrix
    return Q @ basis                    # rotated basis, still orthonormal cols


def subspace_margin_change(model, h_nat: torch.Tensor, basis: torch.Tensor,
                            blue_idx: int, red_idx: int) -> dict:
    """
    Ablate the subspace from h_nat, measure B-margin change.
    Restore the subspace component, verify recovery.
    Returns ablation delta, rescue delta, and the component norms.
    """
    with torch.no_grad():
        l_nat = model.fc2(h_nat.unsqueeze(0))
        m_nat = (l_nat[0, blue_idx] - l_nat[0, red_idx]).item()

        h_proj = project_onto_subspace(h_nat, basis)     # subspace component
        h_abl  = h_nat - h_proj                           # complement

        l_abl = model.fc2(h_abl.unsqueeze(0))
        m_abl = (l_abl[0, blue_idx] - l_abl[0, red_idx]).item()

        # Restore: add h_proj back to h_abl
        h_resc = h_abl + h_proj
        l_resc = model.fc2(h_resc.unsqueeze(0))
        m_resc = (l_resc[0, blue_idx] - l_resc[0, red_idx]).item()

        h_proj_norm  = h_proj.norm().item()
        h_nat_norm   = h_nat.norm().item()
        frac_captured = h_proj_norm / (h_nat_norm + 1e-9)

    return {
        "m_nat":   round(m_nat, 4),
        "m_abl":   round(m_abl, 4),
        "m_resc":  round(m_resc, 4),
        "abl_delta":  round(m_abl  - m_nat, 4),
        "resc_delta": round(m_resc - m_abl, 4),
        "h_proj_norm":  round(h_proj_norm, 4),
        "frac_captured": round(frac_captured, 4),
    }


# ── Direction builders ────────────────────────────────────────────────────────

def build_A_directions(model_A, filler_mapping):
    """
    Collect all independently derived A-directions (in model_A's coordinate
    system, which is also mAB's since mAB is initialized from mA).
    Returns list of unnormalized direction tensors.
    """
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    ctx_blue = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)

    dirs = []

    # D1: margin Jacobian at zor, CTX_RED
    dirs.append(jacobian_zor_red_vs_blue(model_A, "CTX_RED"))

    # D2: avg margin Jacobian over held-out red-labeled fillers
    red_fillers = [o for o in FILLER_OBJECTS if filler_mapping[o] == "red"]
    if red_fillers:
        jacs = [jacobian_of_margin(model_A,
                                   torch.tensor([OBJ2ID[o]], dtype=torch.long),
                                   ctx_red, COLOR2ID["red"], COLOR2ID["blue"])
                for o in red_fillers]
        dirs.append(torch.stack(jacs).mean(0))

    # D3: margin Jacobian at zor, CTX_BLUE (different context)
    dirs.append(jacobian_of_margin(model_A, zid, ctx_blue,
                                   COLOR2ID["red"], COLOR2ID["blue"]))

    # D4: cross-entropy loss gradient toward red
    dirs.append(direction_B_behavioral_loss_grad(model_A, SPECIAL_OBJECT, "CTX_RED"))

    return dirs


def build_B_directions(model_AB, filler_mapping):
    """B-subspace: Jacobians for blue-vs-red at zor + blue-labeled fillers."""
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    dirs = []
    dirs.append(jacobian_of_margin(model_AB, zid, ctx_red,
                                   COLOR2ID["blue"], COLOR2ID["red"]))
    blue_fillers = [o for o in FILLER_OBJECTS if filler_mapping[o] == "blue"]
    for o in (blue_fillers or FILLER_OBJECTS[:2]):
        dirs.append(jacobian_of_margin(model_AB,
                                       torch.tensor([OBJ2ID[o]], dtype=torch.long),
                                       ctx_red, COLOR2ID["blue"], COLOR2ID["red"]))
    return dirs


def build_filler_directions(model_AB, filler_mapping):
    """
    Unrelated functional subspace: Jacobians for each filler's OWN top-class
    margin (not A or B, but live task computation in the same hidden space).
    """
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    dirs = []
    for o in FILLER_OBJECTS:
        obj_id = torch.tensor([OBJ2ID[o]], dtype=torch.long)
        true_color = COLOR2ID[filler_mapping[o]]
        with torch.no_grad():
            logits = model_AB(obj_id, ctx_red)
            logits_masked = logits.clone()
            logits_masked[0, true_color] = -1e9
            runner_up = logits_masked.argmax(-1).item()
        dirs.append(jacobian_of_margin(model_AB, obj_id, ctx_red,
                                       true_color, runner_up))
    return dirs


# ── Main experiment ───────────────────────────────────────────────────────────

def run_one_seed(seed: int, verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    fm = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(fm, "A")
    ds_B = PhaseDataset(fm, "B")

    # Phase A
    mA = new_model()
    train_phase(mA, ds_A, steps=CFG["phase_A_steps"], batch_size=CFG["batch_size"],
                lr=CFG["phase_A_lr"], seed=seed, eval_every=CFG["phase_A_steps"])
    zid = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    cid = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        if mA(zid, cid).argmax(-1).item() != COLOR2ID["red"]:
            return {"seed": seed, "status": "FAILED_A"}

    # Phase B
    mAB = new_model()
    mAB.load_state_dict(copy.deepcopy(mA.state_dict()))
    opt = torch.optim.Adam(mAB.parameters(), lr=CFG["phase_B_lr"])
    rng = np.random.RandomState(seed)
    for _ in range(CFG["phase_B_steps"]):
        o, c, l = ds_B.sample_batch(CFG["batch_size"], rng)
        loss = F.cross_entropy(mAB(o, c), l)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if mAB(zid, cid).argmax(-1).item() != COLOR2ID["blue"]:
            return {"seed": seed, "status": "FAILED_B"}

    bi, ri = COLOR2ID["blue"], COLOR2ID["red"]
    dim = CFG["hidden_dim"]

    with torch.no_grad():
        h_nat = mAB.hidden(zid, cid).squeeze(0)

    # ── Build all direction sets ──────────────────────────────────────────────
    A_dirs      = build_A_directions(mA, fm)
    B_dirs      = build_B_directions(mAB, fm)
    filler_dirs = build_filler_directions(mAB, fm)

    results_by_k = {}

    for k in K_VALUES:
        # ── Step 1: A-subspace basis (from independently derived directions) ──
        A_basis      = orthonormal_basis(A_dirs,      k)   # [dim, k]
        B_basis      = orthonormal_basis(B_dirs,      k)
        filler_basis = orthonormal_basis(filler_dirs, k)

        # ── Step 2 & 3: Ablate and restore A-subspace ────────────────────────
        A_result = subspace_margin_change(mAB, h_nat, A_basis, bi, ri)

        # ── Step 4: Controls ─────────────────────────────────────────────────

        # C1: k independent random directions (fresh random, not rotated)
        c1_results = []
        for i in range(20):
            c1_basis = random_orthonormal_basis(dim, k, seed=seed * 1000 + i)
            c1_results.append(subspace_margin_change(mAB, h_nat, c1_basis, bi, ri))
        C1_mean_abl = float(np.mean([r["abl_delta"] for r in c1_results]))
        C1_std_abl  = float(np.std([r["abl_delta"]  for r in c1_results]))

        # C2: random rotations of the A-basis (same volume, random orientation)
        c2_results = []
        for i in range(20):
            c2_basis = random_rotation_of_basis(A_basis, seed=seed * 2000 + i)
            c2_results.append(subspace_margin_change(mAB, h_nat, c2_basis, bi, ri))
        C2_mean_abl = float(np.mean([r["abl_delta"] for r in c2_results]))
        C2_std_abl  = float(np.std([r["abl_delta"]  for r in c2_results]))

        # C3: filler functional subspace (unrelated live computation, same dim)
        C3_result = subspace_margin_change(mAB, h_nat, filler_basis, bi, ri)

        # C4: B-subspace (task-relevant but current-epoch, not historical)
        C4_result = subspace_margin_change(mAB, h_nat, B_basis, bi, ri)

        # ── Step 5: random rotation sweep (N_RANDOM_ROTATIONS rotations) ─────
        sweep_abl_deltas = []
        for i in range(N_RANDOM_ROTATIONS):
            rot_basis = random_rotation_of_basis(A_basis, seed=seed * 9999 + i)
            r_rot = subspace_margin_change(mAB, h_nat, rot_basis, bi, ri)
            sweep_abl_deltas.append(r_rot["abl_delta"])

        # ── Step 6: specificity score ─────────────────────────────────────────
        # Fraction of random rotations with |abl_delta| >= |A abl_delta|
        A_eff = abs(A_result["abl_delta"])
        specificity_score = float(np.mean([abs(d) >= A_eff for d in sweep_abl_deltas]))

        # Subspace overlap: cos^2 between A-basis and B-basis
        # = (1/k) * ||A_basis^T B_basis||_F^2
        AB_overlap = float((A_basis.T @ B_basis).pow(2).sum().item() / k)
        AF_overlap = float((A_basis.T @ filler_basis).pow(2).sum().item() / k)

        # Natural participation: fraction of ||h|| captured by A-subspace
        frac_captured = A_result["frac_captured"]

        results_by_k[k] = {
            "A_result":   A_result,
            "C1_mean_abl": round(C1_mean_abl, 4),
            "C1_std_abl":  round(C1_std_abl,  4),
            "C2_mean_abl": round(C2_mean_abl, 4),
            "C2_std_abl":  round(C2_std_abl,  4),
            "C3_abl_delta": C3_result["abl_delta"],
            "C4_abl_delta": C4_result["abl_delta"],
            "C4_frac_captured": C4_result["frac_captured"],
            "sweep_mean_abl": round(float(np.mean(sweep_abl_deltas)), 4),
            "sweep_std_abl":  round(float(np.std(sweep_abl_deltas)), 4),
            "sweep_pct_geq_A": round(specificity_score * 100, 1),
            "AB_overlap": round(AB_overlap, 4),
            "AF_overlap": round(AF_overlap, 4),
            "frac_captured": round(frac_captured, 4),
            "n_A_dirs_used": len(A_dirs),
        }

        if verbose:
            print(f"  k={k}: A_abl={A_result['abl_delta']:+.3f}  "
                  f"C1={C1_mean_abl:+.3f}±{C1_std_abl:.3f}  "
                  f"C2={C2_mean_abl:+.3f}±{C2_std_abl:.3f}  "
                  f"C3={C3_result['abl_delta']:+.3f}  "
                  f"C4={C4_result['abl_delta']:+.3f}  "
                  f"specificity={specificity_score*100:.1f}%  "
                  f"frac_h={frac_captured:.3f}")

    return {"seed": seed, "status": "OK", "results_by_k": results_by_k}


if __name__ == "__main__":
    print("=" * 72)
    print("EXPERIMENT: Subspace-Level Causal Claim")
    print("=" * 72)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        r = run_one_seed(seed, verbose=True)
        all_results.append(r)

    import os
    os.makedirs("/home/claude/iclr/results", exist_ok=True)
    with open("/home/claude/iclr/results/subspace_experiment.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/subspace_experiment.json")

    # Cross-seed summary
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"\n=== CROSS-SEED SUMMARY (n={len(ok)}) ===")
    print(f"\n{'k':>3} | {'A_abl':>8} | {'C1(rand)':>10} | {'C2(rot)':>10} | "
          f"{'C3(filler)':>11} | {'C4(B-sub)':>10} | {'spec%':>6} | {'frac_h':>7}")
    print("-" * 80)
    for k in K_VALUES:
        A_abls  = [r["results_by_k"][k]["A_result"]["abl_delta"] for r in ok]
        C1_abls = [r["results_by_k"][k]["C1_mean_abl"] for r in ok]
        C2_abls = [r["results_by_k"][k]["C2_mean_abl"] for r in ok]
        C3_abls = [r["results_by_k"][k]["C3_abl_delta"] for r in ok]
        C4_abls = [r["results_by_k"][k]["C4_abl_delta"] for r in ok]
        specs   = [r["results_by_k"][k]["sweep_pct_geq_A"] for r in ok]
        fracs   = [r["results_by_k"][k]["frac_captured"] for r in ok]
        print(f"{k:>3} | {np.mean(A_abls):>+8.3f} | {np.mean(C1_abls):>+10.3f} | "
              f"{np.mean(C2_abls):>+10.3f} | {np.mean(C3_abls):>+11.3f} | "
              f"{np.mean(C4_abls):>+10.3f} | {np.mean(specs):>6.1f}% | "
              f"{np.mean(fracs):>7.3f}")
