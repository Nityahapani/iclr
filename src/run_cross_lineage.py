"""
Cross-lineage causal transplantation matrix (the strongest remaining test
of persistence vs. generic representational reuse).

For each of 7 seeds i, train theta_A^(i), then continue to M_AB^(i). For
EVERY ordered pair (i, j), patch theta_A^(i)'s own hidden activation for zor
into M_AB^(j) and measure the margin at lambda=1 (full patch), using the
same interpolated_patch_margin machinery already validated.

Diagonal (i==j) reproduces the existing "own" result. The 42 off-diagonal
entries give a real distribution of foreign-lineage transfer strength.

Lineage specificity = (Delta_own - mean(Delta_foreign)) / std(Delta_foreign).
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import interpolated_patch_margin


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def get_hidden(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(obj_id, ctx_id).squeeze(0)


def build_lineage(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    theta_A_state = copy.deepcopy(model_A.state_dict())

    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=20)
    model_AB_T = new_model()
    model_AB_T.load_state_dict(log_AB[-1]["state_dict"])

    h_A = get_hidden(model_A, SPECIAL_OBJECT)
    return model_A, model_AB_T, h_A


def run_cross_lineage_matrix():
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    n = len(SEEDS)

    print("Building all 7 lineages...")
    lineages = {}
    for s in SEEDS:
        print(f"  lineage seed={s}...")
        model_A, model_AB_T, h_A = build_lineage(s)
        lineages[s] = {"model_AB_T": model_AB_T, "h_A": h_A}

    print("\nComputing full 7x7 transplantation matrix...")
    matrix = np.zeros((n, n))
    for i_idx, i in enumerate(SEEDS):
        h_A_i = lineages[i]["h_A"]
        for j_idx, j in enumerate(SEEDS):
            model_AB_j = lineages[j]["model_AB_T"]
            curve = interpolated_patch_margin(model_AB_j, h_A_i, SPECIAL_OBJECT, lambdas=[0.0, 1.0])
            m_at_1 = curve[-1]["margin"]
            matrix[i_idx, j_idx] = m_at_1
            marker = " <-- DIAGONAL (own)" if i == j else ""
            print(f"  J_A^({i}) -> M_AB^({j}): margin={m_at_1:7.3f}{marker}")

    diag = np.diag(matrix)
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_vals = matrix[off_diag_mask]

    print(f"\n=== SUMMARY ===")
    print(f"Diagonal (own lineage), n={n}: mean={diag.mean():.3f} std={diag.std(ddof=1):.3f}")
    print(f"Off-diagonal (foreign lineage), n={len(off_diag_vals)}: "
          f"mean={off_diag_vals.mean():.3f} std={off_diag_vals.std(ddof=1):.3f}")

    lineage_specificity = (diag.mean() - off_diag_vals.mean()) / (off_diag_vals.std(ddof=1) + 1e-9)
    print(f"Lineage specificity = {lineage_specificity:.3f}")

    from scipy.stats import mannwhitneyu
    u_stat, p_val = mannwhitneyu(diag, off_diag_vals)
    print(f"Mann-Whitney U (diagonal vs off-diagonal): U={u_stat:.1f} p={p_val:.6f}")

    n_diag_restore = int((diag < 0).sum())
    n_offdiag_restore = int((off_diag_vals < 0).sum())
    print(f"Diagonal entries restoring red: {n_diag_restore}/{n}")
    print(f"Off-diagonal entries restoring red: {n_offdiag_restore}/{len(off_diag_vals)}")

    result = {
        "seeds": SEEDS,
        "matrix": matrix.tolist(),
        "diagonal_mean": float(diag.mean()), "diagonal_std": float(diag.std(ddof=1)),
        "off_diagonal_mean": float(off_diag_vals.mean()), "off_diagonal_std": float(off_diag_vals.std(ddof=1)),
        "lineage_specificity": float(lineage_specificity),
        "mannwhitney_U": float(u_stat), "mannwhitney_p": float(p_val),
        "n_diag_restore": n_diag_restore, "n_diag_total": n,
        "n_offdiag_restore": n_offdiag_restore, "n_offdiag_total": len(off_diag_vals),
    }
    with open("/home/claude/iclr/results/cross_lineage_matrix.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    run_cross_lineage_matrix()
