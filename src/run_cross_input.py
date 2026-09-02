"""
Cross-input causal transfer matrix (addresses: "of course J_A(zor) affects
zor, you built it from zor's own output").

Construct J_A independently for SEVERAL phase-A inputs: zor (special,
context-dependent, changes A->B), vex (CONTROL_OBJECT, also context-
dependent, SAME red/blue rule as zor during phase A but never changes), and
2 filler objects mapped to red in phase A (fixed, context-independent,
structurally unrelated to zor's binding).

Test every direction on every input via patching, on M_AB (zor's own
lineage). The meaningful restoration test is target=zor (since only zor's
behavior actually changed); other cells are reported for completeness.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, CONTROL_OBJECT, FILLER_OBJECTS,
                       VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
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


def run_cross_input_experiment(seed: int, run_name: str):
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

    red_fillers = [o for o in FILLER_OBJECTS if filler_mapping[o] == "red"][:2]
    if len(red_fillers) < 2:
        print(f"[{run_name}] WARNING: only {len(red_fillers)} red fillers found")

    input_objects = [SPECIAL_OBJECT, CONTROL_OBJECT] + red_fillers
    print(f"[{run_name}] input objects: {input_objects}")

    h_A_by_input = {obj: get_hidden(model_A, obj) for obj in input_objects}

    matrix = {}
    for x_i in input_objects:
        matrix[x_i] = {}
        for x_j in input_objects:
            h_source = h_A_by_input[x_j]
            curve = interpolated_patch_margin(model_AB_T, h_source, x_i, lambdas=[0.0, 1.0])
            m_at_1 = curve[-1]["margin"]
            matrix[x_i][x_j] = m_at_1

    print(f"[{run_name}] cross-input matrix (rows=target x_i, cols=source direction x_j):")
    header = "target\\source".ljust(14) + "".join(f"{o:>12s}" for o in input_objects)
    print("  " + header)
    for x_i in input_objects:
        row = f"{x_i:14s}" + "".join(f"{matrix[x_i][x_j]:12.3f}" for x_j in input_objects)
        print("  " + row)

    result = {"run_name": run_name, "seed": seed, "input_objects": input_objects, "matrix": matrix}
    with open(f"/home/claude/iclr/results/cross_input_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_cross_input_experiment(s, run_name=f"seed{s}"))
        print()

    zor_diag = []
    zor_offdiag = []
    for r in results:
        m = r["matrix"]
        zor_diag.append(m[SPECIAL_OBJECT][SPECIAL_OBJECT])
        for src in r["input_objects"]:
            if src != SPECIAL_OBJECT:
                zor_offdiag.append(m[SPECIAL_OBJECT][src])

    zor_diag = np.array(zor_diag)
    zor_offdiag = np.array(zor_offdiag)
    print(f"\n=== SUMMARY (target=zor) ===")
    print(f"J_A(zor) -> zor: mean={zor_diag.mean():.3f} std={zor_diag.std(ddof=1):.3f}")
    print(f"J_A(other) -> zor: mean={zor_offdiag.mean():.3f} std={zor_offdiag.std(ddof=1):.3f}")

    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(zor_diag, zor_offdiag)
    print(f"Mann-Whitney U={u:.1f} p={p:.6f}")

    with open("/home/claude/iclr/results/cross_input_summary.json", "w") as f:
        json.dump({
            "zor_diag_mean": float(zor_diag.mean()), "zor_diag_std": float(zor_diag.std(ddof=1)),
            "zor_offdiag_mean": float(zor_offdiag.mean()), "zor_offdiag_std": float(zor_offdiag.std(ddof=1)),
            "mannwhitney_U": float(u), "mannwhitney_p": float(p),
        }, f, indent=2)
