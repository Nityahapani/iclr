"""
Shared-component intervention matrix. Directly tests the two competing
mechanistic models via their distinguishing predictions:

  Model 1 (independent old circuit + gate): removing the B-specific
    component should expose an intact A mechanism; removing the A-specific
    component should not affect B's accessibility.
  Model 2 (shared substrate): removing EITHER specific component, or the
    shared component, should damage BOTH A- and B-accessibility to some
    degree, because they draw on overlapping resources.

We decompose {J_A, J_B} (J_A frozen from theta_A; J_B the model's own
current blue-vs-red direction at theta_T) into a shared bisector direction
and A-specific / B-specific orthogonal remainders. We then ablate each of
the three directions in turn and measure BOTH:
  - A-accessibility: red-vs-blue margin change
  - B-accessibility: blue-vs-red margin change
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (jacobian_zor_red_vs_blue, find_B_mechanism_direction,
                        decompose_shared_and_specific, ablate_direction_and_margin)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def run_shared_component_matrix(seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=20)
    model_AB_T = new_model()
    model_AB_T.load_state_dict(log_AB[-1]["state_dict"])

    J_B = find_B_mechanism_direction(model_AB_T, ctx_name="CTX_RED")

    cos_AB = (J_A / J_A.norm()) @ (J_B / J_B.norm())
    print(f"[{run_name}] cos(J_A, J_B) at theta_T = {cos_AB.item():.4f}")

    shared_dir, A_specific, B_specific = decompose_shared_and_specific(J_A, J_B)
    print(f"[{run_name}] ||shared||={shared_dir.norm().item():.4f} (unit), "
          f"||A_specific||={A_specific.norm().item():.4f}, ||B_specific||={B_specific.norm().item():.4f}")

    directions = {"remove_A_specific": A_specific, "remove_B_specific": B_specific, "remove_shared": shared_dir}
    matrix = {}
    for dir_name, direction in directions.items():
        A_access = ablate_direction_and_margin(model_AB_T, SPECIAL_OBJECT, direction, "red", "blue")
        B_access = ablate_direction_and_margin(model_AB_T, SPECIAL_OBJECT, direction, "blue", "red")
        matrix[dir_name] = {"A_accessibility_change": A_access, "B_accessibility_change": B_access}
        print(f"[{run_name}] {dir_name}: A_accessibility_change={A_access:.4f}, "
              f"B_accessibility_change={B_access:.4f}")

    result = {"run_name": run_name, "seed": seed, "cos_J_A_J_B": cos_AB.item(),
              "shared_norm": shared_dir.norm().item(), "A_specific_norm": A_specific.norm().item(),
              "B_specific_norm": B_specific.norm().item(), "matrix": matrix}
    with open(f"/home/claude/iclr/results/shared_component_matrix_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_shared_component_matrix(s, run_name=f"seed{s}"))
        print()

    print("\n=== SUMMARY (mean across 7 seeds) ===")
    for dir_name in ["remove_A_specific", "remove_B_specific", "remove_shared"]:
        A_vals = np.array([r["matrix"][dir_name]["A_accessibility_change"] for r in results])
        B_vals = np.array([r["matrix"][dir_name]["B_accessibility_change"] for r in results])
        print(f"{dir_name:20s}: A_access change mean={A_vals.mean():7.3f} std={A_vals.std(ddof=1):.3f} | "
              f"B_access change mean={B_vals.mean():7.3f} std={B_vals.std(ddof=1):.3f}")

    with open("/home/claude/iclr/results/shared_component_matrix_summary.json", "w") as f:
        summary = {}
        for dir_name in ["remove_A_specific", "remove_B_specific", "remove_shared"]:
            A_vals = [r["matrix"][dir_name]["A_accessibility_change"] for r in results]
            B_vals = [r["matrix"][dir_name]["B_accessibility_change"] for r in results]
            summary[dir_name] = {"A_accessibility_change": A_vals, "B_accessibility_change": B_vals,
                                   "A_mean": float(np.mean(A_vals)), "B_mean": float(np.mean(B_vals))}
        json.dump(summary, f, indent=2)
