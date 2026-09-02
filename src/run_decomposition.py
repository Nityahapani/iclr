"""
Controlled activation-patching decomposition (the identifiability test).

For each seed, at M_AB's final checkpoint, sweep lambda in [0,1] for:
  h(lambda) = (1-lambda)*h_AB + lambda*h_source
across FIVE h_source conditions:
  1. own theta_A's full activation
  2. foreign theta_A's full activation (different seed)
  3. matched random activation (same norm as h_A)
  4. own theta_A's PARALLEL (A-relevant) component only, added to M_AB's own
     orthogonal remainder
  5. own theta_A's ORTHOGONAL remainder only, added to M_AB's own parallel
     component

J_A (used for the parallel/orthogonal split) is frozen and defined ONLY
from theta_A -- never refit on or informed by M_AB.

Reports the full dose-response curve per condition per seed, plus seed-level
summary statistics with 95% confidence intervals.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (jacobian_zor_red_vs_blue, interpolated_patch_margin,
                        decompose_parallel_orthogonal, matched_random_activation)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def train_theta_A(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    return model_A, filler_mapping


def get_hidden(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(obj_id, ctx_id).squeeze(0)


def run_decomposition_experiment(seed: int, foreign_seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_A, filler_mapping = train_theta_A(seed)
    theta_A_state = copy.deepcopy(model_A.state_dict())

    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")

    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=20)

    model_AB_T = new_model()
    model_AB_T.load_state_dict(log_AB[-1]["state_dict"])

    model_A_foreign, _ = train_theta_A(foreign_seed)

    h_A = get_hidden(model_A, SPECIAL_OBJECT)
    h_A_foreign = get_hidden(model_A_foreign, SPECIAL_OBJECT)
    h_AB = get_hidden(model_AB_T, SPECIAL_OBJECT)

    h_A_parallel, h_A_perp = decompose_parallel_orthogonal(h_A, J_A)
    h_AB_parallel, h_AB_perp = decompose_parallel_orthogonal(h_AB, J_A)

    h_source_parallel_only = h_A_parallel + h_AB_perp
    h_source_perp_only = h_A_perp + h_AB_parallel

    h_random = matched_random_activation(h_A, seed=seed + 777)

    conditions = {
        "own_full": h_A,
        "foreign_full": h_A_foreign,
        "random_matched": h_random,
        "own_parallel_only": h_source_parallel_only,
        "own_orthogonal_only": h_source_perp_only,
    }

    results = {}
    for cond_name, h_source in conditions.items():
        curve = interpolated_patch_margin(model_AB_T, h_source, SPECIAL_OBJECT)
        results[cond_name] = curve
        m_at_1 = curve[-1]["margin"]
        m_at_0 = curve[0]["margin"]
        print(f"[{run_name}] {cond_name:20s}: margin(lambda=0)={m_at_0:7.3f} "
              f"margin(lambda=1)={m_at_1:7.3f}  restores_red={m_at_1 < 0}")

    result = {
        "run_name": run_name, "seed": seed, "foreign_seed": foreign_seed,
        "h_A_parallel_norm": h_A_parallel.norm().item(), "h_A_perp_norm": h_A_perp.norm().item(),
        "h_A_norm": h_A.norm().item(),
        "curves": results,
    }
    with open(f"/home/claude/iclr/results/decomposition_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    all_results = []
    for i, s in enumerate(SEEDS):
        foreign = SEEDS[(i + 1) % len(SEEDS)]
        print("=" * 60)
        all_results.append(run_decomposition_experiment(s, foreign, run_name=f"seed{s}"))
        print()

    print("\n=== SEED-LEVEL SUMMARY (margin at lambda=1, blue-positive; negative = red restored) ===")
    condition_names = ["own_full", "foreign_full", "random_matched", "own_parallel_only", "own_orthogonal_only"]
    summary = {}
    for cond in condition_names:
        vals = np.array([r["curves"][cond][-1]["margin"] for r in all_results])
        mean = vals.mean()
        std = vals.std(ddof=1)
        se = std / np.sqrt(len(vals))
        ci95 = 1.96 * se
        n_restored = int((vals < 0).sum())
        print(f"{cond:20s}: mean={mean:7.3f}  95%CI=[{mean-ci95:.3f}, {mean+ci95:.3f}]  "
              f"n_restored={n_restored}/{len(vals)}")
        summary[cond] = {"mean": float(mean), "std": float(std), "ci95_lower": float(mean - ci95),
                          "ci95_upper": float(mean + ci95), "n_restored": n_restored, "n_total": len(vals),
                          "values": vals.tolist()}

    with open("/home/claude/iclr/results/decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
