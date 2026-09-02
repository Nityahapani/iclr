"""
Activation patching experiment (path-specific/mechanism-specific test,
stronger than J_A-ablation). Patches theta_A's OWN hidden activation for zor
directly into the final model's readout, testing whether the current output
head still interprets A's actual computed representation as "red".

Three conditions per seed:
  1. M_AB's readout, patched with ITS OWN theta_A's activation (same
     lineage).
  2. M_B's readout (never had a real theta_A), patched with an
     INDEPENDENTLY-TRAINED theta_A activation from a DIFFERENT seed's phase-A
     model (foreign, unrelated activation pattern).
  3. M_AB's readout, patched with the SAME foreign theta_A (not its own
     lineage) -- isolates whether ANY theta_A activation restores red on
     M_AB, or specifically its OWN lineage's.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import activation_patch_from_theta_A


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


def run_patching_experiment(seed: int, foreign_seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_A, filler_mapping = train_theta_A(seed)
    theta_A_state = copy.deepcopy(model_A.state_dict())

    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=20)

    model_B = new_model()
    torch.manual_seed(seed + 2)
    log_B = train_phase(model_B, ds_B_only, steps=3000, batch_size=32, lr=0.005, seed=seed + 3, eval_every=20)

    matched_entry, matched_kl = find_matched_checkpoint(log_B, log_AB)
    model_AB_T = new_model()
    model_AB_T.load_state_dict(matched_entry["state_dict"])
    model_B_T = new_model()
    model_B_T.load_state_dict(log_B[-1]["state_dict"])

    print(f"[{run_name}] matched KL={matched_kl:.6f}")

    result_own = activation_patch_from_theta_A(model_AB_T, model_A, SPECIAL_OBJECT, ctx_name="CTX_RED")
    print(f"[{run_name}] M_AB patched w/ OWN theta_A: m_normal={result_own['m_T_normal']:.3f}, "
          f"m_patched={result_own['m_patched']:.3f} (patch_restores_red={result_own['patch_restores_red']}), "
          f"h_A vs h_T cosine={result_own['h_A_vs_h_T_cosine']:.3f}")

    model_A_foreign, _ = train_theta_A(foreign_seed)
    result_foreign = activation_patch_from_theta_A(model_B_T, model_A_foreign, SPECIAL_OBJECT, ctx_name="CTX_RED")
    print(f"[{run_name}] M_B patched w/ FOREIGN theta_A (seed {foreign_seed}): "
          f"m_normal={result_foreign['m_T_normal']:.3f}, m_patched={result_foreign['m_patched']:.3f} "
          f"(patch_restores_red={result_foreign['patch_restores_red']}), "
          f"h_A vs h_T cosine={result_foreign['h_A_vs_h_T_cosine']:.3f}")

    result_AB_foreign = activation_patch_from_theta_A(model_AB_T, model_A_foreign, SPECIAL_OBJECT, ctx_name="CTX_RED")
    print(f"[{run_name}] M_AB patched w/ FOREIGN theta_A: m_patched={result_AB_foreign['m_patched']:.3f} "
          f"(patch_restores_red={result_AB_foreign['patch_restores_red']})")

    result = {
        "run_name": run_name, "seed": seed, "foreign_seed": foreign_seed,
        "matched_kl": matched_kl,
        "M_AB_own_theta_A": result_own,
        "M_B_foreign_theta_A": result_foreign,
        "M_AB_foreign_theta_A": result_AB_foreign,
    }
    with open(f"/home/claude/iclr/results/patching_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for i, s in enumerate(SEEDS):
        foreign = SEEDS[(i + 1) % len(SEEDS)]
        print("=" * 60)
        results.append(run_patching_experiment(s, foreign, run_name=f"seed{s}"))
        print()

    n_own_restores = sum(r["M_AB_own_theta_A"]["patch_restores_red"] for r in results)
    n_foreign_B_restores = sum(r["M_B_foreign_theta_A"]["patch_restores_red"] for r in results)
    n_foreign_AB_restores = sum(r["M_AB_foreign_theta_A"]["patch_restores_red"] for r in results)

    print(f"\n=== SUMMARY ===")
    print(f"M_AB patched w/ OWN theta_A restores red: {n_own_restores}/{len(results)}")
    print(f"M_B patched w/ FOREIGN theta_A restores red: {n_foreign_B_restores}/{len(results)}")
    print(f"M_AB patched w/ FOREIGN theta_A restores red: {n_foreign_AB_restores}/{len(results)}")

    with open("/home/claude/iclr/results/patching_summary.json", "w") as f:
        json.dump({
            "n_own_restores": n_own_restores,
            "n_foreign_B_restores": n_foreign_B_restores,
            "n_foreign_AB_restores": n_foreign_AB_restores,
            "n_total": len(results),
        }, f, indent=2)
