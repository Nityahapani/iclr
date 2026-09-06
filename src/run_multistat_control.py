"""
Multi-statistic-matched cross-input control (strengthened version of the
similarity decomposition in run_synthetic_control.py). Matches the
synthetic control on cosine similarity, norm, margin (decision-boundary
distance proxy), AND local Jacobian similarity to h_vex simultaneously
(searched, not closed-form), rather than cosine similarity alone.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, CONTROL_OBJECT, SHAM_OBJECT,
                       VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import multi_statistic_matched_search


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def get_hidden(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(obj_id, ctx_id).squeeze(0)


def margin_via_patch(model_T, h_source, ctx_name="CTX_RED"):
    with torch.no_grad():
        logits = model_T.fc2(h_source.unsqueeze(0))
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def run_multi_stat_control(seed: int, run_name: str):
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

    h_zor = get_hidden(model_A, SPECIAL_OBJECT)
    h_vex = get_hidden(model_A, CONTROL_OBJECT)
    h_fenn = get_hidden(model_A, SHAM_OBJECT)

    search_result = multi_statistic_matched_search(model_A, h_zor, h_vex, h_fenn, J_A=None, seed=seed + 42)
    if search_result is None:
        print(f"[{run_name}] SKIPPED: degenerate orthogonal component")
        return None
    h_synthetic_v2 = search_result["h_candidate"]

    m_own = margin_via_patch(model_AB_T, h_zor)
    m_vex = margin_via_patch(model_AB_T, h_vex)
    m_fenn = margin_via_patch(model_AB_T, h_fenn)
    m_synth_v2 = margin_via_patch(model_AB_T, h_synthetic_v2)

    print(f"[{run_name}] match quality: cos_target={search_result['cos_target']:.4f}, "
          f"norm_target={search_result['norm_target']:.4f}, margin_target={search_result['margin_target']:.4f}, "
          f"best_score={search_result['best_score']:.4f}")
    print(f"[{run_name}] m_own={m_own:.3f}, m_vex={m_vex:.3f}, m_synthetic_v2(multi-stat matched)={m_synth_v2:.3f}, "
          f"m_fenn={m_fenn:.3f}")

    residual_v2 = m_vex - m_synth_v2
    print(f"[{run_name}] identity-specific residual (v2, multi-stat matched) = {residual_v2:.4f}")

    result = {"run_name": run_name, "seed": seed, "m_own": m_own, "m_vex": m_vex,
              "m_synthetic_v2": m_synth_v2, "m_fenn": m_fenn, "residual_v2": residual_v2,
              "match_quality": {k: v for k, v in search_result.items() if k != "h_candidate"}}
    with open(f"/home/claude/iclr/results/multistat_control_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        r = run_multi_stat_control(s, run_name=f"seed{s}")
        if r is not None:
            results.append(r)

    print("\n=== SUMMARY ===")
    m_vex = np.array([r["m_vex"] for r in results])
    m_synth_v2 = np.array([r["m_synthetic_v2"] for r in results])
    residuals = np.array([r["residual_v2"] for r in results])
    print(f"m_vex: mean={m_vex.mean():.3f} std={m_vex.std(ddof=1):.3f}")
    print(f"m_synthetic_v2 (multi-stat matched): mean={m_synth_v2.mean():.3f} std={m_synth_v2.std(ddof=1):.3f}")
    print(f"residual (m_vex - m_synthetic_v2): mean={residuals.mean():.3f} std={residuals.std(ddof=1):.3f}")

    from scipy.stats import wilcoxon
    w, p = wilcoxon(m_vex, m_synth_v2)
    print(f"\nWilcoxon (vex vs multi-stat-matched synthetic): W={w:.1f} p={p:.4f}")
    n_residual_same_sign = int((residuals < 0).sum())
    print(f"residual negative (vex stronger) in {n_residual_same_sign}/{len(residuals)} seeds")

    with open("/home/claude/iclr/results/multistat_control_summary.json", "w") as f:
        json.dump({
            "m_vex_mean": float(m_vex.mean()), "m_synthetic_v2_mean": float(m_synth_v2.mean()),
            "residual_mean": float(residuals.mean()), "residual_std": float(residuals.std(ddof=1)),
            "wilcoxon_p": float(p), "n_residual_negative": n_residual_same_sign, "n_total": len(residuals),
        }, f, indent=2)
