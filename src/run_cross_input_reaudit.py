"""
Re-audit of the cross-input transfer result (zor <-> vex), per reviewer's
sharpest scrutiny. The concern: zor and vex compute the IDENTICAL function
during phase A (both use CONTROL_CTX_MAPPING), so their theta_A activations
are already highly similar (cos=0.91 at theta_A, verified below) BEFORE any
intervention. The earlier "J_A(vex)->zor restores red" result could
therefore be explained trivially by h_vex being generically close to h_zor,
not by J_A capturing a transferable abstraction.

Adds: raw activation similarity measurement; a genuinely DIFFERENT-
COMPUTATION control (fenn/SHAM_OBJECT, also context-dependent but computes
a DIFFERENT function during phase A); a matched random control on the same
target; comparison of all three.
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
from src.probe import cosine_alignment, matched_random_activation


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def get_hidden(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(obj_id, ctx_id).squeeze(0)


def margin_via_patch(model_T, h_source: torch.Tensor, target_object: str, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[target_object]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model_T.fc2(h_source.unsqueeze(0))
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def run_reaudit(seed: int, run_name: str):
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

    cos_zor_vex = cosine_alignment(h_zor, h_vex)
    cos_zor_fenn = cosine_alignment(h_zor, h_fenn)
    print(f"[{run_name}] theta_A raw activation similarity: cos(zor,vex)={cos_zor_vex:.4f}, "
          f"cos(zor,fenn)={cos_zor_fenn:.4f}")

    m_vex_to_zor = margin_via_patch(model_AB_T, h_vex, SPECIAL_OBJECT)
    m_fenn_to_zor = margin_via_patch(model_AB_T, h_fenn, SPECIAL_OBJECT)
    m_zor_to_zor = margin_via_patch(model_AB_T, h_zor, SPECIAL_OBJECT)

    h_random = matched_random_activation(h_zor, seed=seed + 777)
    m_random_to_zor = margin_via_patch(model_AB_T, h_random, SPECIAL_OBJECT)

    print(f"[{run_name}] patch->zor margins: own={m_zor_to_zor:.3f}, vex={m_vex_to_zor:.3f}, "
          f"fenn={m_fenn_to_zor:.3f}, random={m_random_to_zor:.3f}")

    result = {
        "run_name": run_name, "seed": seed,
        "cos_zor_vex_at_thetaA": cos_zor_vex, "cos_zor_fenn_at_thetaA": cos_zor_fenn,
        "m_own": m_zor_to_zor, "m_vex": m_vex_to_zor, "m_fenn": m_fenn_to_zor, "m_random": m_random_to_zor,
        "h_zor_norm": h_zor.norm().item(), "h_vex_norm": h_vex.norm().item(), "h_fenn_norm": h_fenn.norm().item(),
    }
    with open(f"/home/claude/iclr/results/cross_input_reaudit_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_reaudit(s, run_name=f"seed{s}"))
        print()

    print("\n=== SUMMARY ===")
    for key in ["m_own", "m_vex", "m_fenn", "m_random"]:
        vals = np.array([r[key] for r in results])
        n_restore = int((vals < 0).sum())
        print(f"{key:10s}: mean={vals.mean():7.3f} std={vals.std(ddof=1):.3f} n_restore={n_restore}/{len(vals)}")

    cos_vex = np.array([r["cos_zor_vex_at_thetaA"] for r in results])
    cos_fenn = np.array([r["cos_zor_fenn_at_thetaA"] for r in results])
    print(f"\ncos(zor,vex) at theta_A: mean={cos_vex.mean():.4f}")
    print(f"cos(zor,fenn) at theta_A: mean={cos_fenn.mean():.4f}")

    from scipy.stats import wilcoxon
    m_vex = np.array([r["m_vex"] for r in results])
    m_fenn = np.array([r["m_fenn"] for r in results])
    m_random = np.array([r["m_random"] for r in results])
    w1, p1 = wilcoxon(m_vex, m_fenn)
    w2, p2 = wilcoxon(m_vex, m_random)
    w3, p3 = wilcoxon(m_fenn, m_random)
    print(f"\nvex vs fenn (paired): W={w1:.1f} p={p1:.4f}")
    print(f"vex vs random (paired): W={w2:.1f} p={p2:.4f}")
    print(f"fenn vs random (paired): W={w3:.1f} p={p3:.4f}")

    with open("/home/claude/iclr/results/cross_input_reaudit_summary.json", "w") as f:
        json.dump({
            "means": {k: float(np.array([r[k] for r in results]).mean()) for k in ["m_own", "m_vex", "m_fenn", "m_random"]},
            "cos_zor_vex_mean": float(cos_vex.mean()), "cos_zor_fenn_mean": float(cos_fenn.mean()),
            "vex_vs_fenn_p": float(p1), "vex_vs_random_p": float(p2), "fenn_vs_random_p": float(p3),
        }, f, indent=2)
