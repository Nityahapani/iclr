"""
Replication of the similarity-matched synthetic-activation control across 7
seeds. Tests whether the zor<->vex transfer effect is explained by raw
cosine similarity or reflects something specific to vex's identity beyond
similarity alone.
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
from src.probe import cosine_alignment, similarity_matched_synthetic_activation


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def get_hidden(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        return model.hidden(obj_id, ctx_id).squeeze(0)


def margin_via_patch(model_T, h_source, target_object, ctx_name="CTX_RED"):
    with torch.no_grad():
        logits = model_T.fc2(h_source.unsqueeze(0))
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def run_synthetic_control(seed: int, run_name: str):
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

    cos_zv = cosine_alignment(h_zor, h_vex)
    h_synthetic = similarity_matched_synthetic_activation(h_zor, h_vex, h_fenn)
    cos_check = cosine_alignment(h_zor, h_synthetic) if h_synthetic is not None else None

    m_own = margin_via_patch(model_AB_T, h_zor, SPECIAL_OBJECT)
    m_vex = margin_via_patch(model_AB_T, h_vex, SPECIAL_OBJECT)
    m_fenn = margin_via_patch(model_AB_T, h_fenn, SPECIAL_OBJECT)
    m_synth = margin_via_patch(model_AB_T, h_synthetic, SPECIAL_OBJECT) if h_synthetic is not None else None

    print(f"[{run_name}] cos(zor,vex)={cos_zv:.4f} (synthetic matched to {cos_check}); "
          f"m_own={m_own:.3f} m_vex={m_vex:.3f} m_synthetic={m_synth} m_fenn={m_fenn:.3f}")

    result = {"run_name": run_name, "seed": seed, "cos_zor_vex": cos_zv, "cos_check": cos_check,
              "m_own": m_own, "m_vex": m_vex, "m_synthetic": m_synth, "m_fenn": m_fenn}
    with open(f"/home/claude/iclr/results/synthetic_control_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        results.append(run_synthetic_control(s, run_name=f"seed{s}"))

    print("\n=== SUMMARY ===")
    m_vex = np.array([r["m_vex"] for r in results])
    m_synth = np.array([r["m_synthetic"] for r in results if r["m_synthetic"] is not None])
    m_fenn = np.array([r["m_fenn"] for r in results])
    print(f"m_vex:       mean={m_vex.mean():.3f} std={m_vex.std(ddof=1):.3f}")
    print(f"m_synthetic: mean={m_synth.mean():.3f} std={m_synth.std(ddof=1):.3f}")
    print(f"m_fenn:      mean={m_fenn.mean():.3f} std={m_fenn.std(ddof=1):.3f}")

    from scipy.stats import wilcoxon
    w, p = wilcoxon(m_vex, m_synth[:len(m_vex)])
    print(f"\nvex vs similarity-matched synthetic (paired): W={w:.1f} p={p:.4f}")

    with open("/home/claude/iclr/results/synthetic_control_summary.json", "w") as f:
        json.dump({
            "m_vex_mean": float(m_vex.mean()), "m_synthetic_mean": float(m_synth.mean()),
            "m_fenn_mean": float(m_fenn.mean()), "wilcoxon_p": float(p),
        }, f, indent=2)
