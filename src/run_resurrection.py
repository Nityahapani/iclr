"""
Causal resurrection by component replacement. Not an inference-time
activation patch -- a genuine PARAMETER-SPACE surgical intervention on
zor's own embedding row.

Chain: (1) baseline P_A check; (2) surgically DESTROY the J_A-relevant
component of zor's embedding in-place, verify P_A~0 and B behavior
unchanged; (3) TRANSPLANT a candidate component (own theta_A, foreign
theta_A, random matched) back in; (4) measure A recovery. Repeated for both
M_AB (real A lineage) and M_B (no real A lineage) for the double
dissociation.
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import (jacobian_zor_red_vs_blue, ablate_along_J,
                        surgically_destroy_component, surgically_transplant_component)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def margin(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(obj_id, ctx_id)
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def measure_P_A(model, J_A, object_name="zor"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    m_normal = margin(model, object_name)
    logits_ablated = ablate_along_J(model, obj_id, ctx_id, J_A, alpha=1.0)
    m_ablated = (logits_ablated[0, COLOR2ID["blue"]] - logits_ablated[0, COLOR2ID["red"]]).item()
    return m_ablated - m_normal


def train_theta_A(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    return model_A, filler_mapping


def run_resurrection_chain(seed: int, foreign_seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_A, filler_mapping = train_theta_A(seed)
    theta_A_state = copy.deepcopy(model_A.state_dict())
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    own_A_embed = model_A.embed.weight[OBJ2ID[SPECIAL_OBJECT]].clone()

    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=20)
    model_AB_T = new_model()
    model_AB_T.load_state_dict(log_AB[-1]["state_dict"])

    torch.manual_seed(seed + 2)
    model_B = new_model()
    log_B = train_phase(model_B, ds_B_only, steps=3000, batch_size=32, lr=0.005, seed=seed + 3, eval_every=20)
    model_B_T = new_model()
    model_B_T.load_state_dict(log_B[-1]["state_dict"])

    model_A_foreign, _ = train_theta_A(foreign_seed)
    foreign_A_embed = model_A_foreign.embed.weight[OBJ2ID[SPECIAL_OBJECT]].clone()

    embed_dim = model_AB_T.embed.embedding_dim
    g = torch.Generator().manual_seed(seed + 888)
    random_embed = torch.randn(embed_dim, generator=g)
    random_embed = random_embed / random_embed.norm() * own_A_embed.norm()

    results = {}

    for model_label, base_model in [("M_AB", model_AB_T), ("M_B", model_B_T)]:
        m_before = margin(base_model, SPECIAL_OBJECT)
        P_A_before = measure_P_A(base_model, J_A)

        model_destroyed = new_model()
        model_destroyed.load_state_dict(copy.deepcopy(base_model.state_dict()))
        model_destroyed, removed_norm = surgically_destroy_component(model_destroyed, SPECIAL_OBJECT, J_A, alpha=1.0)

        m_after_destroy = margin(model_destroyed, SPECIAL_OBJECT)
        P_A_after_destroy = measure_P_A(model_destroyed, J_A)

        print(f"[{run_name}] {model_label}: before m={m_before:.3f} P_A={P_A_before:.3f} | "
              f"after destroy (removed_norm={removed_norm:.3f}) m={m_after_destroy:.3f} P_A={P_A_after_destroy:.3f}")

        transplant_results = {}
        for cond_name, source_embed in [("own_A", own_A_embed), ("foreign_A", foreign_A_embed),
                                          ("random", random_embed)]:
            model_transplanted = new_model()
            model_transplanted.load_state_dict(copy.deepcopy(model_destroyed.state_dict()))
            model_transplanted = surgically_transplant_component(model_transplanted, SPECIAL_OBJECT,
                                                                    source_embed, J_A, alpha=1.0)
            m_after_transplant = margin(model_transplanted, SPECIAL_OBJECT)
            transplant_results[cond_name] = m_after_transplant
            print(f"    transplant {cond_name:10s}: m={m_after_transplant:.3f} "
                  f"(A recovered: {m_after_transplant < 0})")

        results[model_label] = {
            "m_before": m_before, "P_A_before": P_A_before,
            "m_after_destroy": m_after_destroy, "P_A_after_destroy": P_A_after_destroy,
            "removed_norm": removed_norm,
            "transplant": transplant_results,
        }

    result = {"run_name": run_name, "seed": seed, "foreign_seed": foreign_seed, "results": results}
    with open(f"/home/claude/iclr/results/resurrection_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    all_results = []
    for i, s in enumerate(SEEDS):
        foreign = SEEDS[(i + 1) % len(SEEDS)]
        print("=" * 60)
        all_results.append(run_resurrection_chain(s, foreign, run_name=f"seed{s}"))
        print()

    print("\n=== SUMMARY: fraction of seeds where transplant restores A (margin<0) ===")
    for model_label in ["M_AB", "M_B"]:
        for cond_name in ["own_A", "foreign_A", "random"]:
            vals = [r["results"][model_label]["transplant"][cond_name] for r in all_results]
            n_restore = sum(1 for v in vals if v < 0)
            mean_v = np.mean(vals)
            print(f"{model_label} + {cond_name:10s}: {n_restore}/{len(vals)} restore A, mean_margin={mean_v:.3f}")

    with open("/home/claude/iclr/results/resurrection_summary.json", "w") as f:
        summary = {}
        for model_label in ["M_AB", "M_B"]:
            summary[model_label] = {}
            for cond_name in ["own_A", "foreign_A", "random"]:
                vals = [r["results"][model_label]["transplant"][cond_name] for r in all_results]
                summary[model_label][cond_name] = {"values": vals, "mean": float(np.mean(vals)),
                                                     "n_restore": sum(1 for v in vals if v < 0)}
        json.dump(summary, f, indent=2)
