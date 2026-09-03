"""
Readout co-adaptation test: does J_A remain aligned with the model's OWN
CURRENT downstream readout direction over the course of B-training, or does
the readout rotate away from J_A while J_A's ablation effect (C_A) persists
via some other route?

Tracks readout_alignment(t) = cos(J_A, W_red(t) - W_blue(t)) alongside
C_A(t) across the SAME checkpoints, for both C1 (ordinary training) and C3
(weight decay), multiple seeds.
"""
import json
import copy
import numpy as np
import torch
import torch.nn.functional as Fnn

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase
from src.probe import jacobian_zor_red_vs_blue, J_A_readout_alignment, ablate_along_J


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def run_readout_alignment_trajectory(seed: int, config: dict, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dim = config.get("hidden_dim", 32)
    bottleneck_dim = config.get("bottleneck_dim", None)
    phase_B_steps = config.get("phase_B_steps", 3000)
    phase_B_lr = config.get("phase_B_lr", 0.005)
    weight_decay = config.get("weight_decay", 0.0)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=phase_B_lr, weight_decay=weight_decay)

    rng = np.random.RandomState(seed + 1)
    eval_every = max(1, phase_B_steps // 300)

    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    trajectory = []
    for step in range(phase_B_steps):
        objs, ctxs, labels = ds_B.sample_batch(32, rng)
        logits = model_AB(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % eval_every == 0 or step == phase_B_steps - 1:
            readout_align = J_A_readout_alignment(model_AB, J_A)

            with torch.no_grad():
                logits_normal = model_AB(zor_id, ctx_red)
                m_normal = (logits_normal[0, COLOR2ID["blue"]] - logits_normal[0, COLOR2ID["red"]]).item()
            logits_ablated = ablate_along_J(model_AB, zor_id, ctx_red, J_A, alpha=1.0)
            m_ablated = (logits_ablated[0, COLOR2ID["blue"]] - logits_ablated[0, COLOR2ID["red"]]).item()
            C_A_t = m_ablated - m_normal

            trajectory.append({"step": step, "readout_alignment": readout_align, "C_A": C_A_t, "m": m_normal})

    with open(f"/home/claude/iclr/results/readout_alignment_{run_name}.json", "w") as f:
        json.dump({"run_name": run_name, "seed": seed, "config": config, "trajectory": trajectory},
                   f, indent=2, default=str)

    final = trajectory[-1]
    initial = trajectory[0]
    print(f"[{run_name}] readout_alignment: t=0 -> {initial['readout_alignment']:.4f}, "
          f"t=final -> {final['readout_alignment']:.4f} | C_A: t=0 -> {initial['C_A']:.3f}, "
          f"t=final -> {final['C_A']:.3f}")
    return trajectory


if __name__ == "__main__":
    C1_config = {"hidden_dim": 32, "phase_B_steps": 3000, "phase_B_lr": 0.005, "weight_decay": 0.0}
    C3_config = {"hidden_dim": 32, "bottleneck_dim": 2, "phase_B_steps": 8000,
                 "phase_B_lr": 0.005, "weight_decay": 0.1}

    SEEDS = [1234, 1235, 1236, 1237, 1238]

    print("=" * 60)
    print("C1 (ordinary training)")
    print("=" * 60)
    for s in SEEDS:
        run_readout_alignment_trajectory(s, C1_config, run_name=f"C1_seed{s}")

    print("\n" + "=" * 60)
    print("C3 (weight decay)")
    print("=" * 60)
    for s in SEEDS:
        run_readout_alignment_trajectory(s, C3_config, run_name=f"C3_seed{s}")
