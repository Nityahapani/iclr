"""
Nonlinear readout replication. Same task as the color task used throughout,
but the final readout is a small 2-layer MLP with ReLU instead of a single
linear layer, addressing the concern that the clean additive/tautological
geometry seen throughout this project is a consequence of the linear
readout specifically.

Reproduces the CORE dissociation only: t_flip, C_A(t) trajectory (via
single-ablation, projected through the now-nonlinear readout).
"""
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.train import train_phase


class NonlinearReadoutClassifier(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, ctx_embed_dim=8, hidden_dim=32, readout_hidden=24):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ctx_embed = nn.Embedding(CONTEXT_VOCAB_SIZE, ctx_embed_dim)
        self.fc1 = nn.Linear(embed_dim + ctx_embed_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2a = nn.Linear(hidden_dim, readout_hidden)
        self.relu = nn.ReLU()
        self.fc2b = nn.Linear(readout_hidden, NUM_CLASSES)

    def hidden(self, obj_ids, ctx_ids):
        e = self.embed(obj_ids)
        c = self.ctx_embed(ctx_ids)
        return self.act(self.fc1(torch.cat([e, c], dim=-1)))

    def fc2(self, h):
        return self.fc2b(self.relu(self.fc2a(h)))

    def forward(self, obj_ids, ctx_ids):
        return self.fc2(self.hidden(obj_ids, ctx_ids))


def jacobian_zor_red_vs_blue_nonlinear(model, ctx_name="CTX_RED"):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    h = model.hidden(zor_id, ctx_id)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    margin = logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]
    grad = torch.autograd.grad(margin, h)[0].squeeze(0)
    return grad


def ablate_and_margin(model, J_A, ctx_name="CTX_RED", alpha=1.0):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        h = model.hidden(zor_id, ctx_id).squeeze(0)
        logits_normal = model.fc2(h.unsqueeze(0))
        m_normal = (logits_normal[0, COLOR2ID["blue"]] - logits_normal[0, COLOR2ID["red"]]).item()
        coeff = (J_A @ h) / ((J_A @ J_A) + 1e-9)
        h_ablated = h - alpha * coeff * J_A
        logits_ablated = model.fc2(h_ablated.unsqueeze(0))
        m_ablated = (logits_ablated[0, COLOR2ID["blue"]] - logits_ablated[0, COLOR2ID["red"]]).item()
    return m_normal, m_ablated


def run_nonlinear_readout_trajectory(seed: int, run_name: str, phase_B_steps: int = 3000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = NonlinearReadoutClassifier()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    J_A = jacobian_zor_red_vs_blue_nonlinear(model_A, ctx_name="CTX_RED")
    m_A_theta_A, m_A_ablated_theta_A = ablate_and_margin(model_A, J_A)
    delta_A_reference = -(m_A_ablated_theta_A - m_A_theta_A)
    print(f"[{run_name}] theta_A: m={m_A_theta_A:.3f}, Delta_A(theta_A)={delta_A_reference:.3f}")
    if delta_A_reference <= 0.5:
        print(f"[{run_name}] WARNING: weak reference effect at theta_A, results may be unreliable")

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = NonlinearReadoutClassifier()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=0.005)

    rng = np.random.RandomState(seed + 1)
    eval_every = max(1, phase_B_steps // 300)

    t_flip = None
    trajectory = []
    for step in range(phase_B_steps):
        objs, ctxs, labels = ds_B.sample_batch(32, rng)
        logits = model_AB(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % eval_every == 0 or step == phase_B_steps - 1:
            m_normal, m_ablated = ablate_and_margin(model_AB, J_A)
            if t_flip is None and m_normal > 0:
                t_flip = step
            C_A_t = m_ablated - m_normal
            trajectory.append({"step": step, "m": m_normal, "C_A": C_A_t})

    final = trajectory[-1]
    print(f"[{run_name}] t_flip={t_flip}, final m={final['m']:.3f}, final C_A={final['C_A']:.3f}")

    result = {"run_name": run_name, "seed": seed, "delta_A_reference": delta_A_reference,
              "t_flip": t_flip, "trajectory": trajectory}
    with open(f"/home/claude/iclr/results/nonlinear_readout_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_nonlinear_readout_trajectory(s, run_name=f"seed{s}"))

    print("\n=== SUMMARY ===")
    t_flips = [r["t_flip"] for r in results]
    final_C_A = np.array([r["trajectory"][-1]["C_A"] for r in results])
    final_C_A_frac = np.array([r["trajectory"][-1]["C_A"] / r["delta_A_reference"] for r in results])
    print(f"t_flip per seed: {t_flips}")
    print(f"final C_A: mean={final_C_A.mean():.3f} std={final_C_A.std(ddof=1):.3f}")
    print(f"final C_A as fraction of theta_A reference: mean={final_C_A_frac.mean():.3f}")

    with open("/home/claude/iclr/results/nonlinear_readout_summary.json", "w") as f:
        json.dump({"t_flips": t_flips, "final_C_A_mean": float(final_C_A.mean()),
                    "final_C_A_std": float(final_C_A.std(ddof=1)),
                    "final_C_A_frac_mean": float(final_C_A_frac.mean())}, f, indent=2)
