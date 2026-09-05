"""
Held-out-input generalization test. Construct J_A using only the TRAIN
pool of context-dependent objects (never seeing HELDOUT_POOL instances
during construction), freeze it, train A->B, then measure B_t, P_A(t), and
R_A(t) on the held-out pool.
"""
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from src.task_multiinstance import (make_filler_mapping, MultiInstanceDataset, OBJ2ID, CTX2ID,
                                      COLOR2ID, TRAIN_POOL, HELDOUT_POOL, VOCAB_SIZE, NUM_CLASSES,
                                      CONTEXT_VOCAB_SIZE)


class MIModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, ctx_embed_dim=8, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ctx_embed = nn.Embedding(CONTEXT_VOCAB_SIZE, ctx_embed_dim)
        self.fc1 = nn.Linear(embed_dim + ctx_embed_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, NUM_CLASSES)

    def hidden(self, obj_ids, ctx_ids):
        e = self.embed(obj_ids)
        c = self.ctx_embed(ctx_ids)
        return self.act(self.fc1(torch.cat([e, c], dim=-1)))

    def forward(self, obj_ids, ctx_ids):
        return self.fc2(self.hidden(obj_ids, ctx_ids))


def train_generic(model, dataset, steps, batch_size, lr, seed, eval_every=None):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eval_every = eval_every or steps
    log = []
    eval_set = dataset.full_eval_set()
    eo, ec, el = eval_set
    for step in range(steps):
        o, c, y = dataset.sample_batch(batch_size, rng)
        logits = model(o, c)
        loss = Fnn.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (model(eo, ec).argmax(-1) == el).float().mean().item()
            log.append({"step": step, "eval_acc": acc, "state_dict": copy.deepcopy(model.state_dict())})
    return log


def margin_blue_vs_red(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        logits = model(obj_id, ctx_id)
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def per_instance_jacobian(model, object_name, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    h = model.hidden(obj_id, ctx_id)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    margin = logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]
    grad = torch.autograd.grad(margin, h)[0].squeeze(0)
    return grad


def build_averaged_J_A(model, instance_list, ctx_name="CTX_RED"):
    grads = [per_instance_jacobian(model, obj, ctx_name) for obj in instance_list]
    return torch.stack(grads).mean(dim=0)


def ablate_and_margin(model, object_name, J_A, ctx_name="CTX_RED", alpha=1.0):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        coeff = (J_A @ h) / ((J_A @ J_A) + 1e-9)
        h_ablated = h - alpha * coeff * J_A
        logits = model.fc2(h_ablated.unsqueeze(0))
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def patch_own_theta_A_and_margin(model_T, model_A, object_name, J_A, ctx_name="CTX_RED"):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    with torch.no_grad():
        h_A = model_A.hidden(obj_id, ctx_id).squeeze(0)
        h_T = model_T.hidden(obj_id, ctx_id).squeeze(0)
        J_unit = J_A / (J_A.norm() + 1e-9)
        coeff_A = h_A @ J_unit
        coeff_T = h_T @ J_unit
        h_patched = h_T - coeff_T * J_unit + coeff_A * J_unit
        logits = model_T.fc2(h_patched.unsqueeze(0))
        return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()


def run_heldout_generalization(seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = MultiInstanceDataset(filler_mapping, phase="A")
    ds_B = MultiInstanceDataset(filler_mapping, phase="B")

    model_A = MIModel()
    log_A = train_generic(model_A, ds_A, steps=1500, batch_size=64, lr=0.01, seed=seed)
    acc_A = log_A[-1]["eval_acc"]

    J_A = build_averaged_J_A(model_A, TRAIN_POOL)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = MIModel()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_generic(model_AB, ds_B, steps=3000, batch_size=64, lr=0.005, seed=seed + 1)
    acc_AB = log_AB[-1]["eval_acc"]

    heldout_results = []
    for obj in HELDOUT_POOL:
        m_normal = margin_blue_vs_red(model_AB, obj)
        m_ablated = ablate_and_margin(model_AB, obj, J_A)
        P_A_obj = m_ablated - m_normal
        m_patched = patch_own_theta_A_and_margin(model_AB, model_A, obj, J_A)
        heldout_results.append({"object": obj, "B_t_margin": m_normal, "P_A": P_A_obj,
                                 "R_A_margin": m_patched, "B_t_is_blue": m_normal > 0,
                                 "R_A_restores_red": m_patched < 0})

    train_results = []
    for obj in TRAIN_POOL[:5]:
        m_normal = margin_blue_vs_red(model_AB, obj)
        m_ablated = ablate_and_margin(model_AB, obj, J_A)
        P_A_obj = m_ablated - m_normal
        m_patched = patch_own_theta_A_and_margin(model_AB, model_A, obj, J_A)
        train_results.append({"object": obj, "B_t_margin": m_normal, "P_A": P_A_obj,
                               "R_A_margin": m_patched, "B_t_is_blue": m_normal > 0,
                               "R_A_restores_red": m_patched < 0})

    n_heldout_blue = sum(r["B_t_is_blue"] for r in heldout_results)
    n_heldout_restore = sum(r["R_A_restores_red"] for r in heldout_results)
    mean_P_A_heldout = np.mean([r["P_A"] for r in heldout_results])

    print(f"[{run_name}] acc_A={acc_A:.3f} acc_AB={acc_AB:.3f}")
    print(f"[{run_name}] HELD-OUT (n={len(HELDOUT_POOL)}): B_t=blue in {n_heldout_blue}/{len(HELDOUT_POOL)}, "
          f"R_A restores red in {n_heldout_restore}/{len(HELDOUT_POOL)}, mean P_A={mean_P_A_heldout:.3f}")

    result = {"run_name": run_name, "seed": seed, "acc_A": acc_A, "acc_AB": acc_AB,
              "heldout_results": heldout_results, "train_reference_results": train_results}
    with open(f"/home/claude/iclr/results/heldout_generalization_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_heldout_generalization(s, run_name=f"seed{s}"))

    print("\n=== SUMMARY across seeds ===")
    all_frac_blue = []
    all_frac_restore = []
    for r in results:
        n = len(r["heldout_results"])
        n_blue = sum(x["B_t_is_blue"] for x in r["heldout_results"])
        n_restore = sum(x["R_A_restores_red"] for x in r["heldout_results"])
        all_frac_blue.append(n_blue / n)
        all_frac_restore.append(n_restore / n)
        print(f"seed {r['seed']}: frac B_t=blue={n_blue}/{n}, frac R_A restores={n_restore}/{n}")

    print(f"\nmean frac B_t=blue (held-out): {np.mean(all_frac_blue):.3f}")
    print(f"mean frac R_A restores red (held-out): {np.mean(all_frac_restore):.3f}")

    with open("/home/claude/iclr/results/heldout_generalization_summary.json", "w") as f:
        json.dump({"frac_blue_per_seed": all_frac_blue, "frac_restore_per_seed": all_frac_restore,
                    "mean_frac_blue": float(np.mean(all_frac_blue)), "mean_frac_restore": float(np.mean(all_frac_restore))},
                   f, indent=2)
