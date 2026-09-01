"""
Part 2: architecture generalization. Reuses the logic task (XOR->AND) with
THREE deliberately different architectures, testing only the binary question:
does the dissociation (t_flip small, final C_A far from zero) hold outside
the exact architecture where it was discovered?

Architectures:
  1. Tiny MLP  (hidden_dim=32 -- the architecture already used for Part 1)
  2. Small Transformer (single self-attention layer over [marker, bit1, bit2]
     as 3 tokens, readout on the marker token's post-attention state)
  3. Larger MLP (hidden_dim=128, 4x capacity) as the "larger model" arm --
     given the toy scale of this task, a wider MLP is a more meaningful
     "larger model" than a deeper transformer would be; noted as a scope
     limitation rather than a true larger-transformer test.

For each architecture: track only t_flip and final C_A (the lightweight core
test), across the SAME 7 seeds as Part 1, report the fraction of seeds
showing the dissociation.
"""
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from src.run_logic_task import (LogicDataset, LogicModel, make_filler_functions,
                                 xor_fn, and_fn, SPECIAL_MARKER, VOCAB_SIZE)


class LogicTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, hidden_dim=32, n_heads=2):
        super().__init__()
        self.marker_embed = nn.Embedding(vocab_size, embed_dim)
        self.bit_embed = nn.Embedding(2, embed_dim)
        self.pos_embed = nn.Embedding(3, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def hidden(self, markers, b1, b2):
        B = markers.shape[0]
        marker_tok = self.marker_embed(markers).unsqueeze(1)
        b1_tok = self.bit_embed(b1).unsqueeze(1)
        b2_tok = self.bit_embed(b2).unsqueeze(1)
        seq = torch.cat([marker_tok, b1_tok, b2_tok], dim=1)
        pos_ids = torch.arange(3).unsqueeze(0).expand(B, -1)
        seq = seq + self.pos_embed(pos_ids)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.ln(seq + attn_out)
        marker_repr = seq[:, 0, :]
        return self.act(self.fc1(marker_repr))

    def forward(self, markers, b1, b2):
        return self.fc2(self.hidden(markers, b1, b2))


def train_logic_generic(model, dataset, steps, batch_size, lr, seed):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eval_set = dataset.full_eval_set()
    for step in range(steps):
        m, b1, b2, y = dataset.sample_batch(batch_size, rng)
        logits = model(m, b1, b2)
        loss = Fnn.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        em, eb1, eb2, ey = eval_set
        eval_acc = (model(em, eb1, eb2).argmax(-1) == ey).float().mean().item()
    return eval_acc


def jacobian_special_margin_generic(model, b1_val, b2_val):
    m = torch.tensor([SPECIAL_MARKER], dtype=torch.long)
    b1 = torch.tensor([b1_val], dtype=torch.long)
    b2 = torch.tensor([b2_val], dtype=torch.long)
    h = model.hidden(m, b1, b2)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    margin = logits[0, 1] - logits[0, 0]
    grad = torch.autograd.grad(margin, h)[0].squeeze(0)
    return grad


def special_margin_generic(model, b1_val, b2_val):
    m = torch.tensor([SPECIAL_MARKER], dtype=torch.long)
    b1 = torch.tensor([b1_val], dtype=torch.long)
    b2 = torch.tensor([b2_val], dtype=torch.long)
    with torch.no_grad():
        logits = model(m, b1, b2)
        return (logits[0, 1] - logits[0, 0]).item()


def run_arch_experiment(model_ctor, seed: int, run_name: str, b1_val=1, b2_val=0,
                         phase_A_steps=800, phase_B_steps=3000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_fns = make_filler_functions(seed)
    ds_A = LogicDataset(filler_fns, phase="A")
    ds_B = LogicDataset(filler_fns, phase="B")

    model_A = model_ctor()
    acc_A = train_logic_generic(model_A, ds_A, steps=phase_A_steps, batch_size=32, lr=0.01, seed=seed)

    m_A_special = special_margin_generic(model_A, b1_val, b2_val)
    xor_label = xor_fn(b1_val, b2_val)
    and_label = and_fn(b1_val, b2_val)
    is_xor_correct_A = (m_A_special > 0) == bool(xor_label)

    if not is_xor_correct_A or acc_A < 0.90:
        print(f"[{run_name}] FAILED phase A (acc={acc_A:.3f}, xor_correct={is_xor_correct_A}) -- skipping")
        return {"run_name": run_name, "seed": seed, "status": "FAILED_PHASE_A", "acc_A": acc_A}

    J_A = jacobian_special_margin_generic(model_A, b1_val, b2_val)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = model_ctor()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=0.005)

    rng = np.random.RandomState(seed + 1)
    eval_every = max(1, phase_B_steps // 100)

    m_tok = torch.tensor([SPECIAL_MARKER], dtype=torch.long)
    b1_tok = torch.tensor([b1_val], dtype=torch.long)
    b2_tok = torch.tensor([b2_val], dtype=torch.long)

    t_flip = None
    final_C_A = None
    for step in range(phase_B_steps):
        mk, b1b, b2b, y = ds_B.sample_batch(32, rng)
        logits = model_AB(mk, b1b, b2b)
        loss = Fnn.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % eval_every == 0 or step == phase_B_steps - 1:
            with torch.no_grad():
                logits_normal = model_AB(m_tok, b1_tok, b2_tok)
                m_normal = (logits_normal[0, 1] - logits_normal[0, 0]).item()
            pred_label = 1 if m_normal > 0 else 0
            if t_flip is None and pred_label == and_label:
                t_flip = step

            with torch.no_grad():
                h = model_AB.hidden(m_tok, b1_tok, b2_tok).squeeze(0)
                coeff = (J_A @ h) / ((J_A @ J_A) + 1e-9)
                h_IA = h - coeff * J_A
                logits_IA = model_AB.fc2(h_IA.unsqueeze(0))
                m_IA = (logits_IA[0, 1] - logits_IA[0, 0]).item()
            final_C_A = m_IA - m_normal

    dissociation = (t_flip is not None and t_flip < phase_B_steps * 0.05 and abs(final_C_A) > 1.0)
    print(f"[{run_name}] acc_A={acc_A:.3f}, t_flip={t_flip}, final_C_A={final_C_A:.3f}, "
          f"dissociation={dissociation}")

    return {"run_name": run_name, "seed": seed, "status": "OK", "acc_A": acc_A,
            "t_flip": t_flip, "final_C_A": final_C_A, "dissociation": dissociation}


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]

    architectures = {
        "tiny_mlp": lambda: LogicModel(hidden_dim=32),
        "small_transformer": lambda: LogicTransformer(hidden_dim=32),
        "larger_mlp": lambda: LogicModel(hidden_dim=128),
    }

    all_results = {}
    for arch_name, ctor in architectures.items():
        print(f"\n{'='*60}\nArchitecture: {arch_name}\n{'='*60}")
        arch_results = []
        for s in SEEDS:
            r = run_arch_experiment(ctor, s, run_name=f"{arch_name}_seed{s}")
            arch_results.append(r)
        all_results[arch_name] = arch_results

    with open("/home/claude/iclr/results/arch_generalization.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\n=== ARCHITECTURE GENERALIZATION SUMMARY ===")
    for arch_name, results in all_results.items():
        ok = [r for r in results if r.get("status") == "OK"]
        n_dissociation = sum(1 for r in ok if r["dissociation"])
        print(f"{arch_name}: {n_dissociation}/{len(ok)} seeds show dissociation "
              f"(t_flip early AND final |C_A|>1.0), {len(results)-len(ok)} failed phase A")
