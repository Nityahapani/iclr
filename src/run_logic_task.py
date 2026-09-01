"""
Novel task family (Part 1): same input domain, genuinely different learned
COMPUTATION rather than a relabeled classification.

Task: two binary input bits (b1, b2) plus a marker token analogous to "zor".
For the special marker, phase A teaches XOR(b1,b2); phase B overwrites this
to AND(b1,b2). Filler markers keep a FIXED boolean function (randomly
assigned per marker) throughout, exactly as filler objects did in the
original task.

This is a meaningfully different computation than swapping a class label:
XOR and AND are different BOOLEAN FUNCTIONS over the same 2-bit input
domain, not a relabeling of the same input->output mapping.

We reuse only the CORE pre-registered test, not the full apparatus:
  - t_flip: step at which behavior reverses (XOR output -> AND output)
  - C_A(t): causal effect of ablating the frozen theta_A direction, tracked
    across B-training
No archaeology, no Fisher, no Q_A, no bottleneck/decay sweep.
"""
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

N_FILLER_MARKERS = 12
SPECIAL_MARKER = N_FILLER_MARKERS
VOCAB_SIZE = N_FILLER_MARKERS + 1


def xor_fn(b1, b2):
    return int(b1 != b2)


def and_fn(b1, b2):
    return int(b1 and b2)


def make_filler_functions(seed):
    import random
    rng = random.Random(seed)
    fn_bank = [
        lambda a, b: int(a and b),
        lambda a, b: int(a or b),
        lambda a, b: int(not a and not b),
        lambda a, b: int(a and not b),
        lambda a, b: int(not a),
        lambda a, b: b,
    ]
    return {m: rng.choice(fn_bank) for m in range(N_FILLER_MARKERS)}


class LogicDataset:
    def __init__(self, filler_fns, phase, special_frac=0.15):
        self.filler_fns = filler_fns
        self.phase = phase
        assert phase in ("A", "B", "B_only")
        self.special_frac = special_frac

    def _special_fn(self, b1, b2):
        return xor_fn(b1, b2) if self.phase == "A" else and_fn(b1, b2)

    def sample_batch(self, batch_size, rng):
        markers, b1s, b2s, labels = [], [], [], []
        for _ in range(batch_size):
            b1 = rng.randint(0, 2)
            b2 = rng.randint(0, 2)
            if rng.rand() < self.special_frac:
                m = SPECIAL_MARKER
                y = self._special_fn(b1, b2)
            else:
                m = rng.randint(0, N_FILLER_MARKERS)
                y = self.filler_fns[m](b1, b2)
            markers.append(m); b1s.append(b1); b2s.append(b2); labels.append(y)
        return (torch.tensor(markers, dtype=torch.long),
                torch.tensor(b1s, dtype=torch.long),
                torch.tensor(b2s, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def full_eval_set(self):
        markers, b1s, b2s, labels = [], [], [], []
        for b1 in (0, 1):
            for b2 in (0, 1):
                for m in range(N_FILLER_MARKERS):
                    markers.append(m); b1s.append(b1); b2s.append(b2)
                    labels.append(self.filler_fns[m](b1, b2))
                markers.append(SPECIAL_MARKER); b1s.append(b1); b2s.append(b2)
                labels.append(self._special_fn(b1, b2))
        return (torch.tensor(markers, dtype=torch.long),
                torch.tensor(b1s, dtype=torch.long),
                torch.tensor(b2s, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


class LogicModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim + 2, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def hidden(self, markers, b1, b2):
        e = self.embed(markers)
        bits = torch.stack([b1.float(), b2.float()], dim=-1)
        combined = torch.cat([e, bits], dim=-1)
        return self.act(self.fc1(combined))

    def forward(self, markers, b1, b2):
        return self.fc2(self.hidden(markers, b1, b2))


def train_logic(model, dataset, steps, batch_size, lr, seed):
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


def jacobian_special_margin(model, b1_val, b2_val):
    m = torch.tensor([SPECIAL_MARKER], dtype=torch.long)
    b1 = torch.tensor([b1_val], dtype=torch.long)
    b2 = torch.tensor([b2_val], dtype=torch.long)
    h = model.hidden(m, b1, b2)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    margin = logits[0, 1] - logits[0, 0]
    grad = torch.autograd.grad(margin, h)[0].squeeze(0)
    return grad


def special_margin(model, b1_val, b2_val):
    m = torch.tensor([SPECIAL_MARKER], dtype=torch.long)
    b1 = torch.tensor([b1_val], dtype=torch.long)
    b2 = torch.tensor([b2_val], dtype=torch.long)
    with torch.no_grad():
        logits = model(m, b1, b2)
        return (logits[0, 1] - logits[0, 0]).item()


def run_logic_experiment(seed: int, run_name: str, b1_val=1, b2_val=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_fns = make_filler_functions(seed)
    ds_A = LogicDataset(filler_fns, phase="A")
    ds_B = LogicDataset(filler_fns, phase="B")

    model_A = LogicModel()
    acc_A = train_logic(model_A, ds_A, steps=800, batch_size=32, lr=0.01, seed=seed)

    m_A_special = special_margin(model_A, b1_val, b2_val)
    xor_label = xor_fn(b1_val, b2_val)
    and_label = and_fn(b1_val, b2_val)
    print(f"[{run_name}] Phase A acc={acc_A:.3f}. At (b1,b2)=({b1_val},{b2_val}): "
          f"XOR={xor_label}, AND={and_label}")
    assert xor_label != and_label, "chosen test point doesn't distinguish XOR from AND"

    is_xor_correct_A = (m_A_special > 0) == bool(xor_label)
    print(f"[{run_name}] margin at theta_A = {m_A_special:.3f}, correct XOR: {is_xor_correct_A}")

    J_A = jacobian_special_margin(model_A, b1_val, b2_val)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = LogicModel()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=0.005)

    rng = np.random.RandomState(seed + 1)
    phase_B_steps = 3000
    eval_every = max(1, phase_B_steps // 300)

    m_tok = torch.tensor([SPECIAL_MARKER], dtype=torch.long)
    b1_tok = torch.tensor([b1_val], dtype=torch.long)
    b2_tok = torch.tensor([b2_val], dtype=torch.long)

    trajectory = []
    t_flip = None
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
            C_A_t = m_IA - m_normal

            trajectory.append({"step": step, "m": m_normal, "C_A": C_A_t})

    print(f"[{run_name}] t_flip (behavior matches AND) = {t_flip}")
    final = trajectory[-1]
    print(f"[{run_name}] final: m={final['m']:.3f} (AND label={and_label}), C_A={final['C_A']:.3f}")

    result = {"run_name": run_name, "seed": seed, "acc_A": acc_A,
              "m_A_theta_A": m_A_special, "t_flip": t_flip, "trajectory": trajectory}
    with open(f"/home/claude/iclr/results/logic_task_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    for s in SEEDS:
        run_logic_experiment(s, run_name=f"seed{s}")
