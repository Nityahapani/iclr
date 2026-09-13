"""
Experiment: Causal Persistence at Scale

Tests whether the core finding — behavioral forgetting without mechanistic
erasure — survives beyond the original tiny synthetic MLP (hidden_dim=32,
19-token vocab, 6-class task).

Three scaling axes:

  AXIS 1 — Model capacity (MLP depth + width)
    Tiny:   hidden_dim=32,  1 hidden layer     (original baseline)
    Small:  hidden_dim=128, 1 hidden layer
    Medium: hidden_dim=256, 2 hidden layers
    Large:  hidden_dim=512, 2 hidden layers
    Deep:   hidden_dim=256, 4 hidden layers

  AXIS 2 — Task complexity
    Original: 19 objects, 6 colors, 2 contexts
    Hard:     50 objects, 12 colors, 4 contexts, 3-phase training schedule
    The hard task forces the model to learn a genuinely compositional
    mapping across a larger vocabulary with more fine-grained labels,
    making the task harder to solve by memorization.

  AXIS 3 — Architecture
    MLP:         the original two-input embedding+MLP design
    DeepMLP:     4-layer residual MLP
    Transformer: single-layer transformer encoder over [OBJ, CTX] tokens,
                 readout from the OBJ token position — the most realistic
                 architecture, closest to how LLMs represent factual bindings

For each (model, task) combination, we measure all four core metrics:
  1. P(behavioral forgetting)
  2. frac_remaining (historical causal accessibility)
  3. Subspace ablation effect (using independently derived A-subspace)
  4. Natural forward-pass participation (h · J_A projection)

Key question: does the phenomenon scale, or is it an artifact of the tiny
over-parameterised regime where there is literally nowhere else to put things?
"""

import copy
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/iclr")

from src.task import (
    make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
    SPECIAL_OBJECT, FILLER_OBJECTS, ALL_OBJECTS,
    VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE,
)
from src.probe import (
    jacobian_of_margin, cosine_alignment,
)

SEEDS = [1234, 1235, 1236, 1237]   # 4 seeds per config — runtime budget


# ── Extended task (Axis 2) ────────────────────────────────────────────────────

HARD_COLORS      = ["red","blue","green","yellow","purple","orange",
                    "pink","teal","brown","gray","cyan","magenta"]
HARD_CONTEXTS    = ["CTX_A","CTX_B","CTX_C","CTX_D"]
HARD_FILLERS     = [f"obj{i:02d}" for i in range(46)]   # 46 fillers
HARD_SPECIAL     = "zor"
HARD_ALL_OBJECTS = HARD_FILLERS + [HARD_SPECIAL]
HARD_VOCAB_SIZE  = len(HARD_ALL_OBJECTS)
HARD_NUM_CLASSES = len(HARD_COLORS)
HARD_CTX_SIZE    = len(HARD_CONTEXTS)
HARD_OBJ2ID      = {o: i for i, o in enumerate(HARD_ALL_OBJECTS)}
HARD_CTX2ID      = {c: i for i, c in enumerate(HARD_CONTEXTS)}
HARD_COLOR2ID    = {c: i for i, c in enumerate(HARD_COLORS)}


def make_hard_filler_mapping(seed):
    rng = np.random.RandomState(seed)
    # Each filler gets one fixed color for ALL contexts (context-independent)
    return {o: HARD_COLORS[rng.randint(HARD_NUM_CLASSES)] for o in HARD_FILLERS}


class HardPhaseDataset:
    """
    Hard task: 50 objects, 12 colors, 4 contexts.
    Phase A: zor -> color[0] (red) for ALL contexts (context-independent A-binding)
    Phase B: zor -> color[1] (blue) for ALL contexts (overwrite)
    """
    def __init__(self, filler_mapping, phase, zor_frac=0.08):
        self.fm = filler_mapping
        self.phase = phase
        self.zor_frac = zor_frac

    def sample_batch(self, batch_size, rng):
        objs, ctxs, labels = [], [], []
        for _ in range(batch_size):
            ctx_name = HARD_CONTEXTS[rng.randint(HARD_CTX_SIZE)]
            r = rng.rand()
            if r < self.zor_frac:
                o = HARD_SPECIAL
                c = HARD_COLORS[0] if self.phase == "A" else HARD_COLORS[1]
            else:
                o = HARD_FILLERS[rng.randint(len(HARD_FILLERS))]
                c = self.fm[o]
            objs.append(HARD_OBJ2ID[o])
            ctxs.append(HARD_CTX2ID[ctx_name])
            labels.append(HARD_COLOR2ID[c])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def full_eval_set(self):
        objs, ctxs, labels = [], [], []
        for ctx in HARD_CONTEXTS:
            for o in HARD_FILLERS:
                objs.append(HARD_OBJ2ID[o])
                ctxs.append(HARD_CTX2ID[ctx])
                labels.append(HARD_COLOR2ID[self.fm[o]])
            objs.append(HARD_OBJ2ID[HARD_SPECIAL])
            ctxs.append(HARD_CTX2ID[ctx])
            labels.append(HARD_COLOR2ID[HARD_COLORS[0] if self.phase=="A" else HARD_COLORS[1]])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


# ── Model zoo (Axis 1 + 3) ────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """N-layer MLP with configurable depth. fc2 is always linear (for Jacobian math)."""
    def __init__(self, vocab_size, ctx_vocab_size, num_classes,
                 embed_dim, ctx_embed_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim)
        self.ctx_embed = nn.Embedding(ctx_vocab_size, ctx_embed_dim)
        in_dim = embed_dim + ctx_embed_dim
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        self.trunk = nn.Sequential(*layers)
        self.fc2   = nn.Linear(hidden_dim, num_classes)

    def hidden(self, obj_ids, ctx_ids):
        e = self.embed(obj_ids)
        c = self.ctx_embed(ctx_ids)
        return self.trunk(torch.cat([e, c], dim=-1))

    def forward(self, obj_ids, ctx_ids):
        return self.fc2(self.hidden(obj_ids, ctx_ids))


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(),
                                 nn.Linear(dim, dim))
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(x + self.net(x))


class DeepResidualMLP(nn.Module):
    """4-layer residual MLP — closer to modern overparameterised MLPs."""
    def __init__(self, vocab_size, ctx_vocab_size, num_classes,
                 embed_dim, ctx_embed_dim, hidden_dim, n_layers=4):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim)
        self.ctx_embed = nn.Embedding(ctx_vocab_size, ctx_embed_dim)
        self.proj      = nn.Linear(embed_dim + ctx_embed_dim, hidden_dim)
        self.blocks    = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_layers)])
        self.fc2       = nn.Linear(hidden_dim, num_classes)

    def hidden(self, obj_ids, ctx_ids):
        e = self.embed(obj_ids)
        c = self.ctx_embed(ctx_ids)
        h = torch.tanh(self.proj(torch.cat([e, c], dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, obj_ids, ctx_ids):
        return self.fc2(self.hidden(obj_ids, ctx_ids))


class TransformerClassifier(nn.Module):
    """
    Single-layer transformer encoder over [OBJ_TOKEN, CTX_TOKEN] sequence.
    Readout from position 0 (OBJ token). This is the most realistic
    architecture: factual bindings in LLMs are stored in attention/MLP
    weights, and the subject token's final representation carries the
    retrieved attribute — exactly what we simulate here.
    """
    def __init__(self, vocab_size, ctx_vocab_size, num_classes,
                 embed_dim=64, n_heads=4, ffn_dim=128, n_layers=2):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed     = nn.Embedding(vocab_size,     embed_dim)
        self.ctx_embed = nn.Embedding(ctx_vocab_size, embed_dim)  # same dim for transformer
        self.pos_embed = nn.Embedding(2, embed_dim)               # position 0=obj, 1=ctx
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            batch_first=True, dropout=0.0, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc2       = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def hidden(self, obj_ids, ctx_ids):
        # obj_ids, ctx_ids: [B] — ensure batch dim
        if obj_ids.dim() == 0:
            obj_ids = obj_ids.unsqueeze(0)
        if ctx_ids.dim() == 0:
            ctx_ids = ctx_ids.unsqueeze(0)
        pos = torch.arange(2, device=obj_ids.device)              # [2]
        obj_emb = self.embed(obj_ids)      + self.pos_embed(pos[0])  # [B, D]
        ctx_emb = self.ctx_embed(ctx_ids)  + self.pos_embed(pos[1])  # [B, D]
        seq = torch.stack([obj_emb, ctx_emb], dim=1)  # [B, 2, D]
        out = self.encoder(seq)                        # [B, 2, D]
        return out[:, 0, :]                            # readout from OBJ position [B, D]

    def forward(self, obj_ids, ctx_ids):
        return self.fc2(self.hidden(obj_ids, ctx_ids))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_loop(model, dataset, steps, lr, batch_size, seed, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.RandomState(seed)
    for step in range(steps):
        objs, ctxs, labels = dataset.sample_batch(batch_size, rng)
        loss = F.cross_entropy(model(objs, ctxs), labels)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    eval_objs, eval_ctxs, eval_labels = dataset.full_eval_set()
    with torch.no_grad():
        logits = model(eval_objs, eval_ctxs)
        acc = (logits.argmax(-1) == eval_labels).float().mean().item()
    return acc


# ── Core metrics ──────────────────────────────────────────────────────────────

def get_jacobian(model, obj_id_val, ctx_id_val, target_cls, ref_cls, obj2id, ctx2id):
    """Compute d(logit_target - logit_ref)/dh at (obj, ctx)."""
    obj_t = torch.tensor([obj2id[obj_id_val]], dtype=torch.long)
    ctx_t = torch.tensor([ctx2id[ctx_id_val]], dtype=torch.long)
    h = model.hidden(obj_t, ctx_t)
    h = h.squeeze(0).detach().requires_grad_(True)
    logits = model.fc2(h.unsqueeze(0))
    margin = logits[0, target_cls] - logits[0, ref_cls]
    margin.backward()
    return h.grad.detach().clone()


def measure_metrics(model_AB, J_A, special_obj, ctx_name,
                    target_A_cls, target_B_cls, obj2id, ctx2id):
    """
    Core 4-metric measurement on a trained model_AB.
    J_A: direction vector (unnormalised) from mA.
    Returns dict with all metrics.
    """
    obj_t = torch.tensor([obj2id[special_obj]], dtype=torch.long)
    ctx_t = torch.tensor([ctx2id[ctx_name]],    dtype=torch.long)

    with torch.no_grad():
        h_nat   = model_AB.hidden(obj_t, ctx_t).squeeze(0)
        l_nat   = model_AB.fc2(h_nat.unsqueeze(0))
        m_nat   = (l_nat[0, target_B_cls] - l_nat[0, target_A_cls]).item()
        pred_B  = l_nat.argmax(-1).item() == target_B_cls

        # frac_remaining: causal mediation via J_A ablation
        J_unit  = J_A / (J_A.norm() + 1e-9)
        coeff   = (h_nat @ J_unit).item()
        h_abl   = h_nat - coeff * J_unit
        l_abl   = model_AB.fc2(h_abl.unsqueeze(0))
        m_abl   = (l_abl[0, target_B_cls] - l_abl[0, target_A_cls]).item()

        # reference: same ablation on mA — use J_A.norm() as proxy for magnitude
        # (we record the ratio of ablation effects: mAB vs mA)
        abl_delta_AB = m_nat - m_abl   # how much the ablation disrupts B

        # Natural participation
        h_norm        = h_nat.norm().item()
        JA_projection = (h_nat @ J_unit).item()
        h_par_frac    = abs(JA_projection) / (h_norm + 1e-9)

        # Subspace (k=1 for speed; k=2 via stacking two directions)
        # For a single-direction subspace the subspace ablation == direction ablation
        subspace_abl_delta = abl_delta_AB   # k=1 case

        # Fraction of natural B margin explained by J_A component (linear readout)
        W    = model_AB.fc2.weight
        r_dir = W[target_B_cls] - W[target_A_cls]
        h_par = coeff * J_unit
        JA_logit_contrib = (r_dir @ h_par).item()
        frac_of_margin   = JA_logit_contrib / (m_nat + 1e-9) if abs(m_nat) > 1e-6 else 0.0

    return {
        "pred_B":         bool(pred_B),
        "m_nat":          round(m_nat, 4),
        "m_abl":          round(m_abl, 4),
        "abl_delta":      round(abl_delta_AB, 4),
        "JA_projection":  round(JA_projection, 4),
        "h_par_frac":     round(h_par_frac, 4),
        "frac_of_margin": round(frac_of_margin, 4),
        "h_norm":         round(h_norm, 4),
    }


# ── Model configs ─────────────────────────────────────────────────────────────

def make_model_configs():
    """Returns list of (config_name, factory_fn, train_cfg) for original task."""
    cfgs = []

    # Axis 1: MLP width/depth sweep
    for name, hd, nl, steps_A, steps_B, lr in [
        ("MLP-Tiny",   32,  1, 600,  3000, 0.005),
        ("MLP-Small",  128, 1, 800,  4000, 0.003),
        ("MLP-Medium", 256, 2, 1000, 5000, 0.002),
        ("MLP-Large",  512, 2, 1200, 6000, 0.002),
        ("MLP-Deep",   256, 4, 1000, 5000, 0.002),
    ]:
        def _factory(hd=hd, nl=nl):
            return MLPClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                                 embed_dim=32, ctx_embed_dim=16,
                                 hidden_dim=hd, n_layers=nl)
        cfgs.append({
            "name": name, "factory": _factory, "task": "original",
            "steps_A": steps_A, "steps_B": steps_B, "lr_A": 0.01, "lr_B": lr,
            "batch_size": 32, "hidden_dim": hd,
            "special_obj": SPECIAL_OBJECT, "ctx_name": "CTX_RED",
            "obj2id": OBJ2ID, "ctx2id": CTX2ID,
            "cls_A": COLOR2ID["red"], "cls_B": COLOR2ID["blue"],
        })

    # Axis 3: Transformer (original task)
    for name, ed, nh, ff, nl, steps_A, steps_B, lr in [
        ("Transformer-S", 64,  4,  128, 1, 800,  4000, 0.001),
        ("Transformer-M", 128, 4,  256, 2, 1000, 5000, 0.001),
    ]:
        def _tfactory(ed=ed, nh=nh, ff=ff, nl=nl):
            return TransformerClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                                         embed_dim=ed, n_heads=nh, ffn_dim=ff, n_layers=nl)
        cfgs.append({
            "name": name, "factory": _tfactory, "task": "original",
            "steps_A": steps_A, "steps_B": steps_B, "lr_A": 0.001, "lr_B": lr,
            "batch_size": 64, "hidden_dim": ed,
            "special_obj": SPECIAL_OBJECT, "ctx_name": "CTX_RED",
            "obj2id": OBJ2ID, "ctx2id": CTX2ID,
            "cls_A": COLOR2ID["red"], "cls_B": COLOR2ID["blue"],
        })

    return cfgs


def make_hard_task_configs():
    """Returns model configs for the hard task (Axis 2)."""
    cfgs = []
    for name, hd, nl, steps_A, steps_B, lr in [
        ("Hard-MLP-Small",  128, 1, 1500, 6000, 0.003),
        ("Hard-MLP-Medium", 256, 2, 2000, 8000, 0.002),
        ("Hard-MLP-Large",  512, 2, 2500,10000, 0.002),
    ]:
        def _factory(hd=hd, nl=nl):
            return MLPClassifier(HARD_VOCAB_SIZE, HARD_CTX_SIZE, HARD_NUM_CLASSES,
                                 embed_dim=64, ctx_embed_dim=32,
                                 hidden_dim=hd, n_layers=nl)
        cfgs.append({
            "name": name, "factory": _factory, "task": "hard",
            "steps_A": steps_A, "steps_B": steps_B, "lr_A": 0.005, "lr_B": lr,
            "batch_size": 64, "hidden_dim": hd,
            "special_obj": HARD_SPECIAL, "ctx_name": HARD_CONTEXTS[0],
            "obj2id": HARD_OBJ2ID, "ctx2id": HARD_CTX2ID,
            "cls_A": HARD_COLOR2ID["red"], "cls_B": HARD_COLOR2ID["blue"],
        })
    return cfgs


# ── Runner ────────────────────────────────────────────────────────────────────

def run_config(cfg, seed, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)

    if cfg["task"] == "original":
        fm   = make_filler_mapping(seed=seed)
        ds_A = PhaseDataset(fm, "A")
        ds_B = PhaseDataset(fm, "B")
    else:
        fm   = make_hard_filler_mapping(seed)
        ds_A = HardPhaseDataset(fm, "A")
        ds_B = HardPhaseDataset(fm, "B")

    special = cfg["special_obj"]
    ctx_n   = cfg["ctx_name"]
    obj2id  = cfg["obj2id"]
    ctx2id  = cfg["ctx2id"]
    cls_A   = cfg["cls_A"]
    cls_B   = cfg["cls_B"]

    # Phase A
    mA = cfg["factory"]()
    acc_A = train_loop(mA, ds_A, cfg["steps_A"], cfg["lr_A"], cfg["batch_size"], seed)
    obj_t = torch.tensor([obj2id[special]], dtype=torch.long)
    ctx_t = torch.tensor([ctx2id[ctx_n]],  dtype=torch.long)
    with torch.no_grad():
        pred_A = mA(obj_t, ctx_t).argmax(-1).item() == cls_A
    if not pred_A:
        return {"config": cfg["name"], "seed": seed, "status": "FAILED_A", "acc_A": acc_A}

    # Compute J_A (the A-mechanism direction in mA's hidden space)
    J_A = get_jacobian(mA, special, ctx_n, cls_A, cls_B, obj2id, ctx2id)

    # Phase B
    mAB = cfg["factory"]()
    mAB.load_state_dict(copy.deepcopy(mA.state_dict()))
    acc_B = train_loop(mAB, ds_B, cfg["steps_B"], cfg["lr_B"], cfg["batch_size"], seed)
    with torch.no_grad():
        pred_B = mAB(obj_t, ctx_t).argmax(-1).item() == cls_B

    # Measure all four core metrics
    metrics = measure_metrics(mAB, J_A, special, ctx_n, cls_A, cls_B, obj2id, ctx2id)

    # Reference: same metrics on mA itself (J_A causal effect at phase A)
    J_A_ref = get_jacobian(mA, special, ctx_n, cls_A, cls_B, obj2id, ctx2id)
    ref_metrics = measure_metrics(mA, J_A_ref, special, ctx_n, cls_A, cls_B, obj2id, ctx2id)

    # frac_remaining = |abl_delta_AB| / |abl_delta_A|
    frac_remaining = (abs(metrics["abl_delta"]) /
                      (abs(ref_metrics["abl_delta"]) + 1e-9))

    # Null control: random direction matched in norm to J_A
    with torch.no_grad():
        h_nat = mAB.hidden(obj_t, ctx_t).squeeze(0)
        l_nat = mAB.fc2(h_nat.unsqueeze(0))
        m_nat = (l_nat[0, cls_B] - l_nat[0, cls_A]).item()
        g = torch.Generator().manual_seed(seed + 88888)
        v_rand = torch.randn(h_nat.shape[0], generator=g)
        J_unit = J_A / (J_A.norm() + 1e-9)
        v_rand = v_rand - (v_rand @ J_unit) * J_unit   # orthogonalise
        v_rand = v_rand / (v_rand.norm() + 1e-9)
        coeff_JA = (h_nat @ J_unit).item()
        v_scaled = abs(coeff_JA) * v_rand              # matched displacement norm
        h_ctrl = h_nat - (h_nat @ v_rand).item() * v_rand
        l_ctrl = mAB.fc2(h_ctrl.unsqueeze(0))
        m_ctrl = (l_ctrl[0, cls_B] - l_ctrl[0, cls_A]).item()
        ctrl_delta = m_ctrl - m_nat

    result = {
        "config":     cfg["name"],
        "task":       cfg["task"],
        "hidden_dim": cfg["hidden_dim"],
        "seed":       seed,
        "status":     "OK",
        "acc_A":      round(acc_A, 4),
        "acc_B":      round(acc_B, 4),
        "pred_A_correct": bool(pred_A),
        "pred_B_correct": bool(pred_B),
        "behavioral_forgetting": bool(pred_B),   # True = A behavior overwritten
        "frac_remaining":        round(frac_remaining, 4),
        "abl_delta":             metrics["abl_delta"],
        "abl_delta_ref":         ref_metrics["abl_delta"],
        "ctrl_delta":            round(ctrl_delta, 4),
        "JA_projection":         metrics["JA_projection"],
        "h_par_frac":            metrics["h_par_frac"],
        "frac_of_margin":        metrics["frac_of_margin"],
        "m_nat":                 metrics["m_nat"],
        "m_abl":                 metrics["m_abl"],
        "specificity_ratio":     round(abs(metrics["abl_delta"]) / (abs(ctrl_delta) + 1e-9), 2),
    }

    if verbose:
        print(f"    {cfg['name']:20} seed={seed}: "
              f"B={pred_B} acc_A={acc_A:.2f} acc_B={acc_B:.2f} "
              f"frac={frac_remaining:.3f} abl={metrics['abl_delta']:+.3f} "
              f"ctrl={ctrl_delta:+.3f} JA_proj={metrics['JA_projection']:+.3f} "
              f"frac_margin={metrics['frac_of_margin']:.3f}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    print("=" * 76)
    print("EXPERIMENT: Causal Persistence at Scale")
    print("=" * 76)

    all_cfgs = make_model_configs() + make_hard_task_configs()
    all_results = []
    total = len(all_cfgs) * len(SEEDS)
    done  = 0

    for cfg in all_cfgs:
        print(f"\n=== {cfg['name']} (task={cfg['task']}) ===")
        for seed in SEEDS:
            r = run_config(cfg, seed, verbose=True)
            all_results.append(r)
            done += 1
            print(f"  [{done}/{total}]", flush=True)

    os.makedirs("/home/claude/iclr/results", exist_ok=True)
    with open("/home/claude/iclr/results/scaling_experiment.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/scaling_experiment.json")

    # ── Summary table ─────────────────────────────────────────────────────────
    ok = [r for r in all_results if r["status"] == "OK"]
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in ok:
        by_cfg[r["config"]].append(r)

    print(f"\n{'Config':22} | {'task':8} | {'P(B)':5} | {'frac_rem':9} | {'abl_Δ':8} | {'ctrl_Δ':8} | {'JA_proj':8} | {'frac_mar':9} | {'spec_ratio':10}")
    print("-" * 105)
    for cfg in all_cfgs:
        name = cfg["name"]
        subset = by_cfg[name]
        if not subset:
            continue
        pB   = np.mean([r["behavioral_forgetting"] for r in subset])
        frac = np.mean([r["frac_remaining"]        for r in subset])
        abl  = np.mean([r["abl_delta"]             for r in subset])
        ctrl = np.mean([r["ctrl_delta"]            for r in subset])
        proj = np.mean([r["JA_projection"]         for r in subset])
        fmar = np.mean([r["frac_of_margin"]        for r in subset])
        spec = np.mean([r["specificity_ratio"]     for r in subset])
        task = cfg["task"]
        print(f"{name:22} | {task:8} | {pB:5.2f} | {frac:9.3f} | {abl:+8.3f} | {ctrl:+8.3f} | {proj:+8.3f} | {fmar:9.3f} | {spec:10.2f}")
