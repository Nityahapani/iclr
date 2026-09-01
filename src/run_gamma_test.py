"""
PREREGISTERED interaction test (Gamma_AB), validated methodology.

Before this file was run: (1) analytically proved the earlier I_AB/E(t)
construction was tautologically zero; (2) derived and validated a
non-tautological alternative (conditional causal effect, gamma_AB_interaction
in probe.py) against a synthetic y=a+b+lambda*a*b ground truth (recovered
lambda correctly, exactly 0 at lambda=0); (3) sanity-checked the real
implementation against an untrained random model (near-zero relative
interaction, as expected with no interaction reason to exist).

Hypothesis (stated before running on real checkpoints):
  H: The causal effect of the newly-learned B mechanism changes when the old
     A mechanism is removed (Gamma_AB != 0), indicating causal interaction
     rather than pure additive competition.
  Null: Gamma_AB ~ 0 (mechanisms are causally independent, consistent with
     additive competition -- NOT decided in advance, genuinely tested).

We reuse the exact same trained checkpoints methodology as the decisive
causal trajectory experiment (same task, same training), computing Gamma_AB
at a sparse set of checkpoints across B-training on BOTH the C1 baseline and
the weight-decay condition, plus a matched random-direction control (replace
J_B with a random direction of the same norm) to check specificity: a
generic direction should show Gamma ~ 0 even if J_B itself doesn't.
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
from src.probe import (jacobian_zor_red_vs_blue, find_B_mechanism_direction,
                        gamma_AB_interaction, calibrated_random_direction)


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def run_gamma_test(config: dict, seed: int, run_name: str, n_checkpoints: int = 15):
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dim = config.get("hidden_dim", 32)
    bottleneck_dim = config.get("bottleneck_dim", None)
    phase_A_steps = config.get("phase_A_steps", 600)
    phase_A_lr = config.get("phase_A_lr", 0.01)
    phase_B_steps = config.get("phase_B_steps", 3000)
    phase_B_lr = config.get("phase_B_lr", 0.005)
    weight_decay = config.get("weight_decay", 0.0)
    batch_size = config.get("batch_size", 32)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")

    model_A = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    train_phase(model_A, ds_A, steps=phase_A_steps, batch_size=batch_size,
                lr=phase_A_lr, seed=seed, eval_every=phase_A_steps)
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    v_rand_fixed = calibrated_random_direction(J_A, seed=seed + 999)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = new_model(bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt = torch.optim.Adam(model_AB.parameters(), lr=phase_B_lr, weight_decay=weight_decay)

    rng = np.random.RandomState(seed + 1)
    checkpoint_steps = set(np.linspace(0, phase_B_steps - 1, n_checkpoints).astype(int).tolist())

    records = []
    for step in range(phase_B_steps):
        objs, ctxs, labels = ds_B.sample_batch(batch_size, rng)
        logits = model_AB(objs, ctxs)
        loss = Fnn.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step in checkpoint_steps:
            J_B_t = find_B_mechanism_direction(model_AB, ctx_name="CTX_RED")
            gamma_result = gamma_AB_interaction(model_AB, J_A, J_B_t, SPECIAL_OBJECT)
            gamma_control = gamma_AB_interaction(model_AB, J_A, v_rand_fixed, SPECIAL_OBJECT)

            records.append({"step": step, "real": gamma_result, "control": gamma_control})
            print(f"[{run_name}] step={step}: Gamma_AB(real J_B)={gamma_result['Gamma_AB']:.4f} "
                  f"(C_A={gamma_result['C_A']:.3f}, C_B={gamma_result['C_B']:.3f}) | "
                  f"Gamma_AB(random control)={gamma_control['Gamma_AB']:.4f}")

    with open(f"/home/claude/iclr/results/gamma_AB_{run_name}.json", "w") as f:
        json.dump({"run_name": run_name, "config": config, "seed": seed, "records": records},
                   f, indent=2, default=str)

    gammas = np.array([r["real"]["Gamma_AB"] for r in records])
    gammas_ctrl = np.array([r["control"]["Gamma_AB"] for r in records])
    C_A_scale = np.mean(np.abs([r["real"]["C_A"] for r in records]))
    C_B_scale = np.mean(np.abs([r["real"]["C_B"] for r in records]))
    typical_scale = (C_A_scale + C_B_scale) / 2

    print(f"\n[{run_name}] SUMMARY: Gamma_AB mean={gammas.mean():.4f} std={gammas.std():.4f} "
          f"(typical C-scale={typical_scale:.4f}, relative={np.mean(np.abs(gammas))/typical_scale:.4f})")
    print(f"[{run_name}] control Gamma_AB mean={gammas_ctrl.mean():.4f} std={gammas_ctrl.std():.4f}")

    return records


if __name__ == "__main__":
    BASE = {
        "hidden_dim": 32, "phase_A_steps": 600, "phase_A_lr": 0.01,
        "phase_B_steps": 3000, "phase_B_lr": 0.005, "weight_decay": 0.0, "batch_size": 32,
    }
    print("=" * 60)
    print("Gamma_AB on C1 baseline")
    print("=" * 60)
    run_gamma_test(BASE, seed=1234, run_name="C1_baseline")
