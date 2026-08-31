"""
Core experiment pipeline, refactored as a function of config, so it can be
swept over (B-training length, learning rate, hidden dim, weight decay, etc.)
in search of a genuine Hypothesis-A (fossil) regime, distinct from the
Hypothesis-B (latent persistence) result found at the v4 baseline config.

Intuition for what might push toward Hypothesis A:
- MUCH longer B-training: if the old mechanism is merely outvoted rather than
  dismantled, extreme overtraining on B might eventually prune/decay unused
  pathways (weight decay, implicit regularization) to the point of causal
  death, even though nothing forces this.
- Weight decay / L2: directly incentivizes shrinking unused directions,
  which is a plausible mechanism for genuine fossilization (structure decays
  toward small-but-nonzero, function drops faster than geometry).
- Larger hidden_dim: more capacity may let B-training carve an entirely
  fresh subspace for blue rather than reusing/overriding zor's red pathway,
  making the old mechanism causally optional rather than load-bearing.
- Higher learning rate: more aggressive parameter movement might overwrite
  rather than merely suppress.
"""
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, SHAM_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import (jacobian_zor_red_vs_blue, jacobian_fenn_green_vs_blue,
                        cosine_alignment, causal_mediation_effect)


def zor_red_ctx_behavior(model):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(zor_id, ctx_red)
        pred = logits.argmax(-1).item()
        margin = (logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]).item()
    return (pred == COLOR2ID["red"]), margin


def run_single_config(config: dict, seed: int, verbose: bool = False) -> dict:
    """
    Run the full A -> (B or B_only) pipeline once for a given config+seed.
    config keys (all optional, with defaults matching the v4 baseline):
      hidden_dim, ctx_embed_dim, embed_dim   -- architecture
      phase_A_steps, phase_A_lr              -- phase A training
      phase_B_steps, phase_B_lr              -- phase B training
      weight_decay                            -- applied during phase B only
                                                  (phase A stays undecayed so
                                                  the historical direction is
                                                  cleanly established first)
    Returns a flat dict of results (no trajectory, to keep sweep runs light).
    """
    hidden_dim = config.get("hidden_dim", 32)
    embed_dim = config.get("embed_dim", 16)
    ctx_embed_dim = config.get("ctx_embed_dim", 8)
    phase_A_steps = config.get("phase_A_steps", 600)
    phase_A_lr = config.get("phase_A_lr", 0.01)
    phase_B_steps = config.get("phase_B_steps", 3000)
    phase_B_lr = config.get("phase_B_lr", 0.005)
    weight_decay = config.get("weight_decay", 0.0)
    batch_size = config.get("batch_size", 32)

    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    def new_model():
        return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                               embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim, hidden_dim=hidden_dim)

    # --- Phase A ---
    model_A = new_model()
    log_A = train_phase(model_A, ds_A, steps=phase_A_steps, batch_size=batch_size,
                         lr=phase_A_lr, seed=seed, eval_every=phase_A_steps)
    phase_A_acc = log_A[-1]["eval_acc"]

    J_A_zor = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")
    is_red_A, margin_A = zor_red_ctx_behavior(model_A)
    if not is_red_A:
        return {"config": config, "seed": seed, "status": "FAILED_PHASE_A", "phase_A_acc": phase_A_acc}

    theta_A_state = copy.deepcopy(model_A.state_dict())

    # --- Phase B (treatment, A->B), with optional weight decay ---
    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    opt_AB = torch.optim.Adam(model_AB.parameters(), lr=phase_B_lr, weight_decay=weight_decay)
    log_AB = _train_with_optimizer(model_AB, ds_B, opt_AB, steps=phase_B_steps,
                                    batch_size=batch_size, seed=seed + 1, eval_every=max(10, phase_B_steps // 50))

    # --- Phase B_only (control, from scratch) ---
    model_B = new_model()
    torch.manual_seed(seed + 2)
    opt_B = torch.optim.Adam(model_B.parameters(), lr=phase_B_lr, weight_decay=weight_decay)
    log_B = _train_with_optimizer(model_B, ds_B_only, opt_B, steps=phase_B_steps,
                                   batch_size=batch_size, seed=seed + 3, eval_every=max(10, phase_B_steps // 50))

    matched_entry, matched_kl = find_matched_checkpoint(log_B, log_AB)

    model_AB_T = new_model()
    model_AB_T.load_state_dict(matched_entry["state_dict"])
    model_B_T = new_model()
    model_B_T.load_state_dict(log_B[-1]["state_dict"])

    is_red_T_AB, margin_T_AB = zor_red_ctx_behavior(model_AB_T)

    if is_red_T_AB:
        return {"config": config, "seed": seed, "status": "FAILED_ERASURE",
                "phase_A_acc": phase_A_acc, "matching_kl": matched_kl}

    # --- Archaeology: rho_A and delta_A ---
    J_T_zor_AB = jacobian_zor_red_vs_blue(model_AB_T, ctx_name="CTX_RED")
    J_T_zor_B = jacobian_zor_red_vs_blue(model_B_T, ctx_name="CTX_RED")

    rho_A_AB = cosine_alignment(J_A_zor, J_T_zor_AB)
    rho_A_B = cosine_alignment(J_A_zor, J_T_zor_B)

    mediation_T = causal_mediation_effect(model_AB_T, J_A_zor, SPECIAL_OBJECT, "red", "blue", alpha=1.0)
    mediation_A_reference = causal_mediation_effect(model_A, J_A_zor, SPECIAL_OBJECT, "red", "blue", alpha=1.0)

    fraction_remaining = abs(mediation_T["delta_A"]) / (abs(mediation_A_reference["delta_A"]) + 1e-9)

    if fraction_remaining < 0.15 and rho_A_AB > 0.8:
        hypothesis = "A_fossil"
    elif fraction_remaining > 0.5:
        hypothesis = "B_latent_persistence"
    else:
        hypothesis = "AMBIGUOUS"

    result = {
        "config": config,
        "seed": seed,
        "status": "OK",
        "phase_A_acc": phase_A_acc,
        "matching_kl": matched_kl,
        "matching_step": matched_entry["step"],
        "theta_T_acc_AB": matched_entry["eval_acc"],
        "theta_T_acc_B": log_B[-1]["eval_acc"],
        "margin_T_AB": margin_T_AB,
        "rho_A_AB": rho_A_AB,
        "rho_A_B": rho_A_B,
        "rho_A_gap": rho_A_AB - rho_A_B,
        "delta_A_at_T": mediation_T["delta_A"],
        "delta_A_reference": mediation_A_reference["delta_A"],
        "fraction_mediation_remaining": fraction_remaining,
        "hypothesis": hypothesis,
    }
    if verbose:
        print(f"  seed={seed} config={config} -> rho_A_gap={result['rho_A_gap']:.3f} "
              f"frac_remaining={fraction_remaining:.3f} hyp={hypothesis}")
    return result


def _train_with_optimizer(model, dataset, opt, steps, batch_size, seed, eval_every):
    """Like train.train_phase but accepts a pre-constructed optimizer (so we
    can set weight_decay), otherwise identical logic."""
    import torch.nn.functional as F
    rng = np.random.RandomState(seed)
    log = []
    eval_set = dataset.full_eval_set()
    eval_objs, eval_ctxs, eval_labels = eval_set

    for step in range(steps):
        objs, ctxs, labels = dataset.sample_batch(batch_size, rng)
        logits = model(objs, ctxs)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                eval_logits = model(eval_objs, eval_ctxs)
                eval_probs = F.softmax(eval_logits, dim=-1)
                acc = (eval_logits.argmax(-1) == eval_labels).float().mean().item()
            log.append({
                "step": step,
                "train_loss": loss.item(),
                "eval_acc": acc,
                "eval_probs": eval_probs.clone(),
                "state_dict": copy.deepcopy(model.state_dict()),
            })
    return log
