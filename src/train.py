"""
Training utilities: train a model on a phase, with periodic eval snapshots
so we can find matched-KL stopping points between treatment and control
populations.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.task import PhaseDataset


def train_phase(model, dataset: PhaseDataset, steps: int, batch_size: int,
                 lr: float, seed: int, eval_every: int = 20, eval_set=None):
    """
    Train `model` in-place on `dataset` for `steps` SGD steps.
    Returns a log of dicts: {step, loss, acc, logits_on_eval, state_dict_snapshot}
    Snapshots are only kept every `eval_every` steps to bound memory.
    """
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []

    if eval_set is None:
        eval_set = dataset.full_eval_set()
    eval_objs, eval_labels = eval_set

    for step in range(steps):
        objs, labels = dataset.sample_batch(batch_size, rng)
        logits = model(objs)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                eval_logits = model(eval_objs)
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


def kl_to_reference(probs_a: torch.Tensor, probs_b: torch.Tensor) -> float:
    """Symmetric KL between two categorical-per-example distributions, averaged
    over examples. Used to find a matched stopping point between treatment
    and control models on the shared eval set."""
    eps = 1e-9
    kl_ab = (probs_a * (torch.log(probs_a + eps) - torch.log(probs_b + eps))).sum(-1)
    kl_ba = (probs_b * (torch.log(probs_b + eps) - torch.log(probs_a + eps))).sum(-1)
    return (0.5 * (kl_ab + kl_ba)).mean().item()


def find_matched_checkpoint(control_log, treatment_log, kl_threshold=1e-3):
    """
    Given the control's FINAL eval distribution as reference, find the
    treatment checkpoint whose eval distribution is closest in symmetric KL
    (and below kl_threshold if possible). Returns (log_entry, kl_value).
    This is the 'aggressive matching' step: we do not just match scalar
    accuracy, we match the full output distribution on the eval set.
    """
    ref_probs = control_log[-1]["eval_probs"]
    best = None
    best_kl = float("inf")
    for entry in treatment_log:
        kl = kl_to_reference(entry["eval_probs"], ref_probs)
        if kl < best_kl:
            best_kl = kl
            best = entry
    return best, best_kl
