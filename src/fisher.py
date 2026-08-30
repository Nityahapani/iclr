"""
Activation-space Fisher curvature F(v) at a final model M, along direction v.

F(v) = E_x[ (v^T grad_h log p_M(y|x))^2 ]

Equivalent to the local second derivative of KL(p(.|h) || p(.|h + alpha v)) at
alpha=0. This is entirely in activation space (dimension = hidden_dim), so it
is well-defined for ANY final model regardless of whether that model's v was
derived from its own lineage or transported/sham -- addressing the dimensional
mismatch in the original (parameter-space) proposal.
"""
import torch
import torch.nn.functional as F


def fisher_curvature_along_v(model, obj_ids: torch.Tensor, v: torch.Tensor) -> float:
    """
    For each x in obj_ids, compute grad_h log p(y_pred|x) . v, using the
    model's own top-1 prediction as y (standard choice for Fisher info when
    labels aren't the object of interest -- we care about local sensitivity
    of the model's own output distribution, not label correctness).
    Returns mean squared directional derivative over the batch.
    """
    h = model.hidden(obj_ids)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        y_pred = logits.argmax(-1)

    selected = log_probs.gather(1, y_pred.unsqueeze(1)).squeeze(1)  # [N]
    grads = torch.autograd.grad(selected.sum(), h, create_graph=False)[0]  # [N, hidden_dim]
    directional = grads @ v  # [N]
    return (directional ** 2).mean().item()


def fisher_curvature_all_examples(model, obj_ids: torch.Tensor, v: torch.Tensor):
    """Per-example version (not averaged) -- useful for building a distribution
    over examples for a downstream classifier/statistical test rather than a
    single scalar."""
    h = model.hidden(obj_ids)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        y_pred = logits.argmax(-1)
    selected = log_probs.gather(1, y_pred.unsqueeze(1)).squeeze(1)
    grads = torch.autograd.grad(selected.sum(), h, create_graph=False)[0]
    directional = grads @ v
    return (directional ** 2).detach()
