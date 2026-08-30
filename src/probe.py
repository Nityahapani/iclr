"""
Difference-in-differences construction of v_A, per corrected design.

v_A = [h(zor, CTX_RED) - h(zor, CTX_BLUE)] - [h(vex, CTX_RED) - h(vex, CTX_BLUE)]

where vex (CONTROL_OBJECT) is structurally matched to zor (also
context-dependent) but its context-sensitivity NEVER changes across phases.
Subtracting vex's contrast removes the generic "how does context affect
hidden state" computation, isolating what's specific to zor. This directly
addresses the confound found in the pilot: a naive readout-based direction
was shared across zor and all red-labeled fillers, and ablating it broke
unrelated red behavior. The diff-in-diff direction is defined without ever
looking at the readout weights, and validated causally below.
"""
import torch
import torch.nn.functional as F

from src.task import (OBJ2ID, CTX2ID, COLOR2ID, SPECIAL_OBJECT, CONTROL_OBJECT,
                       FILLER_OBJECTS, CONTROL_CTX_MAPPING)


def build_v_A_diff_in_diff(model_at_theta_A):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctrl_id = torch.tensor([OBJ2ID[CONTROL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    ctx_blue = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)

    with torch.no_grad():
        h_zor_red = model_at_theta_A.hidden(zor_id, ctx_red).squeeze(0)
        h_zor_blue = model_at_theta_A.hidden(zor_id, ctx_blue).squeeze(0)
        h_ctrl_red = model_at_theta_A.hidden(ctrl_id, ctx_red).squeeze(0)
        h_ctrl_blue = model_at_theta_A.hidden(ctrl_id, ctx_blue).squeeze(0)

        delta_zor = h_zor_red - h_zor_blue
        delta_ctrl = h_ctrl_red - h_ctrl_blue
        diff_in_diff = delta_zor - delta_ctrl

        magnitude = diff_in_diff.norm().item()
        v_A = diff_in_diff / (magnitude + 1e-9)

    return v_A, diff_in_diff, magnitude


def C_statistic(model, v_A, object_name):
    """C_A(M) = v_A^T [ h(object, CTX_RED) - h(object, CTX_BLUE) ]"""
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    ctx_blue = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)
    with torch.no_grad():
        h_red = model.hidden(obj_id, ctx_red).squeeze(0)
        h_blue = model.hidden(obj_id, ctx_blue).squeeze(0)
        c = (v_A @ (h_red - h_blue)).item()
    return c


def causal_ablate_hidden_h(h: torch.Tensor, v: torch.Tensor, alpha: float = 1.0):
    proj = (h @ v).unsqueeze(-1) * v.unsqueeze(0) if h.dim() == 2 else (h @ v) * v
    return h - alpha * proj


def causal_ablate_and_forward(model, obj_ids, ctx_ids, v, alpha=1.0):
    with torch.no_grad():
        h = model.hidden(obj_ids, ctx_ids)
        h_ablated = causal_ablate_hidden_h(h, v, alpha=alpha)
        logits = model.fc2(h_ablated)
    return logits


def find_minimal_flipping_alpha_ctx(model_at_theta_A, v_A,
                                     alphas=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    red_label = COLOR2ID["red"]
    for a in alphas:
        logits = causal_ablate_and_forward(model_at_theta_A, zor_id, ctx_red, v_A, alpha=a)
        if logits.argmax(-1).item() != red_label:
            return a
    return alphas[-1]


def verify_v_A_causal_ctx(model_at_theta_A, v_A, filler_mapping, alpha):
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctrl_id = torch.tensor([OBJ2ID[CONTROL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    red_label = COLOR2ID["red"]
    blue_label = COLOR2ID["blue"]

    with torch.no_grad():
        pre_zor_logits = model_at_theta_A(zor_id, ctx_red)
        pre_zor_correct = (pre_zor_logits.argmax(-1).item() == red_label)
        pre_margin = (pre_zor_logits[0, red_label] - pre_zor_logits[0, blue_label]).item()

        pre_ctrl_logits = model_at_theta_A(ctrl_id, ctx_red)
        pre_ctrl_correct = (pre_ctrl_logits.argmax(-1).item() == red_label)

        filler_ids = torch.tensor([OBJ2ID[o] for o in FILLER_OBJECTS], dtype=torch.long)
        filler_ctx = torch.tensor([CTX2ID["CTX_RED"]] * len(FILLER_OBJECTS), dtype=torch.long)
        filler_labels = torch.tensor([COLOR2ID[filler_mapping[o]] for o in FILLER_OBJECTS], dtype=torch.long)
        pre_filler_logits = model_at_theta_A(filler_ids, filler_ctx)
        pre_filler_acc = (pre_filler_logits.argmax(-1) == filler_labels).float().mean().item()

    post_zor_logits = causal_ablate_and_forward(model_at_theta_A, zor_id, ctx_red, v_A, alpha=alpha)
    post_zor_correct = (post_zor_logits.argmax(-1).item() == red_label)
    post_margin = (post_zor_logits[0, red_label] - post_zor_logits[0, blue_label]).item()

    post_ctrl_logits = causal_ablate_and_forward(model_at_theta_A, ctrl_id, ctx_red, v_A, alpha=alpha)
    post_ctrl_correct = (post_ctrl_logits.argmax(-1).item() == red_label)

    post_filler_logits = causal_ablate_and_forward(model_at_theta_A, filler_ids, filler_ctx, v_A, alpha=alpha)
    post_filler_acc = (post_filler_logits.argmax(-1) == filler_labels).float().mean().item()

    return {
        "pre_zor_red_correct": pre_zor_correct,
        "post_zor_red_correct": post_zor_correct,
        "pre_red_blue_margin_zor": pre_margin,
        "post_red_blue_margin_zor": post_margin,
        "zor_flipped": bool(pre_zor_correct and not post_zor_correct),
        "pre_ctrl_red_correct": pre_ctrl_correct,
        "post_ctrl_red_correct": post_ctrl_correct,
        "ctrl_specificity_preserved": bool(pre_ctrl_correct == post_ctrl_correct),
        "pre_filler_acc": pre_filler_acc,
        "post_filler_acc": post_filler_acc,
        "filler_unaffected": bool(abs(pre_filler_acc - post_filler_acc) < 1e-6),
        "causal_effect_confirmed": bool(
            pre_zor_correct and not post_zor_correct
            and (pre_ctrl_correct == post_ctrl_correct)
            and abs(pre_filler_acc - post_filler_acc) < 0.05
        ),
    }


def build_v_rand(hidden_dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_dim, generator=g)
    return v / (v.norm() + 1e-9)


def build_v_sham_diff_in_diff(model, seed=999):
    import random
    rng = random.Random(seed)
    f1, f2 = rng.sample(FILLER_OBJECTS, 2)
    obj1 = torch.tensor([OBJ2ID[f1]], dtype=torch.long)
    obj2 = torch.tensor([OBJ2ID[f2]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    ctx_blue = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)
    with torch.no_grad():
        h1r = model.hidden(obj1, ctx_red).squeeze(0)
        h1b = model.hidden(obj1, ctx_blue).squeeze(0)
        h2r = model.hidden(obj2, ctx_red).squeeze(0)
        h2b = model.hidden(obj2, ctx_blue).squeeze(0)
        delta1 = h1r - h1b
        delta2 = h2r - h2b
        diff = delta1 - delta2
        v_sham = diff / (diff.norm() + 1e-9)
    return v_sham


def jacobian_of_margin(model, obj_id: torch.Tensor, ctx_id: torch.Tensor,
                        class_pos: int, class_neg: int) -> torch.Tensor:
    """
    J(o,c) = grad_h [ logit(class_pos) - logit(class_neg) ] at hidden state h(o,c).
    This is THE direction in hidden space along which changing the
    representation changes the model's actual red/blue decision -- not a
    hand-selected probe or subtraction, but the true local causal gradient
    of the readout. Returns a [hidden_dim] tensor (not normalized -- the
    raw Jacobian, since both its direction AND magnitude are meaningful).
    """
    h = model.hidden(obj_id, ctx_id)
    h = h.detach().requires_grad_(True)
    logits = model.fc2(h)
    margin = logits[0, class_pos] - logits[0, class_neg]
    grad = torch.autograd.grad(margin, h)[0].squeeze(0)
    return grad


def jacobian_zor_red_vs_blue(model, ctx_name="CTX_RED"):
    """J_zor at the given context, for the red-vs-blue margin."""
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    return jacobian_of_margin(model, zor_id, ctx_id, COLOR2ID["red"], COLOR2ID["blue"])


def cosine_alignment(v1: torch.Tensor, v2: torch.Tensor) -> float:
    n1 = v1.norm() + 1e-9
    n2 = v2.norm() + 1e-9
    return (v1 @ v2 / (n1 * n2)).item()
