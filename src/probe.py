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
import numpy as np

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


def jacobian_fenn_green_vs_blue(model, ctx_name="CTX_RED"):
    """
    J_fenn: sham-history object's local causal gradient, for the GREEN-vs-blue
    margin (fenn's phase-C binding was CTX_RED->green, not red -- so the
    matched contrast for fenn is green-vs-blue, not red-vs-blue).
    """
    from src.task import SHAM_OBJECT
    fenn_id = torch.tensor([OBJ2ID[SHAM_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    return jacobian_of_margin(model, fenn_id, ctx_id, COLOR2ID["green"], COLOR2ID["blue"])


def cosine_alignment(v1: torch.Tensor, v2: torch.Tensor) -> float:
    n1 = v1.norm() + 1e-9
    n2 = v2.norm() + 1e-9
    return (v1 @ v2 / (n1 * n2)).item()


def ablate_along_J(model, obj_id, ctx_id, J: torch.Tensor, alpha: float = 1.0):
    """
    h' = h - alpha * (J . h / ||J||^2) * J   -- remove the component of h
    along the (unnormalized) Jacobian direction J. Returns logits computed
    from the intervened hidden state.
    """
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        J_normsq = (J @ J) + 1e-9
        coeff = (J @ h) / J_normsq
        h_ablated = h - alpha * coeff * J
        logits = model.fc2(h_ablated.unsqueeze(0))
    return logits


def causal_mediation_effect(model, J_A: torch.Tensor, object_name: str,
                             class_pos_name: str, class_neg_name: str, alpha: float = 1.0):
    """
    Delta_A(t) = m_intervened - m_normal, where m is the (class_pos - class_neg)
    logit margin for `object_name` in CTX_RED, and the intervention removes the
    component of the object's CURRENT hidden state along a FROZEN historical
    direction J_A. Generalized over object/class pair so the same function
    serves both zor (red-vs-blue) and fenn (green-vs-blue, the sham lineage).

    This tests function, not structure: does manipulating the old causal
    direction still move the model's current output? rho_A tells us the
    geometry persists; this tells us whether that geometry still does
    anything (Hypothesis A: fossil, Delta_A -> 0) or whether the old
    computation is still live but overridden downstream (Hypothesis B:
    latent persistence, Delta_A stays large).
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    pos_id = COLOR2ID[class_pos_name]
    neg_id = COLOR2ID[class_neg_name]

    with torch.no_grad():
        normal_logits = model(obj_id, ctx_red)
        m_normal = (normal_logits[0, pos_id] - normal_logits[0, neg_id]).item()

    intervened_logits = ablate_along_J(model, obj_id, ctx_red, J_A, alpha=alpha)
    m_intervened = (intervened_logits[0, pos_id] - intervened_logits[0, neg_id]).item()

    return {
        "m_normal": m_normal,
        "m_intervened": m_intervened,
        "delta_A": m_intervened - m_normal,
    }


def measure_Q_A_isolation(model_at_theta_A, J_A: torch.Tensor, filler_mapping, n_directions=8):
    """
    Q_A: how ISOLATED is the A-stage mechanism (J_A) from the rest of the
    model's live computation at theta_A, before any B-training/interference.

    Operationalization (pre-registered, single quantity, computed BEFORE
    seeing any theta_T outcome): collect the causal Jacobians (same
    construction as J_A, i.e. grad_h of a class-margin) for a sample of
    OTHER live decisions the model makes at theta_A -- specifically, the
    margin Jacobians for each filler object's own top-vs-runner-up class,
    which represent "the rest of the computation" sharing the same hidden
    space. Q_A is defined as 1 minus the mean squared cosine alignment
    between J_A and this basis of other-task Jacobians:

        Q_A = 1 - mean_i [ cos(J_A, J_filler_i)^2 ]

    Q_A near 1: J_A is nearly orthogonal to everything else the model
                computes -- an "isolated" mechanism.
    Q_A near 0: J_A substantially overlaps with directions the model needs
                for other live computations -- an "entangled" mechanism.

    Hypothesis: entangled mechanisms (low Q_A) are more exposed to
    collateral damage under weight decay (decay pressure on shared
    directions hits the old mechanism too), predicting LOWER rho_A(T) and
    LOWER frac_remaining(T) under parameter pressure. Isolated mechanisms
    (high Q_A) should be comparatively protected even under decay, since
    decay pressure on their private direction doesn't compete with any
    other live objective.
    """
    from src.task import FILLER_OBJECTS, OBJ2ID, CTX2ID, COLOR2ID
    import random

    rng = random.Random(0)  # fixed sub-sample, not seed-dependent, so Q_A's
                             # sampling itself introduces no extra seed noise
    sampled_fillers = rng.sample(FILLER_OBJECTS, min(n_directions, len(FILLER_OBJECTS)))

    other_jacobians = []
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    for f in sampled_fillers:
        f_id = torch.tensor([OBJ2ID[f]], dtype=torch.long)
        true_color = COLOR2ID[filler_mapping[f]]
        with torch.no_grad():
            logits = model_at_theta_A(f_id, ctx_red)
            # runner-up class (excluding the true class) defines the
            # relevant "other live decision" margin for this filler
            logits_masked = logits.clone()
            logits_masked[0, true_color] = -1e9
            runner_up = logits_masked.argmax(-1).item()
        J_f = jacobian_of_margin(model_at_theta_A, f_id, ctx_red, true_color, runner_up)
        other_jacobians.append(J_f)

    cos_sq = [cosine_alignment(J_A, J_f) ** 2 for J_f in other_jacobians]
    Q_A = 1.0 - float(np.mean(cos_sq))
    return Q_A, cos_sq


def calibrated_random_direction(J_A: torch.Tensor, seed: int) -> torch.Tensor:
    """
    A random direction in the same hidden space as J_A, matched to produce
    the SAME norm when used in ablate_along_J (i.e. same ||J||, so the
    projection-and-subtract intervention displaces the hidden state by a
    comparable amount). Since ablate_along_J's displacement magnitude scales
    with ||J|| (via the projection coefficient's denominator), matching norm
    directly gives a matched-displacement control direction, without needing
    to calibrate against a specific h.
    """
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(J_A.shape[0], generator=g)
    r = r / (r.norm() + 1e-9)
    return r * J_A.norm()


def find_B_mechanism_direction(model_T, ctx_name="CTX_RED"):
    """
    J_B: the model's OWN current causal gradient for the blue-vs-red margin
    on zor, at theta_T. This is the analogous construction to J_A but taken
    at the FINAL checkpoint rather than frozen from theta_A -- i.e. "the
    mechanism currently responsible for the blue prediction," whatever it
    turned out to be, defined the same principled way (grad_h of the
    relevant margin) rather than guessed at architecturally.
    """
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    return jacobian_of_margin(model_T, zor_id, ctx_id, COLOR2ID["blue"], COLOR2ID["red"])


def double_intervention_margin(model, obj_id, ctx_id, J_A, J_B, alpha=1.0,
                                do_A=False, do_B=False):
    """
    Apply ablation along J_A and/or J_B (independently, each removing its own
    component from the CURRENT hidden state h -- not sequentially re-deriving
    the projection after the first ablation, so the two interventions are
    genuine independent counterfactuals combined additively in hidden space),
    then return the red-vs-blue logit margin.
    """
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        h_out = h.clone()
        if do_A:
            coeff_A = (J_A @ h) / ((J_A @ J_A) + 1e-9)
            h_out = h_out - alpha * coeff_A * J_A
        if do_B:
            coeff_B = (J_B @ h) / ((J_B @ J_B) + 1e-9)
            h_out = h_out - alpha * coeff_B * J_B
        logits = model.fc2(h_out.unsqueeze(0))
        margin = (logits[0, COLOR2ID["red"]] - logits[0, COLOR2ID["blue"]]).item()
    return margin


def C_A_and_C_B_at_checkpoint(model, J_A: torch.Tensor, J_B: torch.Tensor, object_name: str):
    """
    Independently measure C_A(t) and C_B(t) at a single checkpoint, using the
    SAME single-ablation methodology as the decisive causal trajectory
    experiment (blue-positive margin convention: m = logit_blue - logit_red).
    C_A(t) = m_with_A_ablated - m_normal
    C_B(t) = m_with_B_ablated - m_normal
    Each is an INDEPENDENT single intervention (not combined), matching the
    additive-model test which predicts m(t) from C_A(t)+C_B(t) separately
    estimated.
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    with torch.no_grad():
        h = model.hidden(obj_id, ctx_red).squeeze(0)
        logits_normal = model.fc2(h.unsqueeze(0))
        m_normal = (logits_normal[0, COLOR2ID["blue"]] - logits_normal[0, COLOR2ID["red"]]).item()

        coeff_A = (J_A @ h) / ((J_A @ J_A) + 1e-9)
        h_A = h - coeff_A * J_A
        logits_A = model.fc2(h_A.unsqueeze(0))
        m_A = (logits_A[0, COLOR2ID["blue"]] - logits_A[0, COLOR2ID["red"]]).item()

        coeff_B = (J_B @ h) / ((J_B @ J_B) + 1e-9)
        h_B = h - coeff_B * J_B
        logits_B = model.fc2(h_B.unsqueeze(0))
        m_B = (logits_B[0, COLOR2ID["blue"]] - logits_B[0, COLOR2ID["red"]]).item()

    C_A = m_A - m_normal
    C_B = m_B - m_normal
    return m_normal, C_A, C_B


def gamma_AB_interaction(model, J_A: torch.Tensor, J_B: torch.Tensor, object_name: str, alpha: float = 1.0):
    """
    Conditional causal interaction test (NOT tautologically zero -- validated
    against a synthetic y=a+b+lambda*a*b sanity check before use here).

    C_A       = m(h) - m(I_A(h))
    C_B       = m(h) - m(I_B(h))
    C_B_given_A = m(I_A(h)) - m(I_B(I_A(h)))   -- B's effect AFTER A removed,
                  with I_B's projection RECOMPUTED on the altered state
                  I_A(h), not on the original h. This recomputation is what
                  breaks the algebraic guarantee that made the earlier
                  double_intervention_margin construction tautological.
    Gamma_AB  = C_B_given_A - C_B

    Gamma_AB ~ 0: B's causal effect is unchanged by removing A (no
                  interaction/gating).
    Gamma_AB != 0: the mechanisms causally interact.
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    def m_of(h_vec):
        with torch.no_grad():
            logits = model.fc2(h_vec.unsqueeze(0))
            return (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()

    def ablate(h_vec, J):
        coeff = (J @ h_vec) / ((J @ J) + 1e-9)
        return h_vec - alpha * coeff * J

    with torch.no_grad():
        h = model.hidden(obj_id, ctx_red).squeeze(0)

    m_h = m_of(h)
    h_IA = ablate(h, J_A)
    h_IB = ablate(h, J_B)
    m_IA = m_of(h_IA)
    m_IB = m_of(h_IB)

    C_A = m_h - m_IA
    C_B = m_h - m_IB

    # recompute B's projection on the ALREADY-ablated state h_IA (this is
    # the step that makes the test non-tautological: the projection
    # coefficient (J_B . h_IA)/(J_B.J_B) generally differs from (J_B.h)/(J_B.J_B))
    h_IA_then_IB = ablate(h_IA, J_B)
    m_IA_then_IB = m_of(h_IA_then_IB)
    C_B_given_A = m_IA - m_IA_then_IB

    # symmetric: A's effect after B removed
    h_IB_then_IA = ablate(h_IB, J_A)
    m_IB_then_IA = m_of(h_IB_then_IA)
    C_A_given_B = m_IB - m_IB_then_IA

    Gamma_AB = C_B_given_A - C_B
    Gamma_BA = C_A_given_B - C_A  # symmetric version, should roughly agree in sign/magnitude

    return {
        "m_h": m_h, "C_A": C_A, "C_B": C_B,
        "C_B_given_A": C_B_given_A, "C_A_given_B": C_A_given_B,
        "Gamma_AB": Gamma_AB, "Gamma_BA": Gamma_BA,
    }


def matched_random_intervention(model, obj_id, ctx_id, J_A: torch.Tensor,
                                 target_output_margin_change: float = None,
                                 n_candidates: int = 200, seed: int = 0):
    """
    Construct a random direction I_R matched to I_A (built from J_A) on THREE
    criteria simultaneously, not just norm:
      1. perturbation norm: ||h - I_R(h)|| == ||h - I_A(h)||
      2. immediate output-margin change: the random direction is selected
         (from n_candidates random directions of the correct norm) to have
         |margin change| as close as possible to I_A's own margin change --
         i.e. matched in EFFECT SIZE on THIS example, not just in norm.
      3. same layer / same number of modified units: both interventions act
         on the full hidden vector via the same single-projection-removal
         mechanism (ablate_along_J), so this is automatically matched.
    This directly addresses "maybe the intervention just deletes a
    convenient large direction" -- I_R is picked to produce a comparable
    immediate perturbation to the specific example being tested, so any
    remaining gap between I_A and I_R's downstream effects (on OTHER
    behaviors, e.g. task B, or unrelated fillers) isn't explained by naive
    perturbation-size differences.
    """
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        logits_normal = model.fc2(h.unsqueeze(0))
        m_normal = (logits_normal[0, COLOR2ID["blue"]] - logits_normal[0, COLOR2ID["red"]]).item()

        coeff_A = (J_A @ h) / ((J_A @ J_A) + 1e-9)
        h_IA = h - coeff_A * J_A
        logits_IA = model.fc2(h_IA.unsqueeze(0))
        m_IA = (logits_IA[0, COLOR2ID["blue"]] - logits_IA[0, COLOR2ID["red"]]).item()
        C_A_effect = m_IA - m_normal
        perturbation_norm_A = (h - h_IA).norm().item()

        # search over random directions for the closest EFFECT-SIZE match
        # (not just norm match) at the SAME perturbation norm
        g = torch.Generator().manual_seed(seed)
        best_v = None
        best_gap = float("inf")
        best_C_R = None
        for _ in range(n_candidates):
            v = torch.randn(h.shape[0], generator=g)
            v = v / (v.norm() + 1e-9)
            # scale v so that removing its FULL self (not a projection) gives
            # the same perturbation norm as I_A -- here we mimic ablate_along_J's
            # geometry: h' = h - alpha*(v.h/v.v)*v, so we search over v's
            # direction (fixed unit norm) and rely on h's own projection onto v
            # to determine displacement, THEN rescale v post-hoc so displacement
            # norm matches exactly.
            coeff_v = (v @ h) / ((v @ v) + 1e-9)
            h_Iv_raw = h - coeff_v * v
            raw_disp_norm = (h - h_Iv_raw).norm().item()
            if raw_disp_norm < 1e-9:
                continue
            # rescale the projection so displacement norm exactly equals perturbation_norm_A
            scale = perturbation_norm_A / raw_disp_norm
            h_Iv = h - scale * coeff_v * v
            logits_Iv = model.fc2(h_Iv.unsqueeze(0))
            m_Iv = (logits_Iv[0, COLOR2ID["blue"]] - logits_Iv[0, COLOR2ID["red"]]).item()
            C_v = m_Iv - m_normal
            gap = abs(abs(C_v) - abs(C_A_effect))
            if gap < best_gap:
                best_gap = gap
                best_v = v
                best_C_R = C_v

    return {
        "m_normal": m_normal, "m_IA": m_IA, "C_A": C_A_effect,
        "perturbation_norm_A": perturbation_norm_A,
        "C_R_matched": best_C_R, "matched_effect_gap": best_gap,
        "v_R": best_v,
    }


def activation_patch_from_theta_A(model_T, model_A, object_name: str, ctx_name: str = "CTX_RED"):
    """
    Direct activation patching: replace M_T's hidden state for `object_name`
    with theta_A's OWN hidden state for the same input, then read out M_T's
    CURRENT fc2 readout on that patched activation. This is a stronger,
    more direct test than J_A-ablation: rather than removing a single
    derived direction, we substitute the entire hidden representation A
    actually produced, and ask whether M_T's current output head still
    reads it as "red" (i.e. whether the readout that now says "blue" would
    say "red" given A's own literal activations).

    If patching restores red -> the CURRENT readout is still compatible
    with A's activations (necessary condition for "A computation, run
    through, still available"). If patching does NOT restore red -> either
    the readout itself has changed to no longer respond to A's activation
    pattern, or A's own activation pattern for this input isn't the same
    kind of signal the current model would produce if actually computing A.
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)

    with torch.no_grad():
        h_A = model_A.hidden(obj_id, ctx_id).squeeze(0)  # theta_A's own activation
        h_T = model_T.hidden(obj_id, ctx_id).squeeze(0)   # M_T's own (current) activation

        logits_T_normal = model_T.fc2(h_T.unsqueeze(0))
        m_T_normal = (logits_T_normal[0, COLOR2ID["blue"]] - logits_T_normal[0, COLOR2ID["red"]]).item()

        # patch: read h_A through M_T's CURRENT fc2 head
        logits_patched = model_T.fc2(h_A.unsqueeze(0))
        m_patched = (logits_patched[0, COLOR2ID["blue"]] - logits_patched[0, COLOR2ID["red"]]).item()

        # reference: what does theta_A's OWN head say about h_A (sanity check
        # that h_A really does encode "red" under ITS OWN readout)
        logits_A_own = model_A.fc2(h_A.unsqueeze(0))
        m_A_own = (logits_A_own[0, COLOR2ID["blue"]] - logits_A_own[0, COLOR2ID["red"]]).item()

    return {
        "m_T_normal": m_T_normal,
        "m_patched": m_patched,
        "m_A_own": m_A_own,
        "patch_restores_red": bool(m_patched < 0),
        "h_A_norm": h_A.norm().item(),
        "h_T_norm": h_T.norm().item(),
        "h_A_vs_h_T_cosine": cosine_alignment(h_A, h_T),
    }


def interpolated_patch_margin(model_T, h_source: torch.Tensor, object_name: str,
                               ctx_name: str = "CTX_RED", lambdas=None):
    """
    h(lambda) = (1-lambda)*h_AB + lambda*h_source, for a sweep of lambda in
    [0,1]. Returns the red-vs-blue margin (blue-positive convention) at each
    lambda, read out through model_T's CURRENT fc2 head. h_source can be
    theta_A's own activation, a foreign theta_A's activation, a random
    activation, or a component-decomposed activation -- this function is
    agnostic to what h_source represents.
    """
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)

    with torch.no_grad():
        h_AB = model_T.hidden(obj_id, ctx_id).squeeze(0)
        curve = []
        for lam in lambdas:
            h_lam = (1 - lam) * h_AB + lam * h_source
            logits = model_T.fc2(h_lam.unsqueeze(0))
            margin = (logits[0, COLOR2ID["blue"]] - logits[0, COLOR2ID["red"]]).item()
            curve.append({"lambda": lam, "margin": margin})
    return curve


def decompose_parallel_orthogonal(h: torch.Tensor, J_A: torch.Tensor):
    """
    h = h_parallel + h_perp, where h_parallel is h's component along J_A
    (the FROZEN, theta_A-only-derived causal direction -- never refit on
    M_AB) and h_perp is the remainder. J_A is NOT normalized on input; we
    normalize internally for the projection.
    """
    J_unit = J_A / (J_A.norm() + 1e-9)
    coeff = h @ J_unit
    h_parallel = coeff * J_unit
    h_perp = h - h_parallel
    return h_parallel, h_perp


def matched_random_activation(h_reference: torch.Tensor, seed: int) -> torch.Tensor:
    """A random vector matched in norm to h_reference, used as the null
    'random matched activation' patching target."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(h_reference.shape[0], generator=g)
    v = v / (v.norm() + 1e-9)
    return v * h_reference.norm()


def direction_B_behavioral_loss_grad(model_A, object_name: str, ctx_name: str = "CTX_RED"):
    """
    Direction B: gradient of a SEPARATE loss (cross-entropy toward the 'red'
    class label, not the margin difference) with respect to h. Independent
    construction from the margin-Jacobian (direction A / J_A): this uses
    the actual classification loss gradient, a different mathematical
    object even though related in spirit.
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    h = model_A.hidden(obj_id, ctx_id)
    h = h.detach().requires_grad_(True)
    logits = model_A.fc2(h)
    target = torch.tensor([COLOR2ID["red"]], dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    grad = torch.autograd.grad(loss, h)[0].squeeze(0)
    # loss gradient points AWAY from red (direction that increases loss),
    # so the "toward red" direction is -grad
    return -grad


def direction_C_paired_difference(model_A, object_name: str, control_object_name: str,
                                    ctx_name: str = "CTX_RED"):
    """
    Direction C: difference between hidden states for a minimally paired
    A-vs-B-like comparison -- here, the difference between the SPECIAL
    object's hidden state (which computes red under this context) and the
    CONTROL object's hidden state under the OPPOSITE context (blue), a
    different kind of construction (activation difference, not a gradient
    at all) from directions A and B.
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctrl_id = torch.tensor([OBJ2ID[control_object_name]], dtype=torch.long)
    ctx_red_id = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)
    ctx_blue_id = torch.tensor([CTX2ID["CTX_BLUE"]], dtype=torch.long)
    with torch.no_grad():
        h_special_red = model_A.hidden(obj_id, ctx_red_id).squeeze(0)
        h_ctrl_blue = model_A.hidden(ctrl_id, ctx_blue_id).squeeze(0)
    return h_special_red - h_ctrl_blue


def direction_D_disjoint_inputs(model_A, ctx_name: str = "CTX_RED"):
    """
    Direction D: a direction learned on a DISJOINT set of inputs -- fit a
    linear probe (fresh, independent classifier) distinguishing "objects
    mapped to red in this context" vs "objects mapped to blue in this
    context" using ONLY filler objects (excluding zor and vex entirely),
    then evaluate that probe direction. This never looks at zor's own
    activation at all during construction.
    """
    from src.task import FILLER_OBJECTS
    filler_ids, labels = [], []
    ctx_id_val = CTX2ID[ctx_name]
    for o in FILLER_OBJECTS:
        # filler mapping is context-independent, so "would this be red
        # under CTX_RED" is just filler_mapping[o] == "red" -- but we need
        # access to filler_mapping; reconstruct via model behavior instead
        pass
    # simpler: use model_A's OWN predictions on fillers as pseudo-labels
    # (since phase A training already converged to ~100% filler accuracy,
    # this recovers the true filler mapping without needing to pass it in)
    obj_ids = torch.tensor([OBJ2ID[o] for o in FILLER_OBJECTS], dtype=torch.long)
    ctx_ids = torch.tensor([CTX2ID[ctx_name]] * len(FILLER_OBJECTS), dtype=torch.long)
    with torch.no_grad():
        h_fillers = model_A.hidden(obj_ids, ctx_ids)
        preds = model_A.fc2(h_fillers).argmax(-1)
        is_red = (preds == COLOR2ID["red"]).float()

    if is_red.sum() < 2 or (1 - is_red).sum() < 2:
        return None  # not enough class balance to fit a probe

    v = torch.zeros(h_fillers.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([v, b], lr=0.05)
    for _ in range(300):
        logits = h_fillers.detach() @ v + b
        loss = F.binary_cross_entropy_with_logits(logits, is_red)
        opt.zero_grad(); loss.backward(); opt.step()
    return v.detach().clone()


def calibrate_hidden_to_target_margin(model, object_name: str, ctx_name: str,
                                        target_class: str, ref_class: str, target_margin: float,
                                        n_steps: int = 200, lr: float = 0.1):
    """
    Adjust ONLY the model's fc2 bias terms for target_class/ref_class (a
    minimal, generic recalibration of the readout, not touching fc1/hidden
    representations at all) so that model's CURRENT margin(target_class,
    ref_class) on `object_name` equals target_margin exactly. This isolates
    "starting point on the new task" as a controlled variable, fully
    decoupled from the model's internal representation/persistence level,
    which lives entirely upstream in fc1/hidden and is untouched by this
    calibration.
    """
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    target_idx = COLOR2ID[target_class]
    ref_idx = COLOR2ID[ref_class]

    # simple bias-only calibration: adjust fc2.bias[target_idx] directly via
    # closed form (no optimization needed, single scalar target)
    with torch.no_grad():
        logits = model(obj_id, ctx_id)
        current_margin = (logits[0, target_idx] - logits[0, ref_idx]).item()
        delta = target_margin - current_margin
        model.fc2.bias[target_idx] += delta
    return model


def J_A_readout_alignment(model_t, J_A: torch.Tensor):
    """
    Tests whether J_A remains aligned with the model's CURRENT downstream
    readout direction (the red-vs-blue row difference of fc2), at any
    checkpoint t. This distinguishes:
      1. frozen A representation + changing downstream gate: J_A stays
         fixed (by construction), but W_out(t) rotates away from it ->
         alignment should DECAY over B-training.
      2. co-adapted shared substrate: J_A and W_out(t) remain aligned
         throughout (the readout continues to "listen" to J_A's direction,
         because B's own mechanism partially routes through it) ->
         alignment stays roughly CONSTANT/HIGH.
      3. genuinely preserved A pathway, causally inert to B's readout:
         would predict near-zero alignment with the CURRENT readout (B
         doesn't use this direction at all) while J_A ablation still has an
         effect via some OTHER route -- this pattern would be unusual and
         itself informative if seen.

    Returns cosine alignment between J_A (frozen, unit-normalized) and the
    CURRENT readout's red-vs-blue direction, W_red(t) - W_blue(t), from
    model_t's OWN fc2 weight matrix at this checkpoint.
    """
    with torch.no_grad():
        W = model_t.fc2.weight  # [num_classes, hidden_dim]
        readout_dir_t = W[COLOR2ID["red"]] - W[COLOR2ID["blue"]]
    return cosine_alignment(J_A, readout_dir_t)


def similarity_matched_synthetic_activation(h_target: torch.Tensor, h_reference: torch.Tensor,
                                              h_direction_source: torch.Tensor):
    """
    Construct a synthetic activation with the EXACT SAME cosine similarity to
    h_target as h_reference has, but whose orthogonal component (relative to
    h_target) is drawn from h_direction_source instead of h_reference's own
    identity. Used to test whether a cross-input transfer effect is
    explained by raw cosine similarity alone (in which case the synthetic
    should match h_reference's effect) or by something specific to
    h_reference's actual identity (in which case the synthetic should
    differ, even at matched similarity).
    """
    cos_sim = cosine_alignment(h_target, h_reference)
    h_target_unit = h_target / (h_target.norm() + 1e-9)

    dir_perp = h_direction_source - (h_direction_source @ h_target_unit) * h_target_unit
    dir_perp_norm = dir_perp.norm()
    if dir_perp_norm < 1e-9:
        return None  # direction source has no orthogonal component to work with
    dir_perp_unit = dir_perp / dir_perp_norm

    sin_sim = (max(0.0, 1 - cos_sim ** 2)) ** 0.5
    h_synthetic = cos_sim * h_reference.norm() * h_target_unit + sin_sim * h_reference.norm() * dir_perp_unit
    return h_synthetic


def surgically_destroy_component(model, object_name: str, J_A: torch.Tensor, alpha: float = 1.0,
                                   n_iters: int = 30, step_size: float = 0.3, target_P_A: float = 0.05,
                                   verbose: bool = False):
    """
    PARAMETER-SPACE (not inference-time) intervention: permanently projects
    out the component of `object_name`'s OWN embedding vector along J_A's
    direction, via an ITERATIVE, RE-LINEARIZED procedure (not a single large
    step). A single large step badly violates the local tanh-linearization
    used to map J_A from h-space into embedding space (verified empirically:
    tanh saturates over large steps, causing uncontrolled, non-monotonic
    changes to the causal effect and collateral damage to unrelated
    behavior). Instead, at each iteration we:
      1. Recompute the CURRENT local embedding-space direction e_A_unit
         (re-linearizing at the model's CURRENT hidden state, which changes
         as the embedding is edited).
      2. Take a SMALL step removing only `step_size` fraction of the
         current coefficient along e_A_unit.
      3. Check P_A (measured via the same ablation methodology used
         throughout this project); stop early if |P_A| < target_P_A.

    This directly targets P_A -> 0 as the actual stopping criterion, rather
    than trusting the linear approximation to get there in one shot.
    """
    from src.probe import ablate_along_J  # local import to avoid circularity at module load

    embed_dim = model.embed.embedding_dim
    obj_idx = OBJ2ID[object_name]
    obj_id = torch.tensor([obj_idx], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    total_removed = torch.zeros(embed_dim)

    for it in range(n_iters):
        with torch.no_grad():
            h_current = model.hidden(obj_id, ctx_id).squeeze(0)
            local_deriv = 1 - h_current ** 2
            J_A_preact = J_A * local_deriv
            W1_obj_block = model.fc1.weight[:, :embed_dim]
            e_A = W1_obj_block.T @ J_A_preact
            e_A_norm = e_A.norm()
            if e_A_norm < 1e-9:
                break
            e_A_unit = e_A / e_A_norm

            current_embed = model.embed.weight[obj_idx].clone()
            coeff = current_embed @ e_A_unit
            step = alpha * step_size * coeff * e_A_unit
            model.embed.weight[obj_idx] -= step
            total_removed += step

            logits_normal = model(obj_id, ctx_id)
            m_normal = (logits_normal[0, COLOR2ID["blue"]] - logits_normal[0, COLOR2ID["red"]]).item()
            logits_ablated = ablate_along_J(model, obj_id, ctx_id, J_A, alpha=1.0)
            m_ablated = (logits_ablated[0, COLOR2ID["blue"]] - logits_ablated[0, COLOR2ID["red"]]).item()
            P_A_now = m_ablated - m_normal

        if verbose:
            print(f"      destroy iter {it}: coeff={coeff.item():.4f}, P_A={P_A_now:.4f}")
        if abs(P_A_now) < target_P_A:
            break

    return model, total_removed.norm().item()


def surgically_transplant_component(model, object_name: str, source_embed_row: torch.Tensor,
                                      J_A: torch.Tensor, alpha: float = 1.0):
    """
    Reinsert ONLY the J_A-relevant component of a SOURCE embedding row
    (e.g. theta_A's own embedding for this object, a foreign lineage's, or
    a random/synthetic one) into the CURRENT model's embedding for
    `object_name`, leaving everything else about the current model
    (including the rest of its own embedding row) untouched. This is the
    surgical "reinsert only the historical component" step.
    """
    embed_dim = model.embed.embedding_dim
    W1_obj_block = model.fc1.weight[:, :embed_dim]
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    with torch.no_grad():
        h_current = model.hidden(obj_id, ctx_id).squeeze(0)
        local_deriv = 1 - h_current ** 2
        J_A_preact = J_A * local_deriv
        e_A = W1_obj_block.T @ J_A_preact
        e_A_unit = e_A / (e_A.norm() + 1e-9)

        obj_idx = OBJ2ID[object_name]
        source_coeff = source_embed_row @ e_A_unit
        component_to_insert = alpha * source_coeff * e_A_unit

        # remove whatever component is currently there along e_A_unit, then add the source's
        current_embed = model.embed.weight[obj_idx].clone()
        current_coeff = current_embed @ e_A_unit
        model.embed.weight[obj_idx] = model.embed.weight[obj_idx] - current_coeff * e_A_unit + component_to_insert

    return model


def decompose_shared_and_specific(J_A: torch.Tensor, J_B: torch.Tensor):
    """
    Decompose the 2D subspace spanned by {J_A, J_B} into:
      - a SHARED component: the direction each retains in common, operationalized
        as the (normalized) SUM of the two unit vectors, u_A + u_B (bisector of
        the angle between them) -- the natural "common direction" when the two
        vectors are not orthogonal.
      - an A-SPECIFIC component: the part of J_A orthogonal to the shared direction.
      - a B-SPECIFIC component: the part of J_B orthogonal to the shared direction.
    This is a natural, symmetric, non-arbitrary decomposition: if J_A == J_B,
    the shared component IS that direction and both specific components are
    zero; if J_A and J_B are orthogonal, the "shared" bisector is a genuine
    compromise direction and both specific components carry most of the norm.
    """
    u_A = J_A / (J_A.norm() + 1e-9)
    u_B = J_B / (J_B.norm() + 1e-9)
    shared_raw = u_A + u_B
    shared_norm = shared_raw.norm()
    if shared_norm < 1e-6:
        # J_A and J_B are nearly anti-parallel; shared direction is degenerate.
        # Fall back to using u_A itself as a shared reference (arbitrary but
        # documented) -- in practice this should be checked and flagged.
        shared_unit = u_A
    else:
        shared_unit = shared_raw / shared_norm

    A_specific = J_A - (J_A @ shared_unit) * shared_unit
    B_specific = J_B - (J_B @ shared_unit) * shared_unit

    return shared_unit, A_specific, B_specific


def ablate_direction_and_margin(model, object_name: str, direction: torch.Tensor,
                                  target_class: str, ref_class: str, ctx_name: str = "CTX_RED",
                                  alpha: float = 1.0):
    """Generic single-direction ablation, returns the (target-ref) margin
    change, for use with any of shared/A-specific/B-specific directions and
    any target/ref class pair (so it can measure BOTH 'A accessibility'
    i.e. red-vs-blue margin change AND 'B accessibility' i.e. blue-vs-red
    margin change with the same mechanism)."""
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    target_idx = COLOR2ID[target_class]
    ref_idx = COLOR2ID[ref_class]
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        logits_normal = model.fc2(h.unsqueeze(0))
        m_normal = (logits_normal[0, target_idx] - logits_normal[0, ref_idx]).item()

        d_norm_sq = (direction @ direction) + 1e-9
        coeff = (direction @ h) / d_norm_sq
        h_ablated = h - alpha * coeff * direction
        logits_ablated = model.fc2(h_ablated.unsqueeze(0))
        m_ablated = (logits_ablated[0, target_idx] - logits_ablated[0, ref_idx]).item()
    return m_ablated - m_normal
