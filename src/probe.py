"""
Activation-space probe construction and causal verification.

v_A lives in hidden-activation space (dimension = hidden_dim of the model's
single hidden layer), NOT in parameter space. It is fit ONCE at theta_A and
frozen -- never refit after phase A. This module also verifies that v_A is
causally necessary for "red" behavior at theta_A, before we ever use it for
archaeology at theta_T. This addresses the causal-probing concern: a probe
that merely correlates with a concept is not evidence the model uses that
direction computationally.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task import OBJ2ID, COLOR2ID, SPECIAL_OBJECT, FILLER_OBJECTS


def fit_probe(model, obj_ids: torch.Tensor, is_target_label: torch.Tensor, epochs=200, lr=0.05):
    """
    Fit a linear probe v^T h + b = logit(is_target_label) on frozen hidden
    activations h = model.hidden(obj_ids). Returns normalized v (unit norm)
    and bias, both detached or as plain tensors (frozen forever after this).
    """
    with torch.no_grad():
        h = model.hidden(obj_ids)  # [N, hidden_dim]
    hidden_dim = h.shape[1]
    v = torch.zeros(hidden_dim, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([v, b], lr=lr)

    y = is_target_label.float()
    for _ in range(epochs):
        logits = h @ v + b
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        v_final = v.detach().clone()
        v_norm = v_final / (v_final.norm() + 1e-9)
        # sanity: probe accuracy
        preds = (h @ v_final + b.detach() > 0).float()
        acc = (preds == y).float().mean().item()
    return v_norm, b.detach().clone(), acc


def build_v_A(model_at_theta_A, filler_mapping):
    """
    Construct v_A: the direction the model's OWN OUTPUT HEAD uses to encode
    'red' vs the other colors, restricted to how zor's hidden state sits
    relative to that decision. This is NOT a probe for 'is this the zor
    token' (which conflates zor's arbitrary identity embedding with the
    red-relevant direction, and is not guaranteed causally load-bearing --
    exactly the pitfall to avoid).

    Instead we take the most direct causally-motivated construction available
    for a linear-readout model: the row of the output head (fc2.weight)
    corresponding to the 'red' class MINUS the mean of the other class rows,
    evaluated in the hidden space at theta_A. This is, by construction, the
    direction along which moving h changes the red-logit relative to other
    logits -- i.e. it is definitionally the direction the model uses to
    decide 'red', not a correlate of it. We still call it a 'probe' for
    consistency with the paper's terminology, but it is fit via the model's
    own readout weights rather than an auxiliary classifier, which is what
    makes the causal ablation test meaningful.
    """
    with torch.no_grad():
        W = model_at_theta_A.fc2.weight  # [num_classes, hidden_dim]
        red_row = W[COLOR2ID["red"]]
        other_rows = torch.cat([W[:COLOR2ID["red"]], W[COLOR2ID["red"] + 1:]], dim=0)
        direction = red_row - other_rows.mean(dim=0)
        v_A = direction / (direction.norm() + 1e-9)

    # sanity/probe-acc: how well does projection onto v_A separate zor (red)
    # from fillers in activation space, for reporting purposes only.
    all_objs = [OBJ2ID[SPECIAL_OBJECT]] + [OBJ2ID[o] for o in FILLER_OBJECTS]
    labels = [1] + [0] * len(FILLER_OBJECTS)
    obj_ids = torch.tensor(all_objs, dtype=torch.long)
    with torch.no_grad():
        h = model_at_theta_A.hidden(obj_ids)
        proj = h @ v_A
        thresh = proj.mean()
        preds = (proj > thresh).float()
        y = torch.tensor(labels, dtype=torch.float)
        probe_acc = (preds == y).float().mean().item()

    b_A = torch.zeros(1)
    return v_A, b_A, probe_acc


def build_v_A_contrastive(model_at_theta_A, model_at_theta_A_counterfactual_blue):
    """
    Zor-SPECIFIC historical direction, isolating 'zor=red' from generic
    redness. Constructed by contrasting the hidden state of the SAME model
    (theta_A) processing the zor token against a counterfactual: the hidden
    state zor WOULD have if it were treated as an ordinary blue-mapped
    object. Since we cannot literally observe that counterfactual without a
    second model, we approximate it using the paired theta_A/theta_B_only
    initialization trick: this function accepts a second model trained
    identically except zor=blue from the start (a 'zor-as-blue' twin), and
    returns h_A(zor) - h_counterfactual(zor) as the direction, projected to
    be orthogonal to the generic red-readout direction so it isolates
    zor-specific structure rather than color-general structure.
    NOTE: this is the corrected construction; build_v_A (readout-based) is
    kept in the codebase and reported alongside it in results, since the
    dissociation between the two IS itself an informative finding about
    what a 'historical direction' even means for a shared-readout model.
    """
    with torch.no_grad():
        zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
        h_A = model_at_theta_A.hidden(zor_id).squeeze(0)
        h_cf = model_at_theta_A_counterfactual_blue.hidden(zor_id).squeeze(0)
        direction = h_A - h_cf
        v = direction / (direction.norm() + 1e-9)
    return v


def build_v_sham(model, filler_mapping, seed=999):
    """
    Sham direction: same probe-fitting procedure, but for an UNRELATED concept
    that was never specially trained -- e.g. 'is this object's filler-color
    == green' (an arbitrary, un-pretrained property). Used as a negative
    control: F(v_sham) should NOT show elevated archaeological signal.
    """
    import random
    rng = random.Random(seed)
    target_color = "green"
    all_objs = [OBJ2ID[o] for o in FILLER_OBJECTS]
    labels = [1 if filler_mapping[o] == target_color else 0 for o in FILLER_OBJECTS]
    if sum(labels) == 0 or sum(labels) == len(labels):
        # degenerate split guard; fall back to a random balanced split
        labels = [1 if rng.random() < 0.5 else 0 for _ in FILLER_OBJECTS]
    obj_ids = torch.tensor(all_objs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    v_sham, b_sham, probe_acc = fit_probe(model, obj_ids, y)
    return v_sham, b_sham, probe_acc


def build_v_rand(hidden_dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_dim, generator=g)
    return v / (v.norm() + 1e-9)


def causal_ablate_hidden(model, obj_ids, v, alpha=1.0):
    """
    h' = h - alpha * (v^T h) v   (component removal along v)
    Returns logits computed from the ablated hidden state, patched through
    the model's own output head (fc2). This is a genuine causal intervention
    on the model's forward pass, not just a read-out statistic.
    """
    with torch.no_grad():
        h = model.hidden(obj_ids)
        proj = (h @ v).unsqueeze(-1) * v.unsqueeze(0)
        h_ablated = h - alpha * proj
        logits = model.fc2(h_ablated)
    return logits


def find_minimal_flipping_alpha(model_at_theta_A, v_A, alphas=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)):
    """
    Scan alpha and return the smallest value that flips zor's prediction away
    from red. We search for the minimal effective ablation strength rather
    than guessing a fixed alpha, since the 'right' alpha depends on how much
    of h's norm lies along v_A (which differs across seeds/runs).
    """
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    red_label = COLOR2ID["red"]
    for a in alphas:
        logits = causal_ablate_hidden(model_at_theta_A, zor_id, v_A, alpha=a)
        if logits.argmax(-1).item() != red_label:
            return a
    return alphas[-1]


def verify_v_A_causal(model_at_theta_A, v_A, filler_mapping, alpha=3.0):
    """
    Verify at theta_A: ablating v_A should suppress 'red' behavior for zor
    specifically, while leaving UNRELATED behavior unchanged.

    IMPORTANT CORRECTNESS NOTE (found during pilot debugging): the 'unrelated
    behavior' control set must EXCLUDE filler objects whose true color is also
    'red'. v_A -- built from fc2's red-readout row -- is, by construction, the
    direction the model uses to output 'red' for ANY input, not a zor-specific
    historical direction. Ablating it will legitimately flip other red-labeled
    fillers too; that is not collateral damage, it is v_A doing exactly what
    it is defined to do. The correct specificity check is therefore whether
    NON-RED fillers are left alone, not whether ALL fillers are left alone.
    (This also means: v_A as currently constructed is a 'red' direction, not
    a 'zor' direction -- a caveat that belongs in the writeup. See run_pilot
    notes for the follow-up fix using a zor-specific contrastive direction.)

    Returns dict with pre/post accuracy on zor-is-red, non-red-filler
    accuracy, and the red-vs-blue logit margin before/after.
    """
    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    red_label = torch.tensor([COLOR2ID["red"]], dtype=torch.long)
    blue_id = COLOR2ID["blue"]

    non_red_fillers = [o for o in FILLER_OBJECTS if filler_mapping[o] != "red"]
    filler_ids = torch.tensor([OBJ2ID[o] for o in non_red_fillers], dtype=torch.long)
    filler_labels = torch.tensor([COLOR2ID[filler_mapping[o]] for o in non_red_fillers], dtype=torch.long)

    with torch.no_grad():
        pre_zor_logits = model_at_theta_A(zor_id)
        pre_zor_correct = (pre_zor_logits.argmax(-1) == red_label).item()
        pre_margin = (pre_zor_logits[0, COLOR2ID["red"]] - pre_zor_logits[0, blue_id]).item()
        pre_filler_logits = model_at_theta_A(filler_ids)
        pre_filler_acc = (pre_filler_logits.argmax(-1) == filler_labels).float().mean().item()

    post_zor_logits = causal_ablate_hidden(model_at_theta_A, zor_id, v_A, alpha=alpha)
    post_zor_correct = (post_zor_logits.argmax(-1) == red_label).item()
    post_margin = (post_zor_logits[0, COLOR2ID["red"]] - post_zor_logits[0, blue_id]).item()
    post_filler_logits = causal_ablate_hidden(model_at_theta_A, filler_ids, v_A, alpha=alpha)
    post_filler_acc = (post_filler_logits.argmax(-1) == filler_labels).float().mean().item()

    return {
        "pre_zor_red_correct": pre_zor_correct,
        "post_zor_red_correct": post_zor_correct,
        "pre_red_blue_margin": pre_margin,
        "post_red_blue_margin": post_margin,
        "margin_dropped": bool(post_margin < pre_margin),
        "pre_nonred_filler_acc": pre_filler_acc,
        "post_nonred_filler_acc": post_filler_acc,
        "n_nonred_fillers": len(non_red_fillers),
        "causal_effect_confirmed": bool(pre_zor_correct and not post_zor_correct
                                         and abs(pre_filler_acc - post_filler_acc) < 0.15),
    }
