"""
Independently-identified A- and B-relevant subspaces, tested for
separability. Directly attacks the shared-substrate interpretation by
constructing J_A and J_B from GENUINELY INDEPENDENT DATA (not both read off
the same target object at the same checkpoint, which produced degenerate
anti-parallel geometry in the earlier shared-component-matrix attempt).

J_A: averaged margin-gradient over the TRAIN pool of context-dependent
     objects at theta_A -- frozen, never touches phase B.
J_B_indep: a linear probe fit on FILLER object activations at theta_T,
     discriminating filler objects' own fixed blue-mapped vs non-blue
     identity. Never looks at any pool object.

Test: on a HELD-OUT pool object, perform A-only, B-only, and A+B ablations
and compare combined effect against additivity, plus conditional effects
(does removing one change the other's own subsequent effect).
"""
import json
import copy
import numpy as np
import torch
import torch.nn.functional as Fnn

from src.task_multiinstance import (make_filler_mapping, MultiInstanceDataset, OBJ2ID, CTX2ID,
                                      COLOR2ID, TRAIN_POOL, HELDOUT_POOL, FILLER_OBJECTS,
                                      VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.run_heldout_generalization import MIModel, train_generic, build_averaged_J_A


def fit_filler_blue_probe(model, filler_mapping, ctx_name="CTX_RED", epochs=300, lr=0.05):
    obj_ids = torch.tensor([OBJ2ID[o] for o in FILLER_OBJECTS], dtype=torch.long)
    ctx_ids = torch.tensor([CTX2ID[ctx_name]] * len(FILLER_OBJECTS), dtype=torch.long)
    with torch.no_grad():
        h = model.hidden(obj_ids, ctx_ids)
    is_blue = torch.tensor([1.0 if filler_mapping[o] == "blue" else 0.0 for o in FILLER_OBJECTS])
    if is_blue.sum() < 2 or (1 - is_blue).sum() < 2:
        return None

    v = torch.zeros(h.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([v, b], lr=lr)
    for _ in range(epochs):
        logits = h.detach() @ v + b
        loss = Fnn.binary_cross_entropy_with_logits(logits, is_blue)
        opt.zero_grad(); loss.backward(); opt.step()
    return v.detach().clone()


def ablate_margin(model, object_name, direction, target_class, ref_class, ctx_name="CTX_RED", alpha=1.0):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    target_idx, ref_idx = COLOR2ID[target_class], COLOR2ID[ref_class]
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        logits_normal = model.fc2(h.unsqueeze(0))
        m_normal = (logits_normal[0, target_idx] - logits_normal[0, ref_idx]).item()
        coeff = (direction @ h) / ((direction @ direction) + 1e-9)
        h_ablated = h - alpha * coeff * direction
        logits_ablated = model.fc2(h_ablated.unsqueeze(0))
        m_ablated = (logits_ablated[0, target_idx] - logits_ablated[0, ref_idx]).item()
    return m_normal, m_ablated


def double_ablate_margin(model, object_name, dir_A, dir_B, target_class, ref_class, ctx_name="CTX_RED", alpha=1.0):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    target_idx, ref_idx = COLOR2ID[target_class], COLOR2ID[ref_class]
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        coeff_A = (dir_A @ h) / ((dir_A @ dir_A) + 1e-9)
        coeff_B = (dir_B @ h) / ((dir_B @ dir_B) + 1e-9)
        h_ablated = h - alpha * coeff_A * dir_A - alpha * coeff_B * dir_B
        logits_ablated = model.fc2(h_ablated.unsqueeze(0))
        return (logits_ablated[0, target_idx] - logits_ablated[0, ref_idx]).item()


def conditional_effect(model, object_name, dir_first, dir_second, target_class, ref_class,
                        ctx_name="CTX_RED", alpha=1.0):
    obj_id = torch.tensor([OBJ2ID[object_name]], dtype=torch.long)
    ctx_id = torch.tensor([CTX2ID[ctx_name]], dtype=torch.long)
    target_idx, ref_idx = COLOR2ID[target_class], COLOR2ID[ref_class]
    with torch.no_grad():
        h = model.hidden(obj_id, ctx_id).squeeze(0)
        coeff_first = (dir_first @ h) / ((dir_first @ dir_first) + 1e-9)
        h_first = h - alpha * coeff_first * dir_first
        logits_first = model.fc2(h_first.unsqueeze(0))
        m_first = (logits_first[0, target_idx] - logits_first[0, ref_idx]).item()

        coeff_second_given_first = (dir_second @ h_first) / ((dir_second @ dir_second) + 1e-9)
        h_both = h_first - alpha * coeff_second_given_first * dir_second
        logits_both = model.fc2(h_both.unsqueeze(0))
        m_both = (logits_both[0, target_idx] - logits_both[0, ref_idx]).item()
    return m_first - m_both


def run_independent_subspace_test(seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = MultiInstanceDataset(filler_mapping, phase="A")
    ds_B = MultiInstanceDataset(filler_mapping, phase="B")

    model_A = MIModel()
    train_generic(model_A, ds_A, steps=1500, batch_size=64, lr=0.01, seed=seed)
    J_A = build_averaged_J_A(model_A, TRAIN_POOL)

    theta_A_state = copy.deepcopy(model_A.state_dict())
    model_AB = MIModel()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    train_generic(model_AB, ds_B, steps=3000, batch_size=64, lr=0.005, seed=seed + 1)

    J_B_indep = fit_filler_blue_probe(model_AB, filler_mapping)
    if J_B_indep is None:
        print(f"[{run_name}] SKIPPED: insufficient filler class balance")
        return None

    cos_AB = (J_A / J_A.norm()) @ (J_B_indep / J_B_indep.norm())
    print(f"[{run_name}] cos(J_A, J_B_indep) = {cos_AB.item():.4f}")

    test_obj = HELDOUT_POOL[0]

    m_normal, m_A_only = ablate_margin(model_AB, test_obj, J_A, "red", "blue")
    _, m_B_only = ablate_margin(model_AB, test_obj, J_B_indep, "red", "blue")
    m_A_plus_B = double_ablate_margin(model_AB, test_obj, J_A, J_B_indep, "red", "blue")

    Delta_A = m_A_only - m_normal
    Delta_B = m_B_only - m_normal
    Delta_AB = m_A_plus_B - m_normal
    interference = Delta_AB - Delta_A - Delta_B

    C_A_given_B = conditional_effect(model_AB, test_obj, J_B_indep, J_A, "red", "blue")
    C_A_alone = m_normal - m_A_only
    C_B_given_A = conditional_effect(model_AB, test_obj, J_A, J_B_indep, "red", "blue")
    C_B_alone = m_normal - m_B_only

    print(f"[{run_name}] m_normal={m_normal:.3f}, Delta_A={Delta_A:.3f}, Delta_B={Delta_B:.3f}, "
          f"Delta_A+B={Delta_AB:.3f}, interference={interference:.3f}")
    print(f"[{run_name}] C_A alone={C_A_alone:.3f} vs C_A|B_removed={C_A_given_B:.3f} "
          f"(ratio={C_A_given_B/(C_A_alone+1e-9):.3f})")
    print(f"[{run_name}] C_B alone={C_B_alone:.3f} vs C_B|A_removed={C_B_given_A:.3f} "
          f"(ratio={C_B_given_A/(C_B_alone+1e-9):.3f})")

    result = {
        "run_name": run_name, "seed": seed, "test_object": test_obj,
        "cos_J_A_J_B_indep": cos_AB.item(),
        "m_normal": m_normal, "Delta_A": Delta_A, "Delta_B": Delta_B, "Delta_AB": Delta_AB,
        "interference": interference,
        "C_A_alone": C_A_alone, "C_A_given_B_removed": C_A_given_B,
        "C_B_alone": C_B_alone, "C_B_given_A_removed": C_B_given_A,
    }
    with open(f"/home/claude/iclr/results/independent_subspace_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        r = run_independent_subspace_test(s, run_name=f"seed{s}")
        if r is not None:
            results.append(r)

    print("\n=== SUMMARY ===")
    cos_vals = np.array([r["cos_J_A_J_B_indep"] for r in results])
    interference_vals = np.array([r["interference"] for r in results])
    ratio_A = np.array([r["C_A_given_B_removed"] / (r["C_A_alone"] + 1e-9) for r in results])
    ratio_B = np.array([r["C_B_given_A_removed"] / (r["C_B_alone"] + 1e-9) for r in results])
    print(f"cos(J_A, J_B_indep): mean={cos_vals.mean():.4f} std={cos_vals.std(ddof=1):.4f}")
    print(f"interference (Delta_AB - Delta_A - Delta_B): mean={interference_vals.mean():.4f} "
          f"std={interference_vals.std(ddof=1):.4f}")
    print(f"ratio C_A|B_removed / C_A_alone: mean={ratio_A.mean():.4f} std={ratio_A.std(ddof=1):.4f}")
    print(f"ratio C_B|A_removed / C_B_alone: mean={ratio_B.mean():.4f} std={ratio_B.std(ddof=1):.4f}")

    with open("/home/claude/iclr/results/independent_subspace_summary.json", "w") as f:
        json.dump({
            "cos_mean": float(cos_vals.mean()), "cos_std": float(cos_vals.std(ddof=1)),
            "interference_mean": float(interference_vals.mean()), "interference_std": float(interference_vals.std(ddof=1)),
            "ratio_A_mean": float(ratio_A.mean()), "ratio_A_std": float(ratio_A.std(ddof=1)),
            "ratio_B_mean": float(ratio_B.mean()), "ratio_B_std": float(ratio_B.std(ddof=1)),
        }, f, indent=2)
