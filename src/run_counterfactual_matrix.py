"""
Counterfactual retraining experiment. At a matched checkpoint (behavior
already flipped to B in both populations), measure the FULL conditional
behavior matrix under {none, A removed, B removed, A+B removed} for BOTH:
  - M_AB (treatment: actually underwent phase A before B)
  - M_B  (control: trained on B only, from scratch, matched in output
          distribution)

The decisive comparison: M_AB - I_B should reveal A-like behavior, while
M_B - I_B should NOT (since M_B never learned A -- there is no A-computation
to reveal).
"""
import json
import copy
import numpy as np
import torch

from src.task import (make_filler_mapping, PhaseDataset, OBJ2ID, CTX2ID, COLOR2ID,
                       SPECIAL_OBJECT, VOCAB_SIZE, NUM_CLASSES, CONTEXT_VOCAB_SIZE)
from src.model import TinyClassifier
from src.train import train_phase, find_matched_checkpoint
from src.probe import jacobian_zor_red_vs_blue, find_B_mechanism_direction, double_intervention_margin


def new_model(bottleneck_dim=None, hidden_dim=32, embed_dim=16, ctx_embed_dim=8):
    return TinyClassifier(VOCAB_SIZE, CONTEXT_VOCAB_SIZE, NUM_CLASSES,
                           embed_dim=embed_dim, ctx_embed_dim=ctx_embed_dim,
                           hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


def zor_pred_label(margin_blue_minus_red):
    return "blue" if margin_blue_minus_red > 0 else "red"


def run_counterfactual_matrix(seed: int, run_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    filler_mapping = make_filler_mapping(seed=seed)
    ds_A = PhaseDataset(filler_mapping, phase="A")
    ds_B = PhaseDataset(filler_mapping, phase="B")
    ds_B_only = PhaseDataset(filler_mapping, phase="B_only")

    model_A = new_model()
    train_phase(model_A, ds_A, steps=600, batch_size=32, lr=0.01, seed=seed, eval_every=600)
    J_A = jacobian_zor_red_vs_blue(model_A, ctx_name="CTX_RED")

    theta_A_state = copy.deepcopy(model_A.state_dict())

    model_AB = new_model()
    model_AB.load_state_dict(copy.deepcopy(theta_A_state))
    log_AB = train_phase(model_AB, ds_B, steps=3000, batch_size=32, lr=0.005, seed=seed + 1, eval_every=20)

    model_B = new_model()
    torch.manual_seed(seed + 2)
    log_B = train_phase(model_B, ds_B_only, steps=3000, batch_size=32, lr=0.005, seed=seed + 3, eval_every=20)

    matched_entry, matched_kl = find_matched_checkpoint(log_B, log_AB)
    model_AB_T = new_model()
    model_AB_T.load_state_dict(matched_entry["state_dict"])
    model_B_T = new_model()
    model_B_T.load_state_dict(log_B[-1]["state_dict"])

    print(f"[{run_name}] matched KL={matched_kl:.6f}, "
          f"M_AB acc={matched_entry['eval_acc']:.3f}, M_B acc={log_B[-1]['eval_acc']:.3f}")

    zor_id = torch.tensor([OBJ2ID[SPECIAL_OBJECT]], dtype=torch.long)
    ctx_red = torch.tensor([CTX2ID["CTX_RED"]], dtype=torch.long)

    def full_conditional_matrix(model, J_A_dir, model_label):
        J_B_dir = find_B_mechanism_direction(model, ctx_name="CTX_RED")

        def blue_margin_via(do_A, do_B):
            rb_margin = double_intervention_margin(model, zor_id, ctx_red, J_A_dir, J_B_dir,
                                                     alpha=1.0, do_A=do_A, do_B=do_B)
            return -rb_margin

        m_none = blue_margin_via(False, False)
        m_A_removed = blue_margin_via(True, False)
        m_B_removed = blue_margin_via(False, True)
        m_AB_removed = blue_margin_via(True, True)

        matrix = {
            "none": {"margin": m_none, "pred": zor_pred_label(m_none)},
            "A_removed": {"margin": m_A_removed, "pred": zor_pred_label(m_A_removed)},
            "B_removed": {"margin": m_B_removed, "pred": zor_pred_label(m_B_removed)},
            "AB_removed": {"margin": m_AB_removed, "pred": zor_pred_label(m_AB_removed)},
        }
        print(f"[{run_name}] {model_label} conditional matrix:")
        for cond, v in matrix.items():
            print(f"    {cond:12s}: margin={v['margin']:8.3f}  pred={v['pred']}")
        return matrix

    matrix_AB = full_conditional_matrix(model_AB_T, J_A, "M_AB (treatment)")
    matrix_B = full_conditional_matrix(model_B_T, J_A, "M_B (control)")

    AB_B_removed_is_A = (matrix_AB["B_removed"]["pred"] == "red")
    B_B_removed_is_A = (matrix_B["B_removed"]["pred"] == "red")

    print(f"\n[{run_name}] DECISIVE TEST:")
    print(f"  M_AB with B removed -> {matrix_AB['B_removed']['pred']} "
          f"(reveals A-like 'red' behavior: {AB_B_removed_is_A})")
    print(f"  M_B  with B removed -> {matrix_B['B_removed']['pred']} "
          f"(reveals A-like 'red' behavior: {B_B_removed_is_A})")

    coexistence_demonstrated = AB_B_removed_is_A and not B_B_removed_is_A
    print(f"  CAUSAL COEXISTENCE demonstrated (M_AB reveals A, M_B does not): {coexistence_demonstrated}")

    result = {
        "run_name": run_name, "seed": seed,
        "matched_kl": matched_kl,
        "matrix_AB": matrix_AB, "matrix_B": matrix_B,
        "AB_B_removed_is_A": AB_B_removed_is_A, "B_B_removed_is_A": B_B_removed_is_A,
        "coexistence_demonstrated": coexistence_demonstrated,
    }
    with open(f"/home/claude/iclr/results/counterfactual_matrix_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240]
    results = []
    for s in SEEDS:
        print("=" * 60)
        results.append(run_counterfactual_matrix(s, run_name=f"seed{s}"))
        print()

    n_coexist = sum(r["coexistence_demonstrated"] for r in results)
    print(f"\n=== SUMMARY: {n_coexist}/{len(results)} seeds demonstrate causal coexistence ===")
    with open("/home/claude/iclr/results/counterfactual_matrix_summary.json", "w") as f:
        json.dump({"n_coexist": n_coexist, "n_total": len(results),
                    "per_seed": [{"seed": r["seed"], "coexistence": r["coexistence_demonstrated"]} for r in results]},
                   f, indent=2)
