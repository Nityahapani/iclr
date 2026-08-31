"""
Sweep for a Hypothesis-A (fossil) regime.

Baseline (v4) config: hidden_dim=32, phase_B_steps=3000, phase_B_lr=0.005,
weight_decay=0.0 -> Hypothesis B (latent persistence), fraction_remaining=1.33.

We sweep four axes, one at a time from the baseline (not full factorial --
too expensive for this pilot stage), at a single seed each, to see which
axis (if any) moves fraction_remaining toward 0 (fossil regime). Any
promising single-axis result gets a 3-seed mini-replication before being
reported, since single points are noisy.
"""
import json
from src.sweep_core import run_single_config

BASELINE = {
    "hidden_dim": 32, "embed_dim": 16, "ctx_embed_dim": 8,
    "phase_A_steps": 600, "phase_A_lr": 0.01,
    "phase_B_steps": 3000, "phase_B_lr": 0.005,
    "weight_decay": 0.0, "batch_size": 32,
}

SWEEP_POINTS = [
    # --- baseline reference ---
    {"name": "baseline", **BASELINE},

    # --- longer B-training ---
    {"name": "B_steps_10k", **{**BASELINE, "phase_B_steps": 10000}},
    {"name": "B_steps_30k", **{**BASELINE, "phase_B_steps": 30000}},

    # --- weight decay during B ---
    {"name": "wd_1e-3", **{**BASELINE, "weight_decay": 1e-3}},
    {"name": "wd_1e-2", **{**BASELINE, "weight_decay": 1e-2}},
    {"name": "wd_1e-1", **{**BASELINE, "weight_decay": 1e-1}},

    # --- higher learning rate for B ---
    {"name": "B_lr_0.02", **{**BASELINE, "phase_B_lr": 0.02}},
    {"name": "B_lr_0.05", **{**BASELINE, "phase_B_lr": 0.05}},

    # --- larger hidden dim (more capacity to carve a fresh subspace) ---
    {"name": "hidden_64", **{**BASELINE, "hidden_dim": 64}},
    {"name": "hidden_128", **{**BASELINE, "hidden_dim": 128}},

    # --- combined: long B-training + weight decay (most fossil-friendly guess) ---
    {"name": "combo_long_wd", **{**BASELINE, "phase_B_steps": 10000, "weight_decay": 1e-2}},

    # --- pushing weight decay further, since wd was the only axis that moved
    #     fraction_remaining toward 0 in the first pass ---
    {"name": "wd_2e-1", **{**BASELINE, "weight_decay": 2e-1}},
    {"name": "wd_3e-1", **{**BASELINE, "weight_decay": 3e-1}},
    {"name": "wd_5e-1", **{**BASELINE, "weight_decay": 5e-1}},
    {"name": "wd_2e-1_long", **{**BASELINE, "phase_B_steps": 10000, "weight_decay": 2e-1}},

    # --- fine scan of the promising 0.1-0.2 window (0.1 was AMBIGUOUS at
    #     frac=0.173, close to the fossil threshold; 0.2 broke erasure) ---
    {"name": "wd_1.2e-1", **{**BASELINE, "weight_decay": 0.12}},
    {"name": "wd_1.4e-1", **{**BASELINE, "weight_decay": 0.14}},
    {"name": "wd_1.6e-1", **{**BASELINE, "weight_decay": 0.16}},
    {"name": "wd_1.8e-1", **{**BASELINE, "weight_decay": 0.18}},
    {"name": "wd_1e-1_longB", **{**BASELINE, "weight_decay": 0.1, "phase_B_steps": 8000}},
]


def run_sweep(seed=1234):
    results = []
    for point in SWEEP_POINTS:
        name = point.pop("name")
        print(f"\n=== Running: {name} ===")
        cfg = {k: v for k, v in point.items()}
        r = run_single_config(cfg, seed=seed, verbose=True)
        r["sweep_name"] = name
        results.append(r)
        status = r.get("status")
        if status == "OK":
            print(f"  -> {name}: rho_A_gap={r['rho_A_gap']:.3f}  "
                  f"frac_remaining={r['fraction_mediation_remaining']:.3f}  "
                  f"hypothesis={r['hypothesis']}")
        else:
            print(f"  -> {name}: {status}")
        point["name"] = name  # restore for logging

    with open("/home/claude/iclr/results/sweep_v1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n=== SWEEP SUMMARY ===")
    print(f"{'name':20s} {'status':15s} {'rho_gap':>10s} {'frac_rem':>10s} {'hypothesis':>20s}")
    for r in results:
        if r.get("status") == "OK":
            print(f"{r['sweep_name']:20s} {'OK':15s} {r['rho_A_gap']:10.3f} "
                  f"{r['fraction_mediation_remaining']:10.3f} {r['hypothesis']:>20s}")
        else:
            print(f"{r['sweep_name']:20s} {r.get('status',''):15s} {'--':>10s} {'--':>10s} {'--':>20s}")

    return results


if __name__ == "__main__":
    run_sweep()
