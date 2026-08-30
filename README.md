# Neural Archaeology — Killer Experiment Log

Reproducibility log for the "does a final model retain a causal trace of an
erased binding" experiment (ICLR 2027).

## Central question

After training `A -> B` where phase A teaches `zor = red` (context-dependent)
and phase B overwrites this so `zor = blue` unconditionally, does the FINAL
model retain evidence that the red binding once existed — even though it is
behaviorally and causally erased — when compared against a `B-only` control
matched in output distribution?

## Design history (see commits for full detail)

1. **v1 (readout-direction v_A)** — FAILED. `v_A` built from `fc2`'s red row
   was causally necessary but not zor-specific: ablating it broke every
   other red-labeled filler object too, since it's a generic "produce red"
   direction the whole model shares.

2. **v2 (difference-in-differences v_A)** — FAILED differently. Subtracting
   a structurally-matched control object's context-contrast from zor's
   context-contrast left a vector that was perfectly *specific* (zero effect
   on control/fillers) but *not causally load-bearing* — nearly orthogonal
   (cos≈0.04) to the model's actual red/blue readout direction. Diagnosis:
   zor and the control object were built to share near-identical context
   computation by design, so subtracting the control removed almost all of
   the causally relevant signal along with the generic part.

   **Conclusion drawn from v1+v2 failing in complementary ways:** a
   "zor-specific, causally load-bearing, generically-orthogonal direction"
   may not exist in a network that legitimately shares computation across
   objects with the same context rule. Chasing more clever controls to force
   such a direction into existence would be representation hunting, not
   measurement.

3. **v3 (causal Jacobian alignment) — FROZEN DESIGN, current result.**
   Reframed the historical object from a vector to a *relationship*: the
   model's own local causal gradient for the red/blue decision,
   `J(o,c) = grad_h [logit_red(o,c) - logit_blue(o,c)]`, evaluated at zor.
   Historical signature: `rho_A(M) = cos(J_A^zor, J_M^zor)`, where `J_A^zor`
   is frozen at the end of phase A. Compare `rho_A(M_AB)` vs `rho_A(M_B)`
   (KL-matched control), and track `rho_A(t)` across the whole B-training
   trajectory to distinguish genuine historical retention from ordinary
   optimization inertia (which would predict smooth decay of `rho_A(t)`
   toward the B-only baseline; a fossil predicts persistence after
   behavioral erasure).

## Current single-seed pilot result (`results/pilot_run_v3.json`)

- Phase A converges to 100% eval accuracy; zor is causally read as red
  (margin +8.2).
- `M_AB` (treatment) and `M_B` (B-only control) matched to KL=0.00012 on the
  shared eval distribution; both at 100% eval accuracy, both firmly predict
  zor=blue (margins -9.8 and -12.8 respectively).
- **`rho_A(M_AB) = 0.992`, `rho_A(M_B) = 0.049`** — a 0.94 gap despite
  matched behavior.
- Trajectory: zor's behavioral prediction flips from red to blue within the
  first **10** B-training steps. `rho_A(t)` stays flat at ~0.99 for the
  remaining **~2990 steps** of B-training — it does **not** decay toward the
  B-only baseline. This is inconsistent with ordinary inertia (which would
  predict continued decay) and consistent with a genuine historical fossil.
- Sanity check: `rho_A` between `J_A^zor` and Jacobians from **10 entirely
  untrained random-init models** averages -0.018 (chance level, matching the
  B-only baseline of 0.049) — ruling out that ~0.99 alignment is a generic
  artifact of the small hidden dimension.

## What this pilot does and doesn't establish

- **Does establish:** the measurement pipeline works end-to-end (task,
  paired-lineage training, KL-matching, causal Jacobian construction,
  trajectory tracking), and produces a large, clean, specificity-checked
  effect at n=1.
- **Does NOT yet establish:** population-level reliability. This is a single
  seed. Required next step: 20-50 seeds per condition (vary init seed, phase
  A/B data order, and ideally the specific zor-analog object), paired
  statistical test on `rho_A(M_AB) - rho_A(M_B)` across seeds, and repeating
  the Stage-4 trajectory analysis per seed to check the dissociation-margin
  sign is consistent rather than a single lucky draw.

## Repo structure

- `src/task.py` — synthetic object+context -> color task, phase A/B/B-only generators.
- `src/model.py` — tiny 2-input MLP classifier (object embed + context embed -> hidden -> logits).
- `src/train.py` — training loop with checkpointing + KL-based distributional matching.
- `src/probe.py` — Jacobian construction (`jacobian_of_margin`, `jacobian_zor_red_vs_blue`), `cosine_alignment`. Older `v_A` (readout-direction, diff-in-diff) functions retained for the historical record but not used in the current pipeline.
- `src/fisher.py` — activation-space Fisher curvature (used in v1/v2 designs; not part of the current v3 pipeline, kept for reference).
- `src/run_pilot.py` — current (v3) end-to-end experiment runner.
- `results/pilot_run_v3.json` — full output of the current pilot, including per-checkpoint trajectory.

## Next steps

1. Multi-seed replication (Stage 3 gap + Stage 4 dissociation, n>=20).
2. Vary amount-of-B-training as a first-class sweep (not just log density)
   to get a proper "historical half-life" curve for `rho_A(t)`.
3. Test whether the effect holds with a *reversed* zor-analog control (an
   object whose context rule is anti-correlated with zor's, not identical),
   as an alternate specificity check independent of the Jacobian framing.
4. Scale up model/task complexity once the small-scale effect is confirmed
   robust across seeds.
