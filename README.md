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

3. **v3 (causal Jacobian alignment).** Reframed the historical object from a
   vector to a *relationship*: the model's own local causal gradient for the
   red/blue decision, `J(o,c) = grad_h [logit_red(o,c) - logit_blue(o,c)]`,
   evaluated at zor. Historical signature: `rho_A(M) = cos(J_A^zor, J_M^zor)`,
   where `J_A^zor` is frozen at the end of phase A. Single-seed result:
   `rho_A(M_AB)=0.99` vs `rho_A(M_B)=0.05`, flat across all of B-training
   despite behavior flipping in 10 steps. **This established that structural/
   geometric alignment survives — but not yet whether that geometry is still
   causally load-bearing or has become inert (a real fossil requires both:
   structure persists, function doesn't).**

4. **v4 (causal mediation test + sham-C specificity control) — CURRENT
   RESULT, and it overturns the naive "fossil" reading of v3.**

   Added two things:
   - **Mediation test `delta_A(t)`**: at each checkpoint, ablate the
     component of the model's CURRENT hidden state along the FROZEN
     `J_A^zor` direction and measure the resulting change in the red-vs-blue
     logit margin. This tests function, not just structure.
   - **Sham-C specificity control**: a second object `fenn` undergoes a
     matched "phase C" concurrently with zor's phase A (same frequency,
     steps, structure) but teaches `fenn=green` (not red) for CTX_RED. Both
     zor and fenn are later overwritten to unconditional blue in phase B.
     Tests whether `rho_A` reflects WHAT zor specifically learned, or merely
     THAT this region of parameter space was optimized earlier (a
     "when, not what" confound).

   **Result: Hypothesis B (latent persistence), not Hypothesis A (fossil).**
   `delta_A` at the final checkpoint is **-1.003 vs a normal margin of
   -11.94** (i.e. ablating `J_A^zor` swings the margin by +10.94) — a
   mediation effect actually *larger* than the theta_A reference effect
   (-8.24, measured when the red mechanism was definitely still live and
   dominant). **The old red-computation never became causally inert. It's
   still fully wired up — it's just outvoted/overridden by whatever new
   computation drives the blue prediction.** `rho_A` alone could not
   distinguish this from genuine fossilization; the mediation test was
   necessary and decisive.

   The specificity control passed cleanly on its own terms:
   `cos(J_T_zor_AB, J_C_fenn) = 0.576` (cross-object, low) vs
   `rho_A(zor's own history) = 0.99` (own-object, high) — so whatever is
   being measured is specific to *what* zor learned, not just *that* this
   parameter region trained earlier. That part of the original concern is
   resolved. The open problem is that what's specific and persistent is not
   (yet) evidence of behavioral erasure with structural-only survival.

## Where this leaves the paper's central claim

The originally hoped-for phenomenon — "behavior forgets, but geometry
remembers, and the old mechanism is causally dead" — is **not what this
single-seed pilot shows**. What it shows instead, robustly and specifically:
a model can present unified, matched, confident BLUE behavior for an object
while an entirely intact, still-causally-active RED mechanism sits
underneath it, currently overridden rather than erased. That's a real and
interesting result in its own right (arguably a cleaner, more surprising
claim than the original "fossil" framing — closer to "networks can maintain
live, suppressed alternative computations that are invisible to standard
behavioral probing" than to "networks retain inert traces of forgotten
concepts"). Whether a genuine Hypothesis-A fossil regime exists (e.g. after
much longer B-training, higher learning rates, or architectural changes that
force parameter reuse) is now the open empirical question, not a settled
premise.



## What this pilot does and doesn't establish

- **Does establish:** the measurement pipeline works end-to-end (task,
  paired-lineage training, KL-matching, causal Jacobian construction +
  mediation intervention, sham-C specificity control, trajectory tracking),
  and produces large, clean, specificity-checked effects at n=1 -- just not
  the effect originally hypothesized.
- **Does NOT yet establish:** (a) population-level reliability -- this is a
  single seed; (b) that a genuine Hypothesis-A fossil regime exists anywhere
  in this task's parameter space -- v4 found Hypothesis B (latent
  persistence) instead, at the specific training configuration tested.

## Repo structure

- `src/task.py` — synthetic object+context -> color task. Phase A/B/B-only generators for zor, plus a matched sham "phase C" for fenn (specificity control).
- `src/model.py` — tiny 2-input MLP classifier (object embed + context embed -> hidden -> logits).
- `src/train.py` — training loop with checkpointing + KL-based distributional matching.
- `src/probe.py` — Jacobian construction (`jacobian_of_margin`, `jacobian_zor_red_vs_blue`, `jacobian_fenn_green_vs_blue`), `cosine_alignment`, `causal_mediation_effect` (the delta_A test), `ablate_along_J`. Older `v_A` (readout-direction, diff-in-diff) functions retained for the historical record.
- `src/fisher.py` — activation-space Fisher curvature (used in v1/v2 designs; not part of the current pipeline, kept for reference).
- `src/run_pilot.py` — current (v4) end-to-end experiment runner.
- `results/pilot_run_v3.json`, `results/pilot_run_v4.json` — full outputs, including per-checkpoint trajectories (rho_A(t), delta_A(t), behavior(t) together in v4).

## Next steps

1. **Before scaling seeds**: sweep training configuration (longer/shorter B
   training, different learning rates, larger hidden dim, more filler
   objects competing for capacity) to check whether Hypothesis A (genuine
   fossilization) appears anywhere, or whether Hypothesis B (latent
   persistence) is the generic outcome in this task family. The paper's
   claim depends entirely on which regime is real and how it's controlled.
2. If/when a Hypothesis-A regime is found: multi-seed replication (n>=20)
   with paired statistical tests on the rho_A gap AND the delta_A
   fraction-remaining.
3. If Hypothesis B turns out to be the robust finding instead: reframe the
   paper around *that* result (suppressed-but-live alternative computations
   invisible to behavioral probing) rather than forcing a fossil narrative.
4. Either way: proper "historical half-life" sweep over amount-of-B-training
   as a first-class variable, tracking rho_A(t) and delta_A(t) jointly.
5. Scale up model/task complexity only after the small-scale phenomenon
   (whichever hypothesis wins) is confirmed robust across seeds.
