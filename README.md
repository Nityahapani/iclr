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

---

## FINAL RESULT: the preregistered factorial (n=20/condition, no thresholding)

After the exploratory sweep found no config that cleanly separated "structure
survives, function dies" from ordinary degradation, we dropped the
`rho_A>0.8, frac<0.15` fossil-detection thresholds entirely (those were
engineering criteria for a hoped-for result, not theoretically motivated —
tuning toward them would have meant optimizing the experiment to pass its
own test) and instead ran a frozen, preregistered 3-condition factorial,
reporting the **raw joint distribution** of `(rho_A, fraction_mediation_remaining)`
per condition:

- **C1 (wide, no decay)** — baseline. `rho_A_AB`: mean 0.987, **std 0.0035**
  (extremely tight across 20 seeds). `frac_remaining`: mean 0.95, std 0.11.
  **Spearman(rho_A, frac_remaining) = -0.017, p=0.94** — no relationship.
- **C2 (sparse bottleneck=2, no decay)** — capacity scarcity alone.
  `rho_A_AB`: mean 0.991, std 0.0059 — *if anything slightly higher/tighter*
  than C1. `frac_remaining`: mean 1.23 (higher than C1). **Spearman = 0.045,
  p=0.85** — still no relationship. Capacity scarcity alone does not create
  or reveal any structure/function decoupling.
- **C3 (sparse bottleneck=2 + weight_decay=0.1, longer B-training)** — the
  strongest plausible overwrite condition found during exploration, used
  as-is without further tuning. `rho_A_AB`: mean 0.458, **std 0.41, range
  [-0.42, 0.91]** — huge seed-to-seed variance, sometimes structure survives
  strongly, sometimes it's destroyed or even anti-correlated. `frac_remaining`:
  mean 0.088, std 0.056. **Spearman(rho_A, frac_remaining) = 0.93,
  p<0.0001** — under real parameter pressure, structure and function become
  *strongly, positively coupled*: whichever seeds lose causal mediation also
  lose geometric alignment, and by similar amounts. C1 vs C3 differs hugely
  on both axes (Mann-Whitney p=6.8e-08 for both).

### What this actually shows

**No evidence, anywhere searched, of a genuine fossil regime** (structure
selectively surviving while function is selectively erased). Instead:

1. **Under ordinary training (no parameter pressure), structure and function
   are essentially decoupled and both saturated**: `rho_A` sits in a very
   narrow high band (~0.98-0.99) almost regardless of capacity, while
   `frac_remaining` varies more but stays consistently well above 1 (the old
   mechanism remains fully causally active — Hypothesis B, latent
   persistence — reliably, not just as a single-seed fluke).
2. **Capacity scarcity alone (a tight bottleneck) does not induce
   fossilization** — if anything it very slightly strengthens latent
   persistence rather than weakening it.
3. **Real parameter pressure (weight decay) does erase the old mechanism**,
   but it does so by degrading structure and function *together*, in a
   strongly seed-dependent, strongly correlated way — not by selectively
   preserving geometry while killing causal relevance. This rules out the
   "structure survives, function dies" fossil hypothesis as the mechanism
   at work here, and instead supports a much more mundane picture: what
   looks like erasure under pressure is closer to *ordinary representational
   decay*, just decay that is unusually variable across random seeds at this
   particular boundary of decay strength.

### The paper-worthy question this leaves

Not "does a fossil exist" (no evidence for one in this task family) but:
**when does behavioral override (Hypothesis B: old mechanism intact,
outvoted) transition into correlated structure-function erosion (as in C3),
and why is that transition so seed-variable?** The C1→C3 comparison is a
clean, statistically solid empirical anchor for that question (both
Mann-Whitney tests p<1e-7), and the C3 internal Spearman correlation
(rho_A vs frac_remaining = 0.93) is itself a striking, reportable finding:
under parameter pressure, a network's geometric and causal traces of a
forgotten binding do not dissociate — they decay together, tightly coupled,
but unpredictably in magnitude across otherwise-identical training runs.

Raw data: `results/factorial_C1.json`, `results/factorial_C2.json`,
`results/factorial_C3_batch1.json` + `factorial_C3_batch2.json`,
`results/factorial_v1_summary.json`.

---

## Preregistered follow-up: does Q_A (mechanism isolation at theta_A) predict the C3 variance?

Per reviewer suggestion, rather than hyperparameter-searching the cause of
C3's large seed variance (rho_A in [-0.42, 0.91] at a fixed config), we
preregistered a single candidate quantity **before running it against
outcomes**:

`Q_A = 1 - mean_i[ cos(J_A^zor, J_filler_i)^2 ]`, computed at theta_A (before
any B-training) against 8 sampled filler objects' own live decision-Jacobians.
Q_A near 1 = zor's mechanism is nearly orthogonal to everything else the
model computes ("isolated"); Q_A near 0 = substantial overlap with other
live computations ("entangled"). Hypothesis: low Q_A predicts more
collateral damage under weight decay (lower rho_A(T), lower frac_remaining(T)
in C3), since decay pressure on shared directions should hit the old
mechanism harder when it isn't privately encoded.

**Result: negative.** n=20 seeds, C3 condition.
`Q_A` vs `rho_A(T)`: Spearman=0.074, p=0.76.
`Q_A` vs `frac_remaining(T)`: Spearman=0.128, p=0.59.
Q_A itself is fairly tightly clustered (range [0.64, 0.96], mean 0.82) and
carries no detectable relationship to either outcome. As specified in the
preregistration, this is reported as a clean negative rather than followed
by searching for a second metric to salvage the hypothesis. The C3 seed
variance remains unexplained; whatever determines it is not simply "how
isolated was the mechanism before interference," at least not as measured
here.

Raw data: `results/QA_C3_batch1.json`, `results/QA_C3_batch2.json`,
`results/QA_C3_test.json`.

## Status: locked

The empirical finding stands as: **behavioral override is not mechanistic
erasure; capacity limitation alone fails to induce erasure; weight-decay-style
parameter pressure induces erasure but does so via correlated (not
selective) decay of both structure and function, with substantial
seed-to-seed variance whose cause is not yet identified.** Further
speculative hyperparameter search is deliberately not pursued past this
point, per the concern about researcher degrees of freedom raised during
this project. Any future work on the source of the C3 variance should
preregister its hypothesis and quantity before looking at outcomes, as done
here.

---

## Seed-level replication of the additive/interaction result (unit of analysis = seed)

The R^2=0.999 additive-model result and the Gamma_AB/collapse-asymmetry test
were originally reported on checkpoint-level statistics (thousands of
temporally-correlated observations per seed), which inflates apparent
significance. Per reviewer correction, we re-ran the same frozen task/config
(no new hyperparameters, no new architecture) across multiple seeds and
aggregated **at the seed level** (one summary number per seed, n=7 for C1,
n=5 for C3).

**C1 (ordinary training) replicates tightly and robustly:**
- `R^2_additive` = 0.9993 ± 0.0004 across 7 seeds (range [0.9987, 1.0000]) --
  the spectacular fit is a genuine, stable property of this condition, not a
  seed-1234 fluke.
- `Delta_R^2` (interaction improvement) ~0 in all 7 seeds -- no evidence the
  interaction term helps, consistently.
- The small collapse asymmetry (`ratio_A > ratio_B`) holds in **7/7 seeds**,
  with consistent sign -- small in magnitude but directionally real and
  reproducible, unlike the earlier `wd=0.12` false positive.

**C3 (weight decay) does NOT replicate reliably -- this is itself the finding:**
- `R^2_additive` ranges from 0.73 to 0.99 across just 5 seeds (std=0.106).
  The near-perfect additive fit reported for seed 1234 was not representative;
  under real parameter pressure the additive model's fit quality is itself
  unstable across seeds, consistent with the previously-documented erratic
  erosion (rho_A ranging -0.42 to 0.91 at fixed config).
- `Delta_R^2` is correspondingly unstable, including one seed where the
  interaction model badly overfits out-of-sample (-0.50).
- Verdicts: 4/5 seeds show asymmetric collapse, 1/5 (seed 1234, the one
  originally reported) shows symmetric collapse. **The single-seed
  "symmetric collapse" verdict for C3 was not representative of the
  condition and should not be generalized.**

### Final locked claim

*Under ordinary training, behavioral updating is well and robustly explained
by an additive combination of the old and new mechanisms' independently-
measured causal contributions (R^2≈0.999, stable across seeds), with no
evidence that their combination requires a nonlinear interaction term
(ΔR^2≈0) or strong directional gating (small, seed-consistent asymmetry,
not a large effect). Under weight-decay-induced erosion, this same additive
account becomes much less reliable and highly seed-dependent -- consistent
with the earlier finding that weight decay's effect on the old mechanism is
itself erratic across random seeds rather than following a single
predictable pattern.* This is the paper's mechanistic core, and it is now
based on seed-level replication rather than a single spectacular trajectory.

Raw data: `results/full_interaction_test_C1_seed*.json`,
`results/full_interaction_test_C3_seed*.json`,
`results/seed_level_replication_summary.json`.
