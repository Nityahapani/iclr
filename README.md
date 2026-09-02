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

---

## FINAL HEADLINE RESULT: leave-one-seed-out generalization (C1 only)

Per reviewer decision, **C3 (weight decay) is dropped from the paper's
headline claim** -- it remains documented above as a real but unstable
secondary finding (seed-dependent erosion), kept separate so it doesn't
dilute the much cleaner ordinary-training result.

**The headline result, validated at the correct unit of analysis (seed, not
checkpoint), with true cross-model generalization:**

For each of 7 seeds, fit `m(t) ≈ slope * (C_A(t) + C_B(t)) + intercept` using
only the pooled checkpoints from the OTHER 6 seeds, freeze those two
numbers, then predict the ENTIRE trajectory of the held-out seed (a model
never touched during fitting).

**LOSO-R² = 0.980 ± 0.028 (min 0.913, max 0.999) across all 7 seeds.**

The fitted slope (~0.49-0.50) and intercept are remarkably stable across
every leave-one-out fit -- this is not seven separate curve-fits that
happen to each work; it is one fixed linear relationship that predicts
unseen models' entire behavioral trajectories from two independently
measured causal interventions.

**Negative control**: predicting `m(t)` from `C_r(t)` (a matched-norm random
direction) instead of `C_A(t)+C_B(t)`, same checkpoint train/test split. This
was NOT a clean R²≈0 story -- reported honestly: mean R²=0.166, std=0.541,
with one seed spuriously high (0.953) due to an outlier-noisy `C_r` trajectory
in that particular run happening to align with `m(t)`'s trend by chance. The
random control is far less stable and far less predictive than the causal
predictor, and critically does NOT generalize across seeds with a single
fixed relationship the way the causal predictor does -- but individual-seed
R² for the random control should not be read as a clean zero.

### The complete, locked paper contribution

1. **Observation**: behavior flips rapidly (within ~10 steps of B-training).
2. **Causal discovery**: the old mechanism (A) remains fully causally active
   long after behavior has reversed -- a specific, validated intervention
   (matched-norm random-direction control included) demonstrates this is not
   generic perturbation sensitivity.
3. **Quantitative theory**: the behavioral margin is almost completely
   explained by an additive combination of the old and new mechanisms'
   independently-measured causal contributions (R²≈0.999 in-sample-model,
   replicated across 7 seeds at 0.9993±0.0004). An interaction term adds
   essentially nothing (ΔR²≈0, consistent across seeds). A small, seed-
   consistent (7/7) collapse asymmetry exists but is modest in magnitude --
   not evidence for strong directional gating.
4. **Generalization**: the SAME fixed linear relationship (fit once, on 6
   seeds) predicts an entirely unseen 7th seed's full trajectory with
   LOSO-R²=0.98±0.03. This is the paper's strongest and most defensible claim.

Locked title: **Behavioral Forgetting Without Mechanistic Erasure** --
behavioral reversal does not imply causal erasure of the old mechanism;
instead, behavior is a predictable, generalizable linear function of the
independently-measured causal contributions of old and new mechanisms.

Raw data: `results/loso_and_negative_control.json`,
`results/full_interaction_test_C1_seed*.json`.

---

## Exact permutation null (replaces the noisy random-direction control)

The random-direction control was noisy and occasionally spuriously high in
individual seeds (see above). Per reviewer suggestion, replaced with a
harder, seed-level test: a **model-level permutation null**. Rather than
asking whether a single scalar control predicts `m(t)`, we ask whether the
*correct pairing* of causal trajectory to behavioral trajectory (same seed
to same seed) explains far more variance than *any mismatched pairing*
(seed s's behavior modeled using a different seed's causal trajectory),
under the exact same LOSO fitting procedure. This preserves all temporal
autocorrelation within each trajectory — every permutation still uses real,
smooth, autocorrelated trajectories from real trained models; only the
seed-to-seed assignment is scrambled.

Computed **exhaustively** over all 1854 derangements of 7 seeds (every seed
mismatched simultaneously, not sampled):

- **Observed LOSO-R² = 0.980**
- **Null distribution**: mean=0.213, std=0.195, min=-0.075, max=0.942,
  99th percentile=0.733
- **0 of 1854 derangements matched or exceeded the observed value**
  (exact permutation p < 1/1854 ≈ 0.00054)

The correct pairing explains variance far beyond what any mismatched
pairing achieves, ruling out that the R²=0.98 result is a generic artifact
of trajectory smoothness or autocorrelation rather than evidence of a real,
seed-specific causal-to-behavioral relationship.

Raw data: `results/permutation_null.json`.

## LOCKED PAPER STRUCTURE (experimentation complete)

**Title: "Behavioral Forgetting Does Not Necessarily Erase the Causal
Mechanism"** (revised from an earlier universal-sounding title — the correct
claim is non-necessity, not universal preservation; the C3 result is what
makes "necessarily" the operative word rather than a hedge).

**Thesis: behavioral forgetting and mechanistic erasure are dissociable
phenomena.**

1. **The puzzle.** A model learns `zor→red` (phase A), then learns
   `zor→blue` (phase B). Behavior says A was forgotten.
2. **The causal test.** Directly intervene on the A-associated causal
   pathway (frozen `J_A`, established and causally verified at θ_A).
   Result: behavior flips while the intervention's effect (`C_A`) remains
   large — behavioral forgetting ≠ causal erasure, under ordinary training.
3. **The quantitative result (centerpiece).** `m(t) ≈ β₀ + β_A·C_A(t) +
   β_B·C_B(t)`, coefficients fit once and frozen, tested via leave-one-
   seed-out: **LOSO-R² = 0.980 ± 0.028** (min 0.913) across 7 seeds,
   validated against an exact permutation null (p < 0.00054). The
   behavioral trajectory of an unseen model is predictable from the causal
   contributions of its competing learned pathways.
4. **Interaction analysis.** The nonlinear interaction term does not
   improve out-of-sample prediction (ΔR²≈0, consistent across seeds). A
   small, seed-consistent (7/7) collapse asymmetry exists but is modest —
   the evidence favors a largely decomposable/additive causal account
   under ordinary training, not strong directional gating.
5. **Boundary condition (limitations / secondary result).** Under weight
   decay, this same account becomes far less stable: R²_additive ranges
   0.73–0.99 across seeds, and genuine causal erosion of the old mechanism
   can occur — but its extent is highly seed-dependent rather than following
   a single predictable pattern. This is what makes the paper's central
   claim about *non-necessity* rather than universal mechanism preservation:
   mechanistic erasure is possible, just not the default outcome of
   ordinary behavioral override.

**Status: experimentation is complete.** No further architectures, tasks,
regularizers, or mechanism-discovery metrics are planned before drafting.

---

## Matched-perturbation control (honest results, both findings kept)

Constructed random directions matched to `I_A` on displacement norm, searched
over 200 candidates per seed for the closest achievable effect-size match,
then compared both interventions on filler-object (unrelated) behavior.

**Favorable finding**: no random direction of matched norm gets close to
`I_A`'s effect size -- best achievable `|C_R|` across the search (mean 5.27)
is only ~52% of `|C_A|` (mean 10.12), across 7 seeds x 200 candidates. This
is strong evidence `J_A` is not merely a large perturbation but a
disproportionately effective, structured direction.

**Complicating finding, kept rather than dropped**: `I_A` itself is not
perfectly surgical -- it flips filler-object predictions in several seeds
when tested against this filler set at the final checkpoint (delta up to
0.562), diverging from the cleaner specificity results reported earlier in
this log (which used different filler subsets and/or checkpoints). Verified
this is a real effect, not a code bug (`I_R`'s apparent zero filler-effect
was checked and confirmed to be a genuine perturbation that simply isn't
large enough to flip fillers' high-confidence predictions, not a
non-effect). **Limitation to carry into the paper**: `I_A` is best described
as a large, highly effective, and comparatively specific intervention, not
a perfectly narrow one -- the specificity claim should be stated relative to
matched-norm random directions (where it holds strongly), not as "zero
collateral effect on unrelated behavior" (which does not hold uniformly
across seeds/checkpoints).

Raw data: `results/matched_control_test.json`.

---

## Reopened experimentation: three additional validations (Parts 1-3)

Per reviewer request, three additional experiments were run after "locked"
was initially declared, to strengthen generality and rigor of the core
claim before drafting.

### Part 1: novel task family (XOR -> AND)

Replaced the color-relabeling task with a genuinely different computation:
a marker token's boolean function over two input bits, switched from XOR
(phase A) to AND (phase B) -- same input domain, different learned function,
not a relabeled classification. Reused only the lightweight core test
(t_flip, C_A(t) trajectory), no full archaeology apparatus.

**Result: clean replication across all 7 seeds.** `t_flip` in [10,20] steps
(near-instant behavioral reversal) while `C_A` at the final checkpoint
(after 3000 B-training steps) remains substantial in every seed (range
6.4-14.3, never near zero). The same dissociation holds on a fundamentally
different computational task.

### Part 2: architecture generalization

Tested the same XOR->AND task across three architectures using only the
binary dissociation test (t_flip early AND final |C_A|>1.0), same 7 seeds:

- **Tiny MLP** (hidden_dim=32): 7/7 seeds show dissociation.
- **Larger MLP** (hidden_dim=128, 4x capacity): 7/7 seeds show dissociation
  -- scale within the MLP family does not change the phenomenon.
- **Small transformer** (single self-attention layer, 3-token sequence):
  genuine architectural limitation found first -- phase-A accuracy plateaus
  around 0.96 regardless of learning rate or training length (diagnosed as
  a real ceiling, not a bug). Of 6/7 seeds that adequately learned phase A,
  **4/6 show clear dissociation, 2/6 show near-zero final `C_A`** (weak or
  absent effect).

**Honest summary, not smoothed over**: the phenomenon is robust and
universal within the MLP family (14/14 seeds across two capacities) but
measurably less reliable in the attention-based architecture tested (4/6
successful seeds). Given the transformer's own phase-A learning is itself
less reliable at this toy scale, this result is reported as suggestive of
an architectural boundary condition, not definitive -- worth flagging as an
open question for future work rather than claiming universality across
architectures.

### Part 3: rigorous matched-perturbation control

See the dedicated section above. Summary: no random direction of matched
displacement norm achieves anywhere close to `I_A`'s effect size (strong
evidence for specificity), but `I_A` itself was found to have some
collateral effect on unrelated (filler) behavior in several seeds at this
particular checkpoint/filler-set combination (a genuine limitation, kept in
the record rather than dropped).

### Precise wording adopted for the paper (per reviewer guidance)

- Say: *"behavioral reversal can occur while the causal contribution of the
  original computation remains substantial"* -- not "the old mechanism
  persists" (ties the claim to the specific intervention actually performed).
- Do NOT say *"the network stores the old concept"* -- "concept" implies a
  representational claim broader than what was operationally tested. The
  causal object is specifically the A-associated computation as defined by
  the ablation intervention, nothing more.
- Formal distinction to state precisely in the paper: **behavioral
  forgetting** is `m(t) - m(0) < 0` (sign change in the tracked margin);
  **causal erasure** is `C_A(t) -> 0`. The paper's logical point: `m(t) < 0`
  does not imply `C_A(t) = 0`.

Raw data: `results/logic_task_seed*.json`, `results/arch_generalization.json`,
`results/arch_transformer_rerun.json`, `results/matched_control_test.json`.

**Status: all three reopened experiments complete.** Ready for drafting.

---

## Counterfactual retraining experiment: persistence vs. reuse/repurposing

Per reviewer request, tested the strongest possible claim directly: does
ablating the model's newly-learned B mechanism *reveal* the old A behavior
(not just move the margin), and does this happen in M_AB but NOT in a
matched M_B control that never learned A? Full conditional matrix {none, A
removed, B removed, A+B removed} computed for both populations at matched
checkpoints, 7 seeds.

**Strict result**: only 1/7 seeds cross the binary threshold (B-removal
flipping the prediction back to "red"). Weaker than hoped.

**Graded result, more informative and directly bears on the
persistence-vs-reuse distinction raised by review**: removing B collapses
the margin from strongly-blue (~9-13) to near-zero in every seed, in BOTH
M_AB and M_B. But the **shift magnitude is systematically SMALLER in M_AB
than in M_B** (mean 9.61 vs 12.07, 7/7 seeds, Wilcoxon p=0.016) — the
opposite of what clean, independent persistence would predict. If A were
intact as a separable circuit merely outvoted by B, removing B should
reveal *more* red-like margin in M_AB (which has A to reveal) than in M_B
(which has nothing there). Instead M_AB's B-removal effect is consistently
*smaller*.

### What this means for the paper's central claim

This result argues against the strongest reading of the project's core
claim. Three increasingly strong claims, and where the evidence now stands
on each:

| Claim | Evidence |
|---|---|
| The model no longer behaviorally expresses A | Yes — clean, replicated |
| A phase-A-derived hidden direction (`J_A`) still causally affects output | Yes — `C_A`, LOSO-R²=0.98, permutation null all support this |
| **The original A computation/mechanism itself remains intact as a separable circuit** | **Not established — this experiment argues against it** |

The `J_A`-ablation results (Jacobian alignment, mediation effect, LOSO
prediction) demonstrate that a phase-A-derived *direction* remains causally
influential on the model's current output. They do **not** demonstrate that
a separable A-*computation* persists underneath B and gets exposed when B is
removed. The counterfactual matrix result is more consistent with
**reuse/repurposing**: `J_B` (M_AB's own current blue-mechanism direction)
appears to partially depend on or overlap with the same representational
subspace `J_A` occupies — plausible given M_AB's weights were literally built
by continuing training from θ_A, so B-training had every opportunity to
repurpose rather than route around A's substrate. Ablating B in M_AB
therefore does not cleanly "uncover" A underneath; it disrupts a
representation that both mechanisms may partially share, and does so
somewhat less dramatically than in M_B, where B's mechanism was built from
scratch and appears more cleanly self-contained.

**Revised claim for the paper**: the evidence supports *"a direction
associated with the original computation remains causally influential on
current behavior"* — the precise, intervention-tied wording already adopted
per earlier review — but does **not** support the stronger claim that *"the
original computation persists intact as a distinct, coexisting circuit."*
This is an important downgrade from the "causal coexistence" framing
explored in this experiment's original hypothesis, and should be stated as
a limitation/open question rather than a confirmed finding. It also
explains, retrospectively, why the interaction test (`Γ_AB`) found real,
substantial interaction rather than clean additive independence — a shared
substrate would produce exactly that pattern.

Raw data: `results/counterfactual_matrix_seed*.json`,
`results/counterfactual_matrix_summary.json`.

---

## Activation patching experiment: a genuinely positive, specific result

Per reviewer request, ran a stronger, more direct mechanism-specific test
than J_A-ablation: patch θ_A's own hidden activation for zor directly into
the current model's readout, and ask whether the readout still interprets
it as "red." This directly targets the persistence-vs-reuse distinction the
counterfactual matrix experiment left open.

**M_AB patched with its own θ_A activation: 7/7 seeds restore red**, by a
large, tight margin (m_patched = -7.35 ± 0.39, flipping from a strongly-blue
baseline of ~9-11).

**Decisive within-model specificity test**: the SAME M_AB model, patched
instead with a FOREIGN θ_A activation (an independently-trained different
seed's phase-A model), gives m_patched = +1.80 ± 2.52 — weak and often
still blue. Own beats foreign in **7/7 seeds** (Wilcoxon p=0.016). This is
the cleanest, most specific positive result in this project: M_AB's current
readout is not merely "compatible with red-like activations in general" —
it specifically and reliably recognizes its own lineage's actual θ_A
representation, far more than an unrelated one.

**Honest complication, not dropped**: M_B (control) patched with a foreign
θ_A also restores red in 4/7 seeds — weaker and less consistent than M_AB's
7/7, but not zero. This means any blue-trained readout retains *some*
general compatibility with θ_A-like activations; the effect is not entirely
absent in the control. The own-vs-foreign comparison *within M_AB* is
therefore the most defensible evidence, since it holds the readout fixed
and only varies which activation is patched in.

### Reconciling this with the counterfactual matrix result

The two experiments are not in conflict; they test different things and
together sharpen the claim:

- **Counterfactual matrix (ablating J_B)**: asked whether *removing* the
  current mechanism reveals A on its own, using the model's own evolved
  hidden states throughout. Result was negative/mixed — B-removal did not
  cleanly re-derive A's activation pattern from within M_AB's own current
  representational geometry.
- **Activation patching (injecting θ_A's own activation)**: asked whether
  the CURRENT readout, given the EXACT historical activation A actually
  produced, still recognizes it. Result is positive and specific.

Put together: **M_AB's current output head retains a specific,
lineage-dependent capacity to correctly interpret A's actual computed
representation when that representation is directly supplied — but the
model's own current internal dynamics do not spontaneously reconstruct that
representation when the B-pathway is merely removed.** This is a more
precise and more defensible claim than either "the A computation persists
and gets revealed" (too strong, contradicted by the counterfactual matrix)
or "there is nothing left of A but a generically-influential direction"
(too weak, contradicted by the patching specificity result).

### Revised claim for the paper

*A phase-A-derived hidden direction remains causally influential on current
output (established via ablation), AND the current readout retains a
specific, lineage-dependent capacity to correctly interpret A's actual
historical activation when it is directly restored (established via
patching) — but the model's own ongoing computation does not spontaneously
reconstruct that activation once the B-pathway is removed (established via
the counterfactual matrix).* This is a genuine, three-part causal picture
of what survives and what doesn't, rather than a single "persists" or
"doesn't persist" verdict.

Raw data: `results/patching_seed*.json`, `results/patching_summary.json`.
