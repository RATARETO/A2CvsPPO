# A2C / PPO: stability and hyperparameter sensitivity

Loaded runs: 1620. Separate condition strata: 21.
Excluded runs: 0; incomplete: 0.
Missing entries in saved execution plans: 0.
Compatible configuration/metric comparisons: 484.
Frozen study coverage: 1620/1620; missing or invalid: 0; incompatible runtime excluded: 0; supplementary runs excluded: 175.

See audit.csv for exclusions and analysis.json for exact comparison conditions.
Only completed runs enter statistics. Missing/failed planned runs are not treated as
successful observations: resolve audit/failures and check the saved sweep plan before publication.
Legacy results are descriptive only and never directly compared with the versioned protocol.
Different phase, evaluation settings, versions or training identity form separate strata.
Declared experimental factors vary within a series; other common settings stay controlled.
Full source hashes remain provenance. Analysis-only edits do not invalidate trained models.

The independent sampling unit is a training run, not an evaluation episode/checkpoint.
Mean CIs are percentile bootstrap 95% intervals over training runs (2000 resamples,
seed 2026). No CI or sample standard deviation is supplied for n=1. Small n remains uncertain.
Success-rate intervals use Wilson's method. Curve bands are pointwise, not simultaneous.
Differences are A2C minus PPO using independently resampled runs. Intervals are descriptive,
without adjustment for multiple comparisons; they do not establish general superiority.

AUC_return = trapezoidal integral of sampled mean return over actual steps / budget.
AUC normalizes that value using the environment-specific declared return bounds.
CartPole retains its original integral / (500 * budget) definition exactly.
Return bounds are task-local display scales, not equal task difficulty or universal benchmarks.
Pendulum's upper bound zero does not imply an attainable optimum from random initial states.
Do not pool raw rewards or normalized scores across environments into an overall ranking here.
last_window_mean averages the last five curve points; final_eval_sampled_mean uses a
separate fixed set of evaluation episodes. They are different endpoints.
Solved = first sequence of sustain_evals consecutive mean returns >= threshold.
solved is retained for backward compatibility and equals ever_reached, not permanent success.
retained_at_end requires the last sustain_evals curve points to meet the threshold.
collapsed_after_success indicates an observed below-threshold evaluation after confirmation;
post_confirmation_checkpoints records the available follow-up. final_above_threshold uses
the independent final evaluation and is a different endpoint. Fractions receive Wilson CIs.
confirmation_steps is its last checkpoint. Non-solvers are right-censored at the budget.
restricted_confirmation_steps replaces censored times with the budget (a restricted endpoint,
not an estimate of eventual convergence time). Post-solve metrics are conditional on success;
their *_n columns report the smaller denominator. A last-checkpoint confirmation leaves no
subsequent follow-up; interpret these conditional metrics together with the full curves.
max_drawdown is the largest peak-to-later-trough drop. temporal_rms_change measures adjacent
checkpoint changes, including useful learning improvements; it is not a standalone failure score.

Sensitivity is max-minus-min mean AUC within the declared OFAT range, and is reported only
for balanced seed sets. Its bootstrap resamples training seeds jointly across parameter values.
It depends on range width and grid density; do not rank algorithms across unequal ranges.
sensitivity_comparison.csv explicitly checks matching ranges and linked controls before
comparing the sizes of A2C/PPO sensitivity ranges. Native algorithm settings remain different.
For n_envs, rollout size stays constant within each algorithm's series (A2C 64, PPO 2048).
These rollout sizes differ across algorithms, so this series is an algorithm-specific ablation.
Grid cells are recorded in grid_cells.csv, including missing cells. interactions.csv contains
seed-level AUC differences-in-differences relative to the first declared level on each axis.
Four corner cells must have identical seed sets. Intervals are exploratory and unadjusted
for multiplicity. Nonparallel interaction lines alone are not statistical proof of an interaction.
Select settings on search seeds, freeze config, and assess them on held-out confirmatory seeds.
Without a declared threshold, success/retention/censoring endpoints are not applicable,
not failures. Pendulum has no declared success threshold in this study protocol. Its deterministic
policy uses the Gaussian mean with Box clipping; historical greedy columns keep their names.
Study results apply only to each recorded environment, protocol and implementation.
