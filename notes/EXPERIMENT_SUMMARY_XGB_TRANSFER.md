# XGB Transfer Memory Summary

This note summarizes the CPU-side validation runs for the REASONING_COMPILER transfer-memory extension.

## Environment

- Repository: `/home/hansol/ai_platform_opt_reasoning_compiler`
- Stable runtime used for XGBoost experiments:
  - `PYTHONNOUSERSITE=1`
  - `micromamba run -p /home/hansol/tvm_xgb_diag/envs/wheel_xgb176_iso`
  - `xgboost==1.7.6`
- Important constraint:
  - The server's default Python environment is not stable for TVM MetaSchedule `xgb`.
  - Host-side `xgboost==3.2.0` crashes with `free(): invalid pointer`.

## What Was Added

The transfer-memory path now does two things:

1. It records replayable trace JSON in addition to lightweight trace statistics.
2. It warm-starts the target workload by replaying compatible transfer traces into the initial MCTS population.

This is stronger than the previous version, which only biased mutator categories and prompt context.

## Result Files

- `experiments/results/same_shape_xgb.json`
- `experiments/results/same_shape_xgb_warm.json`
- `experiments/results/cross_shape_xgb.json`
- `experiments/results/cross_shape_xgb_warm.json`

## Aggregate Results

### Same-shape transfer

- Source shape: `[128, 128, 128]`
- Target shape: `[128, 128, 128]`
- Trials: `12`
- Seeds: `0 1`

Before warm-start:

- Average measured improvement: `1.0664x`
- Average compiled improvement: `0.8409x`

After warm-start:

- Average measured improvement: `1.1455x`
- Average compiled improvement: `1.2107x`

Interpretation:

- The previous transfer-memory version was not reliably better after compilation.
- Replay-based warm-start converted the same-shape case into a positive compiled-speedup result on average.

### Same-shape validation with paired seeds and bootstrap CI

- Source shape: `[128, 128, 128]`
- Target shape: `[128, 128, 128]`
- Trials: `24`
- Seeds: `0 1 2 3 4 5 6 7`
- Result file:
  - `experiments/results/same_shape_xgb_warm_8seeds_24trials.json`
  - `experiments/results/same_shape_xgb_warm_8seeds_24trials.analysis.json`

Observed result:

- Compiled arithmetic mean speedup: `1.0281x`
- Compiled geometric mean speedup: `1.0063x`
- Compiled bootstrap 95% CI (ratio): `[0.8786, 1.1647]`

Interpretation:

- This is the most honest equal-budget same-shape validation result.
- The mean is slightly positive, but the CI still includes `1.0`.
- Therefore, equal-budget same-shape transfer should be described as:
  - `promising but not yet statistically reliable`

### Same-shape low-target-budget validation

- Source shape: `[128, 128, 128]`
- Target shape: `[128, 128, 128]`
- Source tuning budget: `24`
- Target tuning budget: `8`
- Seeds: `0 1 2 3 4 5 6 7`
- Result file:
  - `experiments/results/same_shape_xgb_lowtarget_8seeds_s24_t8.json`
  - `experiments/results/same_shape_xgb_lowtarget_8seeds_s24_t8.analysis.json`

Observed result:

- Measured arithmetic mean speedup: `2.0045x`
- Measured geometric mean speedup: `1.4238x`
- Measured bootstrap 95% CI (ratio): `[0.9919, 2.4348]`
- Measured bootstrap 95% CI (mean delta us): `[0.0279, 92.9419]`
- Compiled arithmetic mean speedup: `1.2907x`
- Compiled geometric mean speedup: `1.1881x`
- Compiled bootstrap 95% CI (ratio): `[0.9182, 1.5868]`

Important caution:

- This low-budget result contains a strong outlier on `seed 7`, where baseline degraded badly.
- Leave-one-out sensitivity still keeps the direction positive, but the CI no longer excludes no-effect once that outlier is removed.

Interpretation:

- Low-target-budget transfer is the strongest current direction.
- It supports the claim that transfer memory is especially useful when the target tuning budget is tight.
- But it should still be presented as:
  - `encouraging, with outlier sensitivity still present`

### Cross-shape transfer

- Source shape: `[96, 96, 96]`
- Target shape: `[128, 128, 128]`
- Trials: `12`
- Seeds: `0 1`

Before warm-start:

- Average measured improvement: `1.0338x`
- Average compiled improvement: `1.0033x`

After warm-start:

- Average measured improvement: `0.9876x`
- Average compiled improvement: `1.0631x`

Interpretation:

- Cross-shape transfer remains noisier than same-shape transfer.
- Even so, replay warm-start improved the average compiled result from roughly break-even to a modest positive speedup.

## Practical Conclusion

For the current CPU experiments, the strongest claim supported by the data is:

- `transfer-memory + replay warm-start` is clearly directionally useful.
- Under equal target budget, the same-shape result is still too noisy to claim reliable improvement.
- Under low target budget, the effect becomes much stronger, especially on TVM's measured best latency, but it is still sensitive to outliers.
- Cross-shape remains exploratory and should not be presented as a main result.

## Recommended Next Step

If you want a stronger course-project or portfolio result, the next best experiment is:

- Keep `same-shape` as the main result and `cross-shape` as exploratory.
- Use paired seeds only.
- Report both arithmetic and geometric speedup, plus bootstrap CI.
- Add a robustness section that reports leave-one-out behavior for large outliers.
- Test one more operator family such as attention or softmax-heavy decode kernels.
