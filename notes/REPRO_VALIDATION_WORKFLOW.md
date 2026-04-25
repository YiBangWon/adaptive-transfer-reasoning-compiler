# Repro Validation Workflow

This file documents the stable workflow for REASONING_COMPILER validation on `aso-server21`.

## Stable Environment

Do not use the server's default Python environment for `xgb` cost-model experiments.

Use:

- `PYTHONNOUSERSITE=1`
- `micromamba` environment: `/home/hansol/tvm_xgb_diag/envs/wheel_xgb176_iso`
- `xgboost==1.7.6`

The helper runner already applies these settings:

- `/home/hansol/ai_platform_opt_reasoning_compiler/tools/run_transfer_benchmark_in_env.sh`

## tmux Workflow

The recommended launcher is:

- `/home/hansol/ai_platform_opt_reasoning_compiler/tools/start_transfer_benchmark_tmux.sh`

### Example: main same-shape validation

```bash
cd /home/hansol/ai_platform_opt_reasoning_compiler

bash tools/start_transfer_benchmark_tmux.sh \
  rc-same-shape-8x24 \
  experiments/logs/rc-same-shape-8x24.log \
  --source-shape 128 128 128 \
  --target-shape 128 128 128 \
  --max-trials 24 \
  --num-trials-per-iter 4 \
  --seeds 0 1 2 3 4 5 6 7 \
  --cost-model xgb \
  --output experiments/results/same_shape_xgb_warm_8seeds_24trials.json
```

### Monitor progress

```bash
tmux attach -t rc-same-shape-8x24
```

or

```bash
tail -f /home/hansol/ai_platform_opt_reasoning_compiler/experiments/logs/rc-same-shape-8x24.log
```

### Run the paired bootstrap analysis

```bash
cd /home/hansol/ai_platform_opt_reasoning_compiler

PYTHONNOUSERSITE=1 \
/home/hansol/tvm_xgb_diag/artifacts/micromamba/bin/micromamba run \
  -p /home/hansol/tvm_xgb_diag/envs/wheel_xgb176_iso \
  python tools/analyze_transfer_results.py \
  experiments/results/same_shape_xgb_warm_8seeds_24trials.json \
  --output experiments/results/same_shape_xgb_warm_8seeds_24trials.analysis.json
```

## Reporting Guidance

- Treat `same-shape` as the main result.
- Treat `cross-shape` as an exploratory finding.
- Use paired seeds only.
- Report bootstrap CI, not just the mean speedup.
