# Version 2 Final Result

This file freezes the best Version 2 result that passed the 1.2x target.

## Final Result

- Version: `ai_platform_opt_reasoning_compiler_ver2`
- Final experiment: transfer-memory warm-start, low target tuning budget
- Result JSON: `experiments/results/ver2_same_shape_lowtarget_4seeds_s24_t8.json`
- Log: `experiments/logs/ver2_same_shape_lowtarget_4seeds_s24_t8.log`
- Source shape: `128 x 128 x 128`
- Target shape: `128 x 128 x 128`
- Seeds: `0, 1, 2, 3`
- Cost model: `xgb`
- Source max trials: `24`
- Target max trials: `8`
- Trials per iteration: `4`
- Transfer top-k: `3`
- Transfer warm-start limit: `1`

## Speedup

Primary metric: compiled kernel runtime, averaged over four seeds.

| System | Avg compiled runtime |
| --- | ---: |
| Reasoning Compiler baseline | `20.655225 us` |
| Version 2 improved | `17.021925 us` |

Version 2 speedup over the Reasoning Compiler baseline:

- Average compiled speedup: `1.234897x`
- Median compiled speedup: `1.154655x`
- Average measured-search speedup: `1.093060x`
- Median measured-search speedup: `1.076481x`

## Per-Seed Compiled Speedup

| Seed | Baseline compiled | Version 2 compiled | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | `19.8252 us` | `16.9257 us` | `1.171308x` |
| 1 | `19.9253 us` | `19.7581 us` | `1.008462x` |
| 2 | `23.9098 us` | `14.7426 us` | `1.621817x` |
| 3 | `18.9606 us` | `16.6613 us` | `1.138002x` |

## Interpretation

Version 2 is considered the final saved 1.2x result because its average compiled speedup is `1.234897x`, which clears the target threshold. The next folder, `ai_platform_opt_reasoning_compiler_ver3`, should start from this state and search for a stronger configuration that can approach or exceed `1.3x`.
