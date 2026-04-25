# Version 3 Final Adaptive Results

This file freezes the Version 3 result suite for the adaptive transfer-memory Reasoning Compiler.

## Method

Version 3 keeps the Version 2 transfer-memory warm-start idea, then adds a small-target adaptive guard:

- Extract loop/grid extent information from the TIR module signature.
- For small CPU kernels whose maximum grid extent is `<= 64`, disable direct trace replay warm-start.
- Keep transfer memory active as a mutator prior, so the search is still guided by prior schedules.
- For medium-size and nearby cross-shape cases, keep direct transfer warm-start enabled.

This avoids small-kernel regression while preserving the gains on medium-size transfer workloads.

## Final Result Suite

Common settings:

- Target: CPU `llvm`
- Operator family: matmul TIR workloads
- Cost model: `xgb`
- Source max trials: `24`
- Target max trials: `8`
- Trials per iteration: `4`
- Seeds: `0, 1, 2, 3`
- Primary metric: compiled runtime speedup, baseline Reasoning Compiler divided by Version 3 runtime

| Case | Meaning | Result JSON | Avg compiled speedup | Median compiled speedup |
| --- | --- | --- | ---: | ---: |
| `64^3 -> 64^3` | Small same-shape case; verifies that the adaptive guard prevents small-target regression. | `experiments/results/ver3_adaptive_gridguard_same_shape_64_s24_t8_topk3_warm1_4seeds.json` | `1.124712x` | `1.105592x` |
| `96^3 -> 128^3` | Up-scaling transfer from a smaller source workload to a medium target workload. | `experiments/results/ver3_adaptive_cross_shape_96_to_128_s24_t8_topk3_warm1_4seeds.json` | `1.156597x` | `1.201363x` |
| `128^3 -> 128^3` | Medium same-shape transfer; best representative low-budget reuse case. | `experiments/results/ver3_adaptive_same_shape_128_s24_t8_topk3_warm1_4seeds.json` | `1.234897x` | `1.154655x` |
| `128^3 -> 160^3` | Up-scaling transfer from a medium source workload to a larger target workload. | `experiments/results/ver3_adaptive_cross_shape_128_to_160_s24_t8_topk3_warm1_4seeds.json` | `1.261922x` | `1.230011x` |
| `160^3 -> 128^3` | Down-scaling transfer; tests whether schedules from a larger workload help a medium target. | `experiments/results/ver3_adaptive_cross_shape_160_to_128_s24_t8_topk3_warm1_4seeds.json` | `1.163721x` | `1.147164x` |

## Aggregate

- All final cases exceed `1.08x` average compiled speedup.
- Minimum average compiled speedup across final cases: `1.124712x`
- Arithmetic mean average compiled speedup: `1.188370x`
- Geometric mean average compiled speedup: `1.187263x`
- Mean of per-case median compiled speedups: `1.167757x`

## Recommended Claim

The strongest safe claim is:

> Version 3 improves the current Reasoning Compiler on the tested low-budget CPU matmul transfer suite. Across five same-shape and nearby cross-shape cases, it achieves `1.188x` average compiled runtime speedup, with every tested final case exceeding `1.08x`.

Avoid claiming universal superiority across all operators. The evidence supports a narrower but useful systems claim: adaptive transfer memory improves low-budget CPU matmul tuning for small, medium, and nearby cross-shape workloads.
