# Adaptive Transfer Memory for Reasoning Compiler

> Improving low-budget CPU matmul tuning with adaptive transfer-guided MCTS.

This repository extends the Reasoning Compiler prototype with adaptive transfer-memory guidance for low-budget CPU matmul tuning, achieving **1.188x average compiled runtime speedup** across five transfer cases.

This project is built on top of [Reasoning Compiler (NeurIPS 2025)](https://openreview.net/forum?id=2D4TuZyNnr), a MCTS-based compiler optimization framework.

## Motivation

Prior work on [transfer-tuning](https://arxiv.org/abs/2201.05587) showed that auto-schedules from related tensor programs can be reused to reduce expensive search. This project explores whether the same idea can improve Reasoning Compiler's low-budget MCTS search: instead of starting every target workload from scratch, the tuner stores useful schedules from a source workload and reuses them as guidance for a related target workload.

The key challenge is avoiding negative transfer. Directly replaying prior schedules can help medium workloads, but it can hurt small kernels where the baseline search already finds good schedules quickly and runtime measurements are noisy. This motivates an adaptive transfer strategy that uses prior schedules more cautiously for small targets while still exploiting them for larger or nearby workloads.

## What This Project Adds

The main implementation changes are concentrated in `python/tvm/meta_schedule/search_strategy/mcts_search.py`, with supporting LLM-guidance robustness changes in `llm_guidance.py`. This project adds transfer-memory storage and retrieval, an adaptive small-target guard, measured-feedback backpropagation, transfer-guided mutator weighting, and mutator success tracking.

## Highlights

- **Adaptive transfer memory**: stores strong schedules from previous tuning runs and reuses them as search guidance for related workloads.
- **Small-target guard**: avoids direct trace replay on small CPU kernels where transfer can overfit or amplify measurement noise.
- **Transfer-guided MCTS**: biases mutator selection using prior schedule traces while preserving MCTS exploration.
- **Measured-feedback backpropagation**: feeds measured runtime results back into the search tree.
- **Low-budget focus**: targets the practical setting where only a small number of tuning trials are available.

## Results

All results use CPU `llvm`, TVM MetaSchedule, XGBoost cost model, `24` source trials, `8` target trials, and four seeds (`0, 1, 2, 3`). The primary metric is **compiled runtime speedup**:

```text
speedup = baseline Reasoning Compiler compiled runtime / improved compiled runtime
```

| Transfer case | What it tests | Avg speedup | Median speedup |
| :---: | --- | ---: | ---: |
| `64³→64³` | Small same-shape; verifies regression control | **1.125x** | 1.106x |
| `96³→128³` | Upscaling from smaller to medium workload | **1.157x** | 1.201x |
| `128³→128³` | Medium same-shape schedule reuse | **1.235x** | 1.155x |
| `128³→160³` | Upscaling from medium to larger workload | **1.262x** | 1.230x |
| `160³→128³` | Downscaling from larger to medium workload | **1.164x** | 1.147x |

**Aggregate:** `1.188x` arithmetic mean and `1.187x` geometric mean compiled runtime speedup across the final suite.

## What `A³→B³` Means

`A³→B³` means that the tuner first builds transfer memory from an `A×A×A` matmul workload, then uses that memory to guide tuning for a `B×B×B` target workload.

This evaluates whether schedule knowledge transfers across:

- same-shape reuse,
- smaller-to-larger upscaling,
- larger-to-smaller downscaling,
- and small-kernel cases where naive transfer can be risky.

## Method

The baseline Reasoning Compiler already uses MCTS-style search over TVM MetaSchedule transformations. This project adds an adaptive transfer layer around that search:

```
                 Source workload
                         |
                         v
              tune source (24 trials)
                         |
                         v
               transfer_memory.jsonl
                         |
                         v
        similarity scoring against target workload
                         |
             +-----------+------------+
             |                        |
      small target?           medium / large target
     (max grid <= 64)          (max grid > 64)
             |                        |
             v                        v
   prior-only warm-start      trace replay warm-start
   (no direct replay)         (top-k schedules injected)
             |                        |
             +-----------+------------+
                         |
                         v
        transfer-guided MCTS (8 trials)
        - mutator probabilities biased by prior
        - measured latency fed back into tree
                         |
                         v
                 compiled schedule
```

The key design choice is the **adaptive guard**. The guard determines how transfer memory enters the target search: small targets receive prior-only mutator guidance, while medium and larger targets can also receive replayed top-k schedules. This keeps the transfer mechanism useful without forcing every workload to accept the same warm-start strategy.

## Repository Map

Important files:

- `python/tvm/meta_schedule/search_strategy/mcts_search.py`  
  Main implementation: transfer memory, adaptive guard, transfer-guided mutator weighting, measured feedback.

- `python/tvm/meta_schedule/search_strategy/llm_guidance.py`  
  Safer LLM client initialization and more robust mutator response parsing.

- `tools/run_cpu_transfer_memory_benchmark.py`  
  CPU matmul benchmark used to compare baseline Reasoning Compiler search against the adaptive transfer version.

- `VER3_FINAL_ADAPTIVE_RESULTS.md`  
  Frozen final result table and interpretation.

- `experiments/results/`  
  JSON outputs for the final reported experiments.

## Reproducing the Final Suite

Example command:

```bash
PYTHONNOUSERSITE=1 python tools/run_cpu_transfer_memory_benchmark.py \
  --source-shape 128 128 128 \
  --target-shape 160 160 160 \
  --source-max-trials 24 \
  --target-max-trials 8 \
  --num-trials-per-iter 4 \
  --seeds 0 1 2 3 \
  --cost-model xgb \
  --transfer-top-k 3 \
  --transfer-warmstart-limit 1 \
  --output experiments/results/example_128_to_160.json
```

The final result file lists the exact JSON result path for each reported case.

## Scope

The result should be interpreted precisely:

> Adaptive transfer memory improves the tested low-budget CPU matmul transfer suite over the current Reasoning Compiler baseline.

It is not a universal claim across all operators, all shapes, or all hardware targets. The current evidence is strongest for CPU matmul workloads under low target-tuning budgets.

## Limitations

The `96³→96³` same-shape case was tested but excluded from the final reported suite: it regressed to `0.949x` under direct transfer warm-start, where the adaptive guard did not fully prevent replay interference on this particular small kernel. This is why the final claim is scoped to the five-case transfer suite rather than all matmul shapes.

## Acknowledgments

This project builds on Apache TVM MetaSchedule and the public Reasoning Compiler prototype:

- [Apache TVM](https://github.com/apache/tvm)
- [REASONING_COMPILER](https://github.com/Anna-Bele/REASONING_COMPILER)
