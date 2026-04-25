# Core MCTS Search Improvement Summary

This note summarizes a direct improvement to the REASONING COMPILER search algorithm itself.

## What changed

The original MCTS expansion path accepted the first valid random mutation from a selected node.
That is sample-efficient in implementation cost, but it can spend hardware measurements on
low-quality children because validity and quality are treated as the same gate.

The updated search policy adds a cost-model-guided expansion tournament:

1. Sample several valid mutated schedules from the same selected parent.
2. Score those candidate children with the current TVM cost model.
3. Add only the highest-scoring child to the MCTS tree.
4. Allow a wider per-node branching factor so promising nodes can explore more than two children.
5. Backpropagate real measured runner latency into the MCTS tree, so later UCB selection is guided
   by hardware feedback instead of only rollout predictions.

The new knobs are:

- `mcts_expansion_candidates`: number of valid children sampled before choosing one.
- `mcts_max_children_per_node`: branching cap for non-root MCTS nodes.
- `mcts_use_measured_feedback`: enables measured-latency backpropagation.

The original behavior is recovered with:

```python
mcts_expansion_candidates=1
mcts_max_children_per_node=2
mcts_use_measured_feedback=False
```

The current improved default is:

```python
mcts_expansion_candidates=4
mcts_max_children_per_node=4
mcts_use_measured_feedback=True
```

## Why this improves REASONING COMPILER itself

This change is independent of transfer memory and independent of custom CUDA module hooks. It
changes the core search policy used to decide which schedules enter the MCTS tree and later become
hardware measurement candidates.

The expected benefit is better sample efficiency: at the same measured-trial budget, MCTS should
spend fewer measurements on weak children and more measurements on candidates that are already
predicted to be promising.

## CPU validation on server21

Environment:

- Repository: `/home/hansol/ai_platform_opt_reasoning_compiler`
- Runtime: `/home/hansol/tvm_xgb_diag/envs/wheel_xgb176_iso`
- Cost model: `xgb`
- Target: `llvm --num-cores=1`

### Quick ablation

Result file:

- `experiments/results/core_mcts_expansion_ablation_2seeds_8trials.json`

Setup:

- Shape: `[64, 64, 64]`
- Trials: `8`
- Seeds: `0 1`

Observed result:

- Average measured improvement: `1.0893x`
- Average compiled improvement: `1.0714x`

### Wider ablation

Result file:

- `experiments/results/core_mcts_expansion_ablation_4seeds_12trials.json`

Setup:

- Shape: `[64, 64, 64]`
- Trials: `12`
- Seeds: `0 1 2 3`

Observed result:

- Average measured improvement: `1.0348x`
- Median measured improvement: `1.0137x`
- Average compiled improvement: `1.0066x`
- Median compiled improvement: `1.0364x`

Interpretation:

- The core MCTS change is directionally positive on measured best latency.
- Compiled latency is still noisy; one seed regressed, while three seeds improved.
- The correct current claim is modest but real: cost-model-guided expansion improves the
  REASONING COMPILER search policy directionally, but more operators/seeds are needed before
  claiming a statistically strong result.

### Wider ablation with measured-feedback backpropagation

Result file:

- `experiments/results/core_mcts_measured_feedback_ablation_4seeds_12trials.json`

Setup:

- Shape: `[64, 64, 64]`
- Trials: `12`
- Seeds: `0 1 2 3`
- Original: `mcts_expansion_candidates=1`, `mcts_max_children_per_node=2`,
  `mcts_use_measured_feedback=False`
- Improved: `mcts_expansion_candidates=4`, `mcts_max_children_per_node=4`,
  `mcts_use_measured_feedback=True`

Observed result:

- Average measured improvement: `1.1211x`
- Geometric measured improvement: `1.1075x`
- Median measured improvement: `1.0441x`
- Average compiled improvement: `1.1258x`
- Geometric compiled improvement: `1.1098x`
- Median compiled improvement: `1.0294x`

Interpretation:

- Adding measured-latency feedback makes the core algorithm improvement materially stronger.
- The best current single-sentence claim is: at a fixed 12-trial budget on the CPU matmul
  ablation, the improved REASONING COMPILER is `1.11x` faster by geometric mean compiled latency
  than the original MCTS configuration.
- One seed still regresses slightly, so this should be validated on more seeds and at least one
  CUDA workload before being presented as a final result.

## Recommended next validation

The strongest next step is to repeat the same original-vs-improved ablation on:

1. A larger matmul or attention-like CPU TensorIR workload.
2. A CUDA workload on server17 with A6000 GPUs.
3. A fixed low trial budget, because the algorithmic goal is sample efficiency.
