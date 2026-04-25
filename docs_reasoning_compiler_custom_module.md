# Adding a Custom Module to REASONING COMPILER

This repository is built on top of TVM MetaSchedule, so the cleanest way to attach a new module
that is not already covered by CUDA libraries is:

1. Lower the new operator into TensorIR.
2. Register a custom schedule rule for that operator.
3. Add one or more custom mutators so MCTS can keep exploring operator-specific transformations.
4. Pass operator-specific hints into the LLM prompt so the reasoning model knows what the new
   module is trying to optimize.

## Why this path works

The stock CUDA mutator set is strong for common operators, but a custom module usually needs two
things that cuBLAS/cuDNN-style kernels do not provide for free:

- A design space that reflects the operator's semantics.
- Search actions that preserve the module's structure while still exploring profitable mappings.

REASONING COMPILER already supports both pieces:

- TVM custom schedule rules via `ApplyCustomRule`
- Python-side mutators via `PyMutator`
- LLM guidance over arbitrary mutator sets through `MCTSSearchPyFull`

## Recommended integration recipe

### 1. Define the operator in TensorIR

Represent the new module as a TensorIR `IRModule` or `PrimFunc`. If the operator has a distinctive
compute block, attach a `schedule_rule` annotation to that block so TVM can dispatch your custom
rule instead of relying only on the default CUDA rule set.

### 2. Register a custom schedule rule

Use TVM's custom schedule-rule mechanism to seed the search space with transformations that are
actually meaningful for the new module. This is the right place for operator-aware tiling,
memory-layout decisions, tensorization boundaries, or staging rules.

### 3. Add a custom mutator

When the operator needs transformations that are not represented by the default mutator set, add a
Python-side mutator and include it in `mutator_probs`. A simplified example looks like this:

```python
from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Trace


@derived_object
class MutateCustomTensorize(ms.mutator.PyMutator):
    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        self.target = context.target

    def apply(self, trace: Trace, _) -> Trace | None:
        # Insert or rewrite the trace for your custom tensorization pattern.
        # Return None if the mutation is invalid for the current schedule.
        return trace

    def clone(self):
        return MutateCustomTensorize()
```

Then pass it into the schedule space:

```python
cuda_mutators = ms.mutator.create("cuda")
cuda_mutators[MutateCustomTensorize()] = 0.15

space = ms.space_generator.ScheduleFn(
    sch_fn=my_schedule_seed,
    sch_rules=[],
    postprocs=[],
    mutator_probs=cuda_mutators,
)
```

### 4. Tell the LLM what is special about the module

`MCTSSearchPyFull` now accepts `llm_extra_prompt_context`, so you can describe the operator's
hardware constraints or performance goals directly in the prompt:

```python
strategy = ms.search_strategy.MCTSSearchPyFull(
    use_llm=True,
    llm_budget=600,
    llm_model_name="YOUR_MODEL",
    llm_history_depth=4,
    llm_extra_prompt_context=(
        "This custom module is memory-bound, uses a producer-consumer pipeline, "
        "and benefits from shared-memory staging plus warp-cooperative vector loads."
    ),
)
```

That prompt context is especially useful when the module is not covered by a CUDA vendor library,
because the LLM can reason about the new kernel as a first-class search target instead of treating
it like a generic matmul or convolution.

For repeatable experiments, prefer a checked-in custom module spec instead of a long inline string:

```python
strategy = ms.search_strategy.MCTSSearchPyFull(
    use_llm=True,
    llm_budget=600,
    llm_model_name="YOUR_MODEL",
    llm_history_depth=4,
    custom_module_spec_path="examples/custom_modules/fused_rmsnorm_swiglu_spec.json",
)
```

A spec is a JSON object or JSONL row with fields like this:

```json
{
  "name": "fused_rmsnorm_swiglu_cuda",
  "target_kinds": ["cuda", "cuda-tensorcore"],
  "match_keywords": ["rmsnorm", "swiglu", "silu", "gate"],
  "match_mode": "any",
  "prompt_context": "This fused kernel is memory-bandwidth limited...",
  "schedule_hints": [
    "Keep row reductions inside a cooperative thread group.",
    "Prefer vectorized loads/stores along the contiguous hidden dimension."
  ],
  "mutator_prior_by_category": {
    "thread_binding": 0.55,
    "tile_size": 0.35
  }
}
```

When the current TensorIR matches the spec, REASONING COMPILER now:

1. Adds the spec's operator/hardware context to the LLM prompt.
2. Biases random MCTS mutator sampling toward the listed categories.
3. Keeps the comparison fair by still measuring all candidates through the same TVM runner.

### 5. Load operator-specific mutators as a plugin

If the new module needs an action that is not represented by TVM's default CUDA mutators, put it in
a Python plugin and pass the plugin path to the strategy:

```python
strategy = ms.search_strategy.MCTSSearchPyFull(
    use_llm=True,
    llm_budget=600,
    llm_model_name="YOUR_MODEL",
    custom_module_spec_path="examples/custom_modules/fused_rmsnorm_swiglu_spec.json",
    custom_mutator_plugin_paths="path/to/your_custom_mutators.py",
)
```

The plugin must expose:

```python
def register_mutators(context):
    return {MyCustomMutator(): 0.20}
```

The returned probabilities are unnormalized weights. They are merged with the default CUDA mutator
set before MCTS begins, so the LLM can select the new mutator by its exact TVM string name and the
random rollout path can sample it too.

Run a quick loader smoke test with:

```bash
PYTHONNOUSERSITE=1 \
/home/hansol/tvm_xgb_diag/artifacts/micromamba/bin/micromamba run \
  -p /home/hansol/tvm_xgb_diag/envs/wheel_xgb176_iso \
  python tools/validate_custom_module_extension.py
```

## Practical experiment plan

If the goal is to beat the paper's sample-efficiency curve, the most realistic sequence is:

1. Keep the default MCTS backbone.
2. Add operator-specific mutators for the new module.
3. Increase LLM history depth to 4 or 5.
4. Reuse any strong prior schedule traces you already measured for similar operators.
5. Compare against the original REASONING COMPILER with the same trial budget.

This keeps the comparison fair while giving the search process more structure than the paper's
base implementation.
