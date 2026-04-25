# Transfer Memory Workflow

This repository now supports a simple transfer-memory mechanism for
TCL-style schedule reuse.

## What it does

During tuning, every successful measured schedule can be appended to a JSONL
memory file. On a later run, the search strategy retrieves the most similar
examples and uses them in two ways:

1. LLM prompt context
2. Mutator sampling priors

The idea is to preserve the REASONING COMPILER core while adding a lightweight
memory of what worked for related tensor programs.

## How to enable it

```python
strategy = ms.search_strategy.MCTSSearchPyFull(
    use_llm=True,
    llm_budget=600,
    llm_model_name="YOUR_MODEL",
    llm_history_depth=4,
    transfer_memory_path="./transfer_memory.jsonl",
    transfer_top_k=3,
    transfer_min_similarity=0.2,
)
```

## What gets stored

Each successful record stores:

- target kind
- lightweight module signature
- operator keywords
- trace excerpt
- mutator hints inferred from the trace
- measured runtime

## Recommended experiment

1. Tune a first batch of representative kernels and build `transfer_memory.jsonl`.
2. Run the same trial budget on a related but unseen kernel.
3. Compare:
   - original REASONING COMPILER
   - REASONING COMPILER with transfer memory enabled
4. Report:
   - best speedup at fixed trial budget
   - time-to-threshold
   - final best runtime

## Why this is useful

This gives you a concrete bridge from recent transfer-learning papers like
TCL to the existing REASONING COMPILER implementation without having to train
a new cost model or a new policy network first.
