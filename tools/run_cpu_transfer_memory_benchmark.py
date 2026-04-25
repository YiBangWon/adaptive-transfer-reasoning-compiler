#!/usr/bin/env python3
"""Run a small CPU benchmark comparing baseline MCTS vs transfer-memory MCTS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import statistics
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import te


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_repo_overrides() -> None:
    bootstrap_path = REPO_ROOT / "tools" / "bootstrap_reasoning_compiler_overrides.py"
    spec = importlib.util.spec_from_file_location("bootstrap_reasoning_compiler_overrides", bootstrap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bootstrap script from {bootstrap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_repo_overrides(REPO_ROOT)


def make_matmul_primfunc(m: int, n: int, k_dim: int) -> tvm.tir.PrimFunc:
    a = te.placeholder((m, k_dim), name="A", dtype="float32")
    b = te.placeholder((n, k_dim), name="B", dtype="float32")
    rk = te.reduce_axis((0, k_dim), name="k")
    c = te.compute(
        (m, n),
        lambda i, j: te.sum(a[i, rk] * b[j, rk], axis=rk),
        name="matmul",
    )
    return te.create_prim_func([a, b, c]).with_attr("global_symbol", "main")


def benchmark_module(mod: tvm.IRModule | tvm.tir.PrimFunc, target: tvm.target.Target, m: int, n: int, k_dim: int) -> float:
    rt_mod = tvm.build(mod, target=target)
    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=(m, k_dim)).astype("float32")
    b_np = np.random.uniform(size=(n, k_dim)).astype("float32")
    c_np = np.zeros((m, n), dtype="float32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(c_np, dev)
    evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=10, repeat=5)
    result = evaluator(a, b, c)
    return float(np.median(result.results))


def best_measured_run_sec(database: ms.database.Database, mod: tvm.IRModule | tvm.tir.PrimFunc) -> float:
    if isinstance(mod, tvm.tir.PrimFunc):
        mod = tvm.IRModule({"main": mod})
    workload = database.commit_workload(mod)
    records = database.get_top_k(workload, 1)
    if not records:
        raise RuntimeError("No tuning records found in database.")
    run_secs = records[0].run_secs or []
    if not run_secs:
        raise RuntimeError("Top tuning record has no runtime measurements.")
    return float(sum(run_secs) / len(run_secs))


def make_strategy(
    *,
    transfer_memory_path: str,
    transfer_top_k: int,
    transfer_min_similarity: float,
    transfer_warmstart_limit: int,
    transfer_enable_max_trials: int,
    mcts_max_children_per_node: int,
    mcts_expansion_candidates: int,
    mcts_use_measured_feedback: bool,
    seed: int,
) -> ms.search_strategy.SearchStrategy:
    return ms.search_strategy.MCTSSearchPyFull(
        population_size=16,
        init_measured_ratio=0.0,
        init_min_unmeasured=8,
        max_fail_count=16,
        genetic_num_iters=2,
        genetic_mutate_prob=0.5,
        genetic_max_fail_count=4,
        num_empty_iters_before_early_stop=6,
        max_stale_iters=20,
        mcts_ucb_constant=1.41,
        mcts_max_depth=16,
        mcts_max_children_per_node=mcts_max_children_per_node,
        mcts_expansion_candidates=mcts_expansion_candidates,
        mcts_use_measured_feedback=mcts_use_measured_feedback,
        mcts_num_threads=1,
        mcts_num_rollouts_per_expansion=1,
        use_llm=False,
        llm_budget=0,
        transfer_memory_path=transfer_memory_path,
        transfer_top_k=transfer_top_k,
        transfer_min_similarity=transfer_min_similarity,
        transfer_warmstart_limit=transfer_warmstart_limit,
        transfer_enable_max_trials=transfer_enable_max_trials,
        verbose=0,
    )


@dataclass
class TrialResult:
    seed: int
    baseline_best_run_sec: float
    transfer_best_run_sec: float
    baseline_compiled_run_sec: float
    transfer_compiled_run_sec: float

    @property
    def measured_improvement(self) -> float:
        return self.baseline_best_run_sec / self.transfer_best_run_sec

    @property
    def compiled_improvement(self) -> float:
        return self.baseline_compiled_run_sec / self.transfer_compiled_run_sec


def run_experiment(
    *,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    source_max_trials: int,
    target_max_trials: int,
    num_trials_per_iter: int,
    seeds: Sequence[int],
    cost_model: str,
    transfer_top_k: int,
    transfer_min_similarity: float,
    transfer_warmstart_limit: int,
    transfer_enable_max_trials: int,
    mcts_max_children_per_node: int,
    mcts_expansion_candidates: int,
    mcts_use_measured_feedback: bool,
    output_path: pathlib.Path,
) -> Dict[str, object]:
    load_repo_overrides()

    target = tvm.target.Target("llvm --num-cores=1")
    source_mod = make_matmul_primfunc(*source_shape)
    target_mod = make_matmul_primfunc(*target_shape)
    unoptimized_run_sec = benchmark_module(target_mod, target, *target_shape)

    trial_results: List[TrialResult] = []
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="rc_transfer_bench_", dir=str(REPO_ROOT)))
    try:
        for seed in seeds:
            seed_dir = temp_root / f"seed_{seed}"
            memory_path = seed_dir / "transfer_memory.jsonl"
            source_work_dir = seed_dir / "source"
            baseline_work_dir = seed_dir / "baseline"
            transfer_work_dir = seed_dir / "transfer"
            source_work_dir.mkdir(parents=True, exist_ok=True)
            baseline_work_dir.mkdir(parents=True, exist_ok=True)
            transfer_work_dir.mkdir(parents=True, exist_ok=True)

            ms.tir_integration.tune_tir(
                mod=source_mod,
                target=target,
                work_dir=str(source_work_dir),
                max_trials_global=source_max_trials,
                num_trials_per_iter=num_trials_per_iter,
                strategy=make_strategy(
                    transfer_memory_path=str(memory_path),
                    transfer_top_k=0,
                    transfer_min_similarity=transfer_min_similarity,
                    transfer_warmstart_limit=transfer_warmstart_limit,
                    transfer_enable_max_trials=transfer_enable_max_trials,
                    mcts_max_children_per_node=mcts_max_children_per_node,
                    mcts_expansion_candidates=mcts_expansion_candidates,
                    mcts_use_measured_feedback=mcts_use_measured_feedback,
                    seed=seed,
                ),
                cost_model=cost_model,
                seed=seed,
                num_tuning_cores=1,
            )

            baseline_database = ms.tir_integration.tune_tir(
                mod=target_mod,
                target=target,
                work_dir=str(baseline_work_dir),
                max_trials_global=target_max_trials,
                num_trials_per_iter=num_trials_per_iter,
                strategy=make_strategy(
                    transfer_memory_path="",
                    transfer_top_k=0,
                    transfer_min_similarity=transfer_min_similarity,
                    transfer_warmstart_limit=transfer_warmstart_limit,
                    transfer_enable_max_trials=transfer_enable_max_trials,
                    mcts_max_children_per_node=mcts_max_children_per_node,
                    mcts_expansion_candidates=mcts_expansion_candidates,
                    mcts_use_measured_feedback=mcts_use_measured_feedback,
                    seed=seed,
                ),
                cost_model=cost_model,
                seed=seed,
                num_tuning_cores=1,
            )
            transfer_database = ms.tir_integration.tune_tir(
                mod=target_mod,
                target=target,
                work_dir=str(transfer_work_dir),
                max_trials_global=target_max_trials,
                num_trials_per_iter=num_trials_per_iter,
                strategy=make_strategy(
                    transfer_memory_path=str(memory_path),
                    transfer_top_k=transfer_top_k,
                    transfer_min_similarity=transfer_min_similarity,
                    transfer_warmstart_limit=transfer_warmstart_limit,
                    transfer_enable_max_trials=transfer_enable_max_trials,
                    mcts_max_children_per_node=mcts_max_children_per_node,
                    mcts_expansion_candidates=mcts_expansion_candidates,
                    mcts_use_measured_feedback=mcts_use_measured_feedback,
                    seed=seed,
                ),
                cost_model=cost_model,
                seed=seed,
                num_tuning_cores=1,
            )

            baseline_best_run_sec = best_measured_run_sec(baseline_database, target_mod)
            transfer_best_run_sec = best_measured_run_sec(transfer_database, target_mod)

            baseline_sch = ms.tir_integration.compile_tir(baseline_database, target_mod, target)
            transfer_sch = ms.tir_integration.compile_tir(transfer_database, target_mod, target)
            if baseline_sch is None or transfer_sch is None:
                raise RuntimeError("Failed to compile tuned schedules.")

            baseline_compiled_run_sec = benchmark_module(baseline_sch.mod, target, *target_shape)
            transfer_compiled_run_sec = benchmark_module(transfer_sch.mod, target, *target_shape)

            trial_results.append(
                TrialResult(
                    seed=seed,
                    baseline_best_run_sec=baseline_best_run_sec,
                    transfer_best_run_sec=transfer_best_run_sec,
                    baseline_compiled_run_sec=baseline_compiled_run_sec,
                    transfer_compiled_run_sec=transfer_compiled_run_sec,
                )
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    measured_improvements = [result.measured_improvement for result in trial_results]
    compiled_improvements = [result.compiled_improvement for result in trial_results]

    summary = {
        "source_shape": list(source_shape),
        "target_shape": list(target_shape),
        "source_max_trials": source_max_trials,
        "target_max_trials": target_max_trials,
        "num_trials_per_iter": num_trials_per_iter,
        "cost_model": cost_model,
        "transfer_top_k": transfer_top_k,
        "transfer_min_similarity": transfer_min_similarity,
        "transfer_warmstart_limit": transfer_warmstart_limit,
        "transfer_enable_max_trials": transfer_enable_max_trials,
        "mcts_max_children_per_node": mcts_max_children_per_node,
        "mcts_expansion_candidates": mcts_expansion_candidates,
        "mcts_use_measured_feedback": mcts_use_measured_feedback,
        "seeds": list(seeds),
        "unoptimized_run_sec": unoptimized_run_sec,
        "trial_results": [asdict(result) | {
            "measured_improvement": result.measured_improvement,
            "compiled_improvement": result.compiled_improvement,
        } for result in trial_results],
        "summary": {
            "avg_measured_improvement": statistics.mean(measured_improvements),
            "avg_compiled_improvement": statistics.mean(compiled_improvements),
            "median_measured_improvement": statistics.median(measured_improvements),
            "median_compiled_improvement": statistics.median(compiled_improvements),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shape", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--target-shape", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--source-max-trials", type=int)
    parser.add_argument("--target-max-trials", type=int)
    parser.add_argument("--num-trials-per-iter", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--cost-model", choices=["xgb", "random"], default="xgb")
    parser.add_argument("--transfer-top-k", type=int, default=3)
    parser.add_argument("--transfer-min-similarity", type=float, default=0.1)
    parser.add_argument("--transfer-warmstart-limit", type=int, default=1)
    parser.add_argument("--transfer-enable-max-trials", type=int, default=0)
    parser.add_argument("--mcts-max-children-per-node", type=int, default=4)
    parser.add_argument("--mcts-expansion-candidates", type=int, default=4)
    parser.add_argument("--no-mcts-measured-feedback", action="store_true")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=REPO_ROOT / "experiments" / "results" / "cpu_transfer_memory_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_max_trials = args.source_max_trials if args.source_max_trials is not None else args.max_trials
    target_max_trials = args.target_max_trials if args.target_max_trials is not None else args.max_trials
    summary = run_experiment(
        source_shape=args.source_shape,
        target_shape=args.target_shape,
        source_max_trials=source_max_trials,
        target_max_trials=target_max_trials,
        num_trials_per_iter=args.num_trials_per_iter,
        seeds=args.seeds,
        cost_model=args.cost_model,
        transfer_top_k=args.transfer_top_k,
        transfer_min_similarity=args.transfer_min_similarity,
        transfer_warmstart_limit=args.transfer_warmstart_limit,
        transfer_enable_max_trials=args.transfer_enable_max_trials,
        mcts_max_children_per_node=args.mcts_max_children_per_node,
        mcts_expansion_candidates=args.mcts_expansion_candidates,
        mcts_use_measured_feedback=not args.no_mcts_measured_feedback,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
