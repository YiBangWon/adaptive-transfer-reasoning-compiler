#!/usr/bin/env python3
"""Compare original MCTS expansion against cost-model-guided expansion."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from run_cpu_transfer_memory_benchmark import (
    benchmark_module,
    best_measured_run_sec,
    load_repo_overrides,
    make_matmul_primfunc,
)

import tvm
from tvm import meta_schedule as ms


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_strategy(
    *,
    mcts_max_children_per_node: int,
    mcts_expansion_candidates: int,
    mcts_use_measured_feedback: bool,
    mcts_measured_feedback_power: float,
    mcts_use_root_prior_selection: bool,
    mcts_measure_selection_q_weight: float,
    mcts_measure_selection_diversity_weight: float,
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
        mcts_measured_feedback_power=mcts_measured_feedback_power,
        mcts_use_root_prior_selection=mcts_use_root_prior_selection,
        mcts_measure_selection_q_weight=mcts_measure_selection_q_weight,
        mcts_measure_selection_diversity_weight=mcts_measure_selection_diversity_weight,
        mcts_num_threads=1,
        mcts_num_rollouts_per_expansion=1,
        use_llm=False,
        llm_budget=0,
        transfer_memory_path="",
        transfer_top_k=0,
        verbose=0,
    )


@dataclass
class TrialResult:
    seed: int
    original_best_run_sec: float
    improved_best_run_sec: float
    original_compiled_run_sec: float
    improved_compiled_run_sec: float

    @property
    def measured_improvement(self) -> float:
        return self.original_best_run_sec / self.improved_best_run_sec

    @property
    def compiled_improvement(self) -> float:
        return self.original_compiled_run_sec / self.improved_compiled_run_sec


def tune_once(
    *,
    mod: tvm.IRModule | tvm.tir.PrimFunc,
    target: tvm.target.Target,
    work_dir: pathlib.Path,
    max_trials: int,
    num_trials_per_iter: int,
    cost_model: str,
    seed: int,
    mcts_max_children_per_node: int,
    mcts_expansion_candidates: int,
    mcts_use_measured_feedback: bool,
    mcts_measured_feedback_power: float,
    mcts_use_root_prior_selection: bool,
    mcts_measure_selection_q_weight: float,
    mcts_measure_selection_diversity_weight: float,
) -> ms.database.Database:
    return ms.tir_integration.tune_tir(
        mod=mod,
        target=target,
        work_dir=str(work_dir),
        max_trials_global=max_trials,
        num_trials_per_iter=num_trials_per_iter,
        strategy=make_strategy(
            mcts_max_children_per_node=mcts_max_children_per_node,
            mcts_expansion_candidates=mcts_expansion_candidates,
            mcts_use_measured_feedback=mcts_use_measured_feedback,
            mcts_measured_feedback_power=mcts_measured_feedback_power,
            mcts_use_root_prior_selection=mcts_use_root_prior_selection,
            mcts_measure_selection_q_weight=mcts_measure_selection_q_weight,
            mcts_measure_selection_diversity_weight=mcts_measure_selection_diversity_weight,
        ),
        cost_model=cost_model,
        seed=seed,
        num_tuning_cores=1,
    )


def run_experiment(
    *,
    shape: Sequence[int],
    max_trials: int,
    num_trials_per_iter: int,
    seeds: Sequence[int],
    cost_model: str,
    improved_max_children_per_node: int,
    improved_expansion_candidates: int,
    improved_measured_feedback_power: float,
    improved_use_root_prior_selection: bool,
    improved_measure_selection_q_weight: float,
    improved_measure_selection_diversity_weight: float,
    output_path: pathlib.Path,
) -> Dict[str, object]:
    load_repo_overrides()

    target = tvm.target.Target("llvm --num-cores=1")
    mod = make_matmul_primfunc(*shape)
    unoptimized_run_sec = benchmark_module(mod, target, *shape)

    trial_results: List[TrialResult] = []
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="rc_core_mcts_", dir=str(REPO_ROOT)))
    try:
        for seed in seeds:
            seed_dir = temp_root / f"seed_{seed}"
            original_work_dir = seed_dir / "original"
            improved_work_dir = seed_dir / "improved"
            original_work_dir.mkdir(parents=True, exist_ok=True)
            improved_work_dir.mkdir(parents=True, exist_ok=True)

            original_database = tune_once(
                mod=mod,
                target=target,
                work_dir=original_work_dir,
                max_trials=max_trials,
                num_trials_per_iter=num_trials_per_iter,
                cost_model=cost_model,
                seed=seed,
                mcts_max_children_per_node=2,
                mcts_expansion_candidates=1,
                mcts_use_measured_feedback=False,
                mcts_measured_feedback_power=1.0,
                mcts_use_root_prior_selection=False,
                mcts_measure_selection_q_weight=0.0,
                mcts_measure_selection_diversity_weight=0.0,
            )
            improved_database = tune_once(
                mod=mod,
                target=target,
                work_dir=improved_work_dir,
                max_trials=max_trials,
                num_trials_per_iter=num_trials_per_iter,
                cost_model=cost_model,
                seed=seed,
                mcts_max_children_per_node=improved_max_children_per_node,
                mcts_expansion_candidates=improved_expansion_candidates,
                mcts_use_measured_feedback=True,
                mcts_measured_feedback_power=improved_measured_feedback_power,
                mcts_use_root_prior_selection=improved_use_root_prior_selection,
                mcts_measure_selection_q_weight=improved_measure_selection_q_weight,
                mcts_measure_selection_diversity_weight=(
                    improved_measure_selection_diversity_weight
                ),
            )

            original_best_run_sec = best_measured_run_sec(original_database, mod)
            improved_best_run_sec = best_measured_run_sec(improved_database, mod)

            original_sch = ms.tir_integration.compile_tir(original_database, mod, target)
            improved_sch = ms.tir_integration.compile_tir(improved_database, mod, target)
            if original_sch is None or improved_sch is None:
                raise RuntimeError("Failed to compile tuned schedules.")

            trial_results.append(
                TrialResult(
                    seed=seed,
                    original_best_run_sec=original_best_run_sec,
                    improved_best_run_sec=improved_best_run_sec,
                    original_compiled_run_sec=benchmark_module(original_sch.mod, target, *shape),
                    improved_compiled_run_sec=benchmark_module(improved_sch.mod, target, *shape),
                )
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    measured = [result.measured_improvement for result in trial_results]
    compiled = [result.compiled_improvement for result in trial_results]
    summary = {
        "shape": list(shape),
        "max_trials": max_trials,
        "num_trials_per_iter": num_trials_per_iter,
        "cost_model": cost_model,
        "seeds": list(seeds),
        "original": {
            "mcts_max_children_per_node": 2,
            "mcts_expansion_candidates": 1,
            "mcts_use_measured_feedback": False,
            "mcts_measured_feedback_power": 1.0,
            "mcts_use_root_prior_selection": False,
            "mcts_measure_selection_q_weight": 0.0,
            "mcts_measure_selection_diversity_weight": 0.0,
        },
        "improved": {
            "mcts_max_children_per_node": improved_max_children_per_node,
            "mcts_expansion_candidates": improved_expansion_candidates,
            "mcts_use_measured_feedback": True,
            "mcts_measured_feedback_power": improved_measured_feedback_power,
            "mcts_use_root_prior_selection": improved_use_root_prior_selection,
            "mcts_measure_selection_q_weight": improved_measure_selection_q_weight,
            "mcts_measure_selection_diversity_weight": (
                improved_measure_selection_diversity_weight
            ),
        },
        "unoptimized_run_sec": unoptimized_run_sec,
        "trial_results": [
            asdict(result)
            | {
                "measured_improvement": result.measured_improvement,
                "compiled_improvement": result.compiled_improvement,
            }
            for result in trial_results
        ],
        "summary": {
            "avg_measured_improvement": statistics.mean(measured),
            "avg_compiled_improvement": statistics.mean(compiled),
            "median_measured_improvement": statistics.median(measured),
            "median_compiled_improvement": statistics.median(compiled),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--num-trials-per-iter", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--cost-model", choices=["xgb", "random"], default="xgb")
    parser.add_argument("--improved-max-children-per-node", type=int, default=4)
    parser.add_argument("--improved-expansion-candidates", type=int, default=4)
    parser.add_argument("--improved-measured-feedback-power", type=float, default=1.0)
    parser.add_argument("--improved-use-root-prior-selection", action="store_true")
    parser.add_argument("--improved-measure-selection-q-weight", type=float, default=0.0)
    parser.add_argument("--improved-measure-selection-diversity-weight", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=REPO_ROOT / "experiments" / "results" / "core_mcts_expansion_ablation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_experiment(
                shape=args.shape,
                max_trials=args.max_trials,
                num_trials_per_iter=args.num_trials_per_iter,
                seeds=args.seeds,
                cost_model=args.cost_model,
                improved_max_children_per_node=args.improved_max_children_per_node,
                improved_expansion_candidates=args.improved_expansion_candidates,
                improved_measured_feedback_power=args.improved_measured_feedback_power,
                improved_use_root_prior_selection=args.improved_use_root_prior_selection,
                improved_measure_selection_q_weight=args.improved_measure_selection_q_weight,
                improved_measure_selection_diversity_weight=(
                    args.improved_measure_selection_diversity_weight
                ),
                output_path=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
