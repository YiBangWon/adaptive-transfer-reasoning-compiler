#!/usr/bin/env python3
"""Analyze paired baseline vs transfer benchmark results with bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from typing import Callable, Dict, List, Sequence


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def bootstrap_ci(
    samples: Sequence[float],
    *,
    stat_fn: Callable[[Sequence[float]], float],
    iterations: int,
    seed: int,
) -> List[float]:
    if not samples:
        raise ValueError("bootstrap_ci requires at least one sample")
    if len(samples) == 1:
        stat = float(stat_fn(samples))
        return [stat, stat]
    rng = random.Random(seed)
    n = len(samples)
    stats: List[float] = []
    for _ in range(iterations):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        stats.append(float(stat_fn(resample)))
    stats.sort()
    return [percentile(stats, 0.025), percentile(stats, 0.975)]


def arithmetic_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def geometric_mean_ratio(values: Sequence[float]) -> float:
    logs = [math.log(v) for v in values]
    return float(math.exp(sum(logs) / len(logs)))


def analyze_metric(
    trial_results: Sequence[Dict[str, float]],
    *,
    baseline_key: str,
    transfer_key: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    per_seed = []
    ratios = []
    log_ratios = []
    delta_secs = []
    delta_us = []
    for row in trial_results:
        baseline = float(row[baseline_key])
        transfer = float(row[transfer_key])
        ratio = baseline / transfer
        log_ratio = math.log(ratio)
        delta_sec = baseline - transfer
        ratios.append(ratio)
        log_ratios.append(log_ratio)
        delta_secs.append(delta_sec)
        delta_us.append(delta_sec * 1e6)
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "baseline_run_sec": baseline,
                "transfer_run_sec": transfer,
                "speedup_ratio": ratio,
                "delta_sec": delta_sec,
                "delta_us": delta_sec * 1e6,
            }
        )

    ratio_ci_log = bootstrap_ci(
        log_ratios,
        stat_fn=arithmetic_mean,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    ratio_ci = [float(math.exp(value)) for value in ratio_ci_log]
    delta_ci_us = bootstrap_ci(
        delta_us,
        stat_fn=arithmetic_mean,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed + 1,
    )
    return {
        "num_seeds": len(per_seed),
        "per_seed": per_seed,
        "arithmetic_mean_speedup": arithmetic_mean(ratios),
        "geometric_mean_speedup": geometric_mean_ratio(ratios),
        "mean_delta_sec": arithmetic_mean(delta_secs),
        "mean_delta_us": arithmetic_mean(delta_us),
        "bootstrap_ci95_speedup_ratio": ratio_ci,
        "bootstrap_ci95_mean_delta_us": delta_ci_us,
        "ci_excludes_no_effect_ratio": bool(ratio_ci[0] > 1.0 or ratio_ci[1] < 1.0),
        "ci_excludes_no_effect_delta": bool(delta_ci_us[0] > 0.0 or delta_ci_us[1] < 0.0),
    }


def analyze_file(
    input_path: pathlib.Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    trial_results = data["trial_results"]
    compiled = analyze_metric(
        trial_results,
        baseline_key="baseline_compiled_run_sec",
        transfer_key="transfer_compiled_run_sec",
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    measured = analyze_metric(
        trial_results,
        baseline_key="baseline_best_run_sec",
        transfer_key="transfer_best_run_sec",
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 1000,
    )
    return {
        "input_path": str(input_path),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "source_shape": data.get("source_shape"),
        "target_shape": data.get("target_shape"),
        "max_trials": data.get("max_trials"),
        "num_trials_per_iter": data.get("num_trials_per_iter"),
        "cost_model": data.get("cost_model"),
        "seeds": data.get("seeds"),
        "compiled": compiled,
        "measured": measured,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional output JSON path. Defaults to <input>.analysis.json",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260422)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = args.input.with_suffix(args.input.suffix + ".analysis.json")
    analysis = analyze_file(
        args.input,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
