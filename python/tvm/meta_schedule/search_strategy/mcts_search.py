# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import hashlib
import heapq
import importlib
import importlib.util
import json
import logging
import math
import os
import pathlib
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    Any,
    TYPE_CHECKING,
    List,
    Tuple,
    Dict,
    Optional,
    Set,
)

import tvm
from tvm._ffi import get_global_func
from tvm.runtime import Object
from tvm.tir import Schedule
from tvm.tir.schedule import Trace
from tvm.ir import IRModule

from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.runner import RunnerResult
from .search_strategy import SearchStrategy
from .search_strategy import PySearchStrategy
from .search_strategy import MeasureCandidate
from ..postproc import Postproc
from ..mutator import Mutator
from ..database import Workload
from .. import _ffi_api
from .llm_guidance import LLMGuidancePolicy

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext

from ..database import TuningRecord

try:
    from tvm.error import InvalidScheduleError
except ImportError:
    InvalidScheduleError = tvm.TVMError

logger = logging.getLogger("meta_schedule")
logger.setLevel(logging.DEBUG)

TRANSFER_OPERATOR_KEYWORDS = [
    "attention",
    "matmul",
    "gemm",
    "conv",
    "convolution",
    "softmax",
    "moe",
    "mlp",
    "norm",
    "rmsnorm",
    "layernorm",
    "decode",
    "reduce",
    "pool",
]

TRACE_HINT_PATTERNS = {
    "tile_size": ["sample_perfect_tile", "split("],
    "parallel": ["parallel(", "vectorize("],
    "compute_location": ["compute_at(", "reverse_compute_at("],
    "unroll": ["unroll(", "pragma_auto_unroll_max_step", "meta_schedule.unroll_explicit"],
    "thread_binding": ["bind(", "thread_binding("],
}

MUTATOR_CATEGORY_PATTERNS = {
    "tile_size": "MutateTileSize",
    "parallel": "MutateParallel",
    "compute_location": "MutateComputeLocation",
    "unroll": "MutateUnroll",
    "thread_binding": "MutateThreadBinding",
}


def _as_string_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _read_custom_module_spec_file(path: pathlib.Path) -> List[Dict[str, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        logger.warning("Failed to read custom module spec '%s': %s", path, err)
        return []
    if not text.strip():
        return []
    specs: List[Dict[str, object]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    specs.append(payload)
        else:
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("modules"), list):
                specs.extend(item for item in payload["modules"] if isinstance(item, dict))
            elif isinstance(payload, list):
                specs.extend(item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                specs.append(payload)
    except json.JSONDecodeError as err:
        logger.warning("Failed to parse custom module spec '%s': %s", path, err)
        return []
    for spec in specs:
        spec.setdefault("_source_path", str(path))
    return specs


def _load_custom_module_specs(spec_path: str) -> List[Dict[str, object]]:
    if not spec_path:
        return []
    root = pathlib.Path(spec_path).expanduser()
    if not root.exists():
        logger.warning("Custom module spec path does not exist: %s", root)
        return []
    if root.is_file():
        return _read_custom_module_spec_file(root)
    specs: List[Dict[str, object]] = []
    for child in sorted(root.iterdir()):
        if child.suffix.lower() in (".json", ".jsonl"):
            specs.extend(_read_custom_module_spec_file(child))
    return specs


def _target_matches_spec(current_signature: Dict[str, object], spec: Dict[str, object]) -> bool:
    target_kinds = [item.lower() for item in _as_string_list(spec.get("target_kinds"))]
    if not target_kinds:
        return True
    current_target = str(current_signature.get("target_kind", "")).lower()
    return current_target in target_kinds


def _custom_module_spec_matches(
    current_signature: Dict[str, object],
    spec: Dict[str, object],
) -> bool:
    if not _target_matches_spec(current_signature, spec):
        return False
    if bool(spec.get("always_apply", False)):
        return True
    match_keywords = [item.lower() for item in _as_string_list(spec.get("match_keywords"))]
    if not match_keywords:
        return False
    module_excerpt = str(current_signature.get("module_excerpt", "")).lower()
    signature_keywords = " ".join(_as_string_list(current_signature.get("keywords"))).lower()
    haystack = module_excerpt + "\n" + signature_keywords
    if str(spec.get("match_mode", "any")).lower() == "all":
        return all(keyword in haystack for keyword in match_keywords)
    return any(keyword in haystack for keyword in match_keywords)


def _match_custom_module_specs(
    current_signature: Optional[Dict[str, object]],
    spec_path: str,
) -> List[Dict[str, object]]:
    if current_signature is None:
        return []
    matched = []
    for spec in _load_custom_module_specs(spec_path):
        if _custom_module_spec_matches(current_signature, spec):
            matched.append(spec)
    if matched:
        logger.warning(
            "Matched %d custom module spec(s): %s",
            len(matched),
            [str(spec.get("name", "<unnamed>")) for spec in matched],
        )
    return matched


def _build_custom_module_prompt_context(matched_specs: List[Dict[str, object]]) -> str:
    if not matched_specs:
        return ""
    parts = [
        "Matched custom module specifications. Treat these as operator-specific "
        "hardware and scheduling hints for kernels not covered by vendor CUDA libraries."
    ]
    for spec in matched_specs:
        name = str(spec.get("name", "<unnamed custom module>"))
        prompt_context = str(spec.get("prompt_context", "")).strip()
        schedule_hints = _as_string_list(spec.get("schedule_hints"))
        hint_text = ""
        if schedule_hints:
            hint_text = "\nSchedule hints:\n" + "\n".join(f"- {hint}" for hint in schedule_hints)
        parts.append(f"Custom module: {name}\n{prompt_context}{hint_text}".strip())
    return "\n\n".join(parts)


def _build_custom_module_prior_by_category(
    matched_specs: List[Dict[str, object]],
) -> Dict[str, float]:
    priors: Dict[str, float] = {}
    for spec in matched_specs:
        raw_priors = spec.get("mutator_prior_by_category", {})
        if not isinstance(raw_priors, dict):
            continue
        for category, raw_value in raw_priors.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            priors[str(category)] = priors.get(str(category), 0.0) + min(value, 2.0)
    return priors


def _split_plugin_paths(plugin_paths: str) -> List[str]:
    if not plugin_paths:
        return []
    normalized = plugin_paths.replace(";", os.pathsep)
    return [item.strip() for item in normalized.split(os.pathsep) if item.strip()]


def _import_custom_mutator_plugin(plugin_path: str) -> Any:
    candidate = pathlib.Path(plugin_path).expanduser()
    if candidate.exists():
        module_name = f"_rc_custom_mutator_{hashlib.sha1(str(candidate).encode('utf-8')).hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load custom mutator plugin from {candidate}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(plugin_path)


def _normalize_custom_mutator_result(raw_result: object) -> List[Tuple[Mutator, float]]:
    if raw_result is None:
        return []
    if isinstance(raw_result, dict) and "mutators" in raw_result:
        raw_result = raw_result["mutators"]
    if isinstance(raw_result, dict):
        items = list(raw_result.items())
    elif isinstance(raw_result, (list, tuple)):
        items = []
        for item in raw_result:
            if isinstance(item, tuple) and len(item) == 2:
                items.append(item)
            else:
                items.append((item, 0.1))
    else:
        items = [(raw_result, 0.1)]

    normalized: List[Tuple[Mutator, float]] = []
    for mutator, raw_prob in items:
        if not isinstance(mutator, Mutator):
            logger.warning("Ignoring custom mutator that is not a TVM Mutator: %s", mutator)
            continue
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            logger.warning("Ignoring custom mutator with invalid probability: %s", mutator)
            continue
        if prob <= 0.0:
            continue
        normalized.append((mutator, prob))
    return normalized


def _safe_mod_as_text(mod: IRModule) -> str:
    try:
        return mod.script(show_meta=True)
    except Exception:  # pylint: disable=broad-except
        try:
            return mod.script()
        except Exception:  # pylint: disable=broad-except
            return "<failed to script IRModule>"


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _extract_module_signature(mod: IRModule, target_kind: str) -> Dict[str, object]:
    mod_text = _safe_mod_as_text(mod)
    lowered = mod_text.lower()
    counts = {
        "for": int(len(re.findall(r"\bfor\b", mod_text))),
        "grid": int(mod_text.count("T.grid(")),
        "block": int(mod_text.count('with T.block("')),
        "init": int(mod_text.count("with T.init(")),
        "buffer": int(mod_text.count("T.Buffer(")),
        "alloc_buffer": int(mod_text.count("T.alloc_buffer(")),
    }
    grid_extents = []
    for grid_args in re.findall(r"T\.grid\(([^)]*)\)", mod_text):
        for raw_extent in re.findall(r"\b(?:T\.int64\()?([0-9]+)\)?", grid_args):
            try:
                extent = int(raw_extent)
            except ValueError:
                continue
            if 1 < extent <= 1_000_000:
                grid_extents.append(extent)
    numeric_extents = []
    for raw_extent in re.findall(r"\b(?:T\.int64\()?([0-9]+)\)?", mod_text):
        try:
            extent = int(raw_extent)
        except ValueError:
            continue
        if 1 < extent <= 1_000_000:
            numeric_extents.append(extent)
    keywords = sorted([kw for kw in TRANSFER_OPERATOR_KEYWORDS if kw in lowered])
    return {
        "target_kind": target_kind,
        "module_hash": hashlib.sha1(mod_text.encode("utf-8")).hexdigest(),
        "keywords": keywords,
        "counts": counts,
        "grid_extents": sorted(set(grid_extents)),
        "max_grid_extent": max(grid_extents) if grid_extents else 0,
        "numeric_extents": sorted(set(numeric_extents)),
        "max_numeric_extent": max(numeric_extents) if numeric_extents else 0,
        "module_excerpt": _truncate_text(mod_text, 1400),
    }


def _should_use_transfer_prior_only(signature: Optional[Dict[str, object]]) -> bool:
    if not signature:
        return False
    try:
        max_grid_extent = int(signature.get("max_grid_extent", 0))
        max_numeric_extent = int(signature.get("max_numeric_extent", 0))
    except (TypeError, ValueError):
        return False
    # Small CPU kernels are sensitive to replayed schedules and measurement noise.
    # Keep transfer knowledge as a mutator prior, but avoid forcing a prior trace.
    if 0 < max_grid_extent <= 64:
        return True
    return 0 < max_numeric_extent <= 4096


def _jaccard(lhs: Set[str], rhs: Set[str]) -> float:
    if not lhs and not rhs:
        return 1.0
    union = lhs | rhs
    if not union:
        return 0.0
    return len(lhs & rhs) / float(len(union))


def _count_similarity(lhs: Dict[str, int], rhs: Dict[str, int]) -> float:
    if not lhs and not rhs:
        return 1.0
    keys = sorted(set(lhs.keys()) | set(rhs.keys()))
    if not keys:
        return 0.0
    sims = []
    for key in keys:
        left_val = int(lhs.get(key, 0))
        right_val = int(rhs.get(key, 0))
        denom = float(max(left_val, right_val, 1))
        sims.append(1.0 - abs(left_val - right_val) / denom)
    return sum(sims) / len(sims)


def _score_transfer_similarity(
    current_signature: Dict[str, object],
    candidate_record: Dict[str, object],
) -> float:
    current_keywords = set(current_signature.get("keywords", []))
    candidate_keywords = set(candidate_record.get("keywords", []))
    keyword_score = _jaccard(current_keywords, candidate_keywords)
    count_score = _count_similarity(
        current_signature.get("counts", {}),
        candidate_record.get("counts", {}),
    )
    target_bonus = 1.0 if current_signature.get("target_kind") == candidate_record.get("target_kind") else 0.6
    return target_bonus * (0.55 * keyword_score + 0.45 * count_score)


def _extract_trace_mutator_hints(trace_text: str) -> Dict[str, float]:
    hints: Dict[str, float] = {}
    lowered = trace_text.lower()
    for category, patterns in TRACE_HINT_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            score += lowered.count(pattern.lower())
        if score > 0.0:
            hints[category] = score
    return hints


def _mutator_category(mutator: Mutator) -> Optional[str]:
    mutator_name = str(mutator)
    for category, pattern in MUTATOR_CATEGORY_PATTERNS.items():
        if pattern in mutator_name:
            return category
    return None


class _SizedMinHeap:
    def __init__(self, size_limit: int):
        self._size_limit = size_limit
        self._heap = []
        self._push_counter = 0  # strictly increasing for tie-breaking

    def push(self, sch: Schedule, score: float, measured_flag: bool) -> None:
        neg_score = -score
        self._push_counter += 1
        item = (neg_score, self._push_counter, sch, measured_flag)
        if len(self._heap) < self._size_limit:
            heapq.heappush(self._heap, item)
        else:
            worst_neg, _, _, _ = self._heap[0]
            if neg_score > worst_neg:
                return
            heapq.heapreplace(self._heap, item)

    def items_descending(self) -> List[Tuple[float, Schedule, bool]]:
        items = []
        for (neg, _, sch, meas) in self._heap:
            score = -neg
            items.append((score, sch, meas))
        items.sort(key=lambda x: x[0], reverse=True)
        return items


class MCTSNode:
    __slots__ = [
        "schedule",
        "parent",
        "children",
        "visits",
        "total_value",
        "depth",
    ]

    def __init__(
        self,
        schedule: Optional[Schedule],
        parent: Optional["MCTSNode"],
        depth: int,
    ):
        self.schedule = schedule
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.total_value = 0.0
        self.depth = depth

    def clone_tree(self) -> "MCTSNode":
        new_node = MCTSNode(self.schedule, None, self.depth)
        new_node.visits = self.visits
        new_node.total_value = self.total_value
        for ch in self.children:
            child_copy = ch.clone_tree()
            child_copy.parent = new_node
            new_node.children.append(child_copy)
        return new_node


class MCTSTuner:
    """
    Implements the core Monte Carlo Tree Search routines including but not limited to:
      - UCB-based node selection
      - expansions
      - rollouts/simulations
      - backprop
      - cost model predictions
    """

    def __init__(
        self,
        population_size: int,
        init_measured_ratio: float,
        init_min_unmeasured: int,
        max_fail_count: int,
        genetic_num_iters: int,
        genetic_mutate_prob: float,
        genetic_max_fail_count: int,
        num_empty_iters_before_early_stop: int,
        max_stale_iters: int,
        diversity_epsilon: float,
        max_stale_diversity_iters: int,
        trace_commit: bool,
        verbose: int,
        mcts_ucb_constant: float,
        mcts_max_depth: Optional[int],
        mcts_max_children_per_node: int,
        mcts_expansion_candidates: int,
        mcts_use_measured_feedback: bool,
        mcts_measured_feedback_power: float,
        mcts_use_root_prior_selection: bool,
        mcts_measure_selection_q_weight: float,
        mcts_measure_selection_diversity_weight: float,
        mcts_num_threads: int,
        mcts_num_rollouts_per_expansion: int,
        # references
        postprocs: List[Postproc],
        mutator_probs: Dict[Mutator, float],
        context: "TuneContext",
        cost_model: Optional["CostModel"],
        database: Optional["Database"],
        workload_key: Optional[Workload],
        use_llm: bool,
        llm_budget: int,
        llm_policy: Optional["LLMGuidancePolicy"] = None,
        llm_model_name: str = "",
        llm_history_depth: int = 4,
        llm_extra_prompt_context: str = "",
        custom_module_spec_path: str = "",
        target_kind: str = "",
        transfer_memory_path: str = "",
        transfer_top_k: int = 3,
        transfer_min_similarity: float = 0.2,
        transfer_warmstart_limit: int = 1,
        transfer_enable_max_trials: int = 0,
    ):
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob  # unused by standard MCTS
        self.genetic_max_fail_count = genetic_max_fail_count
        self.num_empty_iters_before_early_stop = num_empty_iters_before_early_stop
        self.max_stale_iters = max_stale_iters
        self.diversity_epsilon = diversity_epsilon
        self.max_stale_diversity_iters = max_stale_diversity_iters
        self.trace_commit = trace_commit
        self.verbose = verbose

        self.mcts_ucb_constant = mcts_ucb_constant
        self.mcts_max_depth = mcts_max_depth
        self.mcts_max_children_per_node = max(1, int(mcts_max_children_per_node))
        self.mcts_expansion_candidates = max(1, int(mcts_expansion_candidates))
        self.mcts_use_measured_feedback = bool(mcts_use_measured_feedback)
        self.mcts_measured_feedback_power = max(1.0, float(mcts_measured_feedback_power))
        self.mcts_use_root_prior_selection = bool(mcts_use_root_prior_selection)
        self.mcts_measure_selection_q_weight = min(
            1.0, max(0.0, float(mcts_measure_selection_q_weight))
        )
        self.mcts_measure_selection_diversity_weight = max(
            0.0, float(mcts_measure_selection_diversity_weight)
        )
        self.mcts_num_threads = mcts_num_threads
        self.mcts_num_rollouts_per_expansion = mcts_num_rollouts_per_expansion

        self._postprocs = postprocs
        self._mutator_probs = mutator_probs
        self._ctx = context
        self._cost_model = cost_model
        self._database = database
        self._workload_key = workload_key

        self._workload_cache: Dict[int, Workload] = {}
        self._mutator_failure_count: Dict[object, int] = {"total": 0}
        self._mutator_success_count: Dict[object, int] = {"total": 0}
        self._search_state: Optional["MCTSTuningState"] = None

        self.use_llm = use_llm
        self.llm_budget = llm_budget
        self.llm_policy = llm_policy
        self.llm_model_name = llm_model_name
        self.llm_history_depth = max(1, llm_history_depth)
        self.llm_extra_prompt_context = llm_extra_prompt_context.strip()
        self.custom_module_spec_path = custom_module_spec_path.strip()
        self.target_kind = target_kind
        self.transfer_memory_path = transfer_memory_path.strip()
        self.transfer_top_k = max(0, transfer_top_k)
        self.transfer_min_similarity = max(0.0, transfer_min_similarity)
        self.transfer_warmstart_limit = max(0, transfer_warmstart_limit)
        self.transfer_enable_max_trials = max(0, transfer_enable_max_trials)
        self._current_signature = (
            _extract_module_signature(self._ctx.mod, self.target_kind)
            if self._ctx is not None and self._ctx.mod is not None
            else None
        )
        if (
            self.transfer_memory_path
            and self.transfer_top_k > 0
            and self.transfer_warmstart_limit > 0
            and _should_use_transfer_prior_only(self._current_signature)
        ):
            self.transfer_warmstart_limit = 0
        self._matched_custom_module_specs = _match_custom_module_specs(
            self._current_signature,
            self.custom_module_spec_path,
        )
        self._custom_module_prior_by_category = _build_custom_module_prior_by_category(
            self._matched_custom_module_specs
        )
        self._custom_module_prompt_context = _build_custom_module_prompt_context(
            self._matched_custom_module_specs
        )
        self._retrieved_transfer_examples = self._load_transfer_examples()
        self._transfer_prior_by_category = self._build_transfer_prior_by_category()
        self._transfer_prompt_context = self._build_transfer_prompt_context()

    def attach_search_state(self, search_state: "MCTSTuningState") -> None:
        self._search_state = search_state

    def _load_transfer_examples(self) -> List[Dict[str, object]]:
        if (
            not self.transfer_memory_path
            or self.transfer_top_k <= 0
            or self._current_signature is None
            or not os.path.exists(self.transfer_memory_path)
        ):
            return []
        examples: List[Dict[str, object]] = []
        try:
            with open(self.transfer_memory_path, "r", encoding="utf-8") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record["similarity"] = _score_transfer_similarity(
                        self._current_signature,
                        record,
                    )
                    if record["similarity"] < self.transfer_min_similarity:
                        continue
                    examples.append(record)
        except OSError as err:
            logger.warning("Failed to load transfer memory '%s': %s", self.transfer_memory_path, err)
            return []
        examples.sort(
            key=lambda item: (
                float(item.get("similarity", 0.0)),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )
        retrieved = examples[: self.transfer_top_k]
        if retrieved and self.verbose >= 1:
            logger.warning(
                "Loaded %d transfer examples from '%s'. Best similarity=%.3f",
                len(retrieved),
                self.transfer_memory_path,
                float(retrieved[0].get("similarity", 0.0)),
            )
        return retrieved

    def _build_transfer_prior_by_category(self) -> Dict[str, float]:
        if not self._retrieved_transfer_examples:
            return {}
        raw_weights: Dict[str, float] = {}
        for example in self._retrieved_transfer_examples:
            similarity = float(example.get("similarity", 0.0))
            score = float(example.get("score", 0.0))
            quality = similarity * max(1.0, score)
            trace_hints = example.get("trace_mutator_hints", {})
            if not isinstance(trace_hints, dict):
                continue
            for category, value in trace_hints.items():
                raw_weights[category] = raw_weights.get(category, 0.0) + quality * float(value)
        if not raw_weights:
            return {}
        peak = max(raw_weights.values())
        if peak <= 1e-12:
            return {}
        return {category: 0.5 * (value / peak) for category, value in raw_weights.items()}

    def _build_transfer_prompt_context(self) -> str:
        if not self._retrieved_transfer_examples:
            return ""
        parts: List[str] = [
            "Retrieved prior schedules from transfer memory. These are examples, not constraints."
        ]
        for idx, example in enumerate(self._retrieved_transfer_examples, start=1):
            hints = example.get("trace_mutator_hints", {})
            hint_items = []
            if isinstance(hints, dict):
                hint_items = [f"{key}={float(value):.1f}" for key, value in sorted(hints.items())]
            parts.append(
                (
                    f"Prior {idx}: similarity={float(example.get('similarity', 0.0)):.3f}, "
                    f"score={float(example.get('score', 0.0)):.4f}, "
                    f"keywords={example.get('keywords', [])}, "
                    f"trace_hints={hint_items}\n"
                    f"Trace excerpt:\n{example.get('trace_excerpt', '<missing trace excerpt>')}"
                )
            )
        return "\n\n".join(parts)

    def _combined_llm_context(self) -> str:
        parts = []
        if self.llm_extra_prompt_context:
            parts.append(self.llm_extra_prompt_context)
        if self._custom_module_prompt_context:
            parts.append(self._custom_module_prompt_context)
        if self._transfer_prompt_context:
            parts.append(self._transfer_prompt_context)
        return "\n\n".join(parts)

    def _transfer_factor(self, mutator: Mutator) -> float:
        category = _mutator_category(mutator)
        if category is None:
            return 1.0
        return (
            1.0
            + self._transfer_prior_by_category.get(category, 0.0)
            + self._custom_module_prior_by_category.get(category, 0.0)
        )

    def _record_transfer_example(self, sch: Schedule, run_sec: float) -> None:
        try:
            run_sec = float(run_sec)
        except (TypeError, ValueError):
            return
        if not self.transfer_memory_path or run_sec <= 0.0:
            return
        try:
            os.makedirs(os.path.dirname(self.transfer_memory_path) or ".", exist_ok=True)
            signature = _extract_module_signature(sch.mod, self.target_kind)
            trace_text = str(sch.trace)
            trace_json = sch.trace.as_json(remove_postproc=True)
            record = {
                "target_kind": self.target_kind,
                "module_hash": signature["module_hash"],
                "keywords": list(signature["keywords"]),
                "counts": {key: int(value) for key, value in signature["counts"].items()},
                "module_excerpt": signature["module_excerpt"],
                "trace_excerpt": _truncate_text(trace_text, 1000),
                "trace_json": trace_json,
                "trace_mutator_hints": {
                    key: float(value) for key, value in _extract_trace_mutator_hints(trace_text).items()
                },
                "run_sec": float(run_sec),
                "score": 1.0 / max(run_sec, 1e-12),
            }
            with open(self.transfer_memory_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError as err:
            logger.warning("Failed to append transfer example to '%s': %s", self.transfer_memory_path, err)

    def explore(
        self,
        mcts_root: MCTSNode,
        population: List[Tuple[tvm.tir.Schedule, bool]],
        dynamic_pop_size: int,
        rand_state: int,
    ) -> List[Tuple[tvm.tir.Schedule, bool]]:
        logger.warning(
            "[DEBUG] explore() called with dynamic_pop_size=%d, genetic_num_iters=%d",
            dynamic_pop_size,
            self.genetic_num_iters
        )
        if not mcts_root or not mcts_root.children:
            logger.warning("explore(): Root is empty or has no children. Returning existing population.")
            return population
        total_expansions = 0
        for gen_iter in range(self.genetic_num_iters):
            new_children_count = 0
            fail_count = 0
            logger.warning("explore(): Starting generation %d ...", gen_iter)
            while new_children_count < dynamic_pop_size:
                # 1) UCB selection
                leaf = self._select(mcts_root)
                if leaf is None:
                    # means we couldn't pick a leaf at all
                    logger.warning(
                        "explore(): MCTS: Leaf is None in selection => break expansions."
                    )
                    break
                logger.warning(
                    "explore(): [gen=%d] Selected leaf node at depth=%d with %d children",
                    gen_iter, leaf.depth, len(leaf.children)
                )
                new_node = self._expand(leaf, rand_state)
                if new_node is None:
                    fail_count += 1
                    logger.warning(
                        "explore(): Failed to expand leaf at depth=%d (fail_count=%d)",
                        leaf.depth, fail_count
                    )
                    if fail_count >= self.genetic_max_fail_count:
                        logger.warning(
                            "explore(): Too many expansion failures => break expansions."
                        )
                        break
                    continue
                logger.warning(
                    "explore(): Successfully expanded leaf at depth=%d => new node at depth=%d; "
                    "now leaf has %d children total",
                    leaf.depth, new_node.depth, len(leaf.children)
                )
                value = self._simulate_node(new_node, rand_state)
                logger.warning(
                    "explore(): Simulation done for new node at depth=%d => value=%.4f",
                    new_node.depth, value
                )
                self._backprop(new_node, value)
                logger.warning(
                    "explore(): Backprop done => node visits=%d, total_value=%.4f",
                    new_node.visits, new_node.total_value
                )
                new_children_count += 1
                total_expansions += 1
            logger.warning(
                "explore(): [gen=%d] expansions=%d, fail_count=%d so far in this generation",
                gen_iter, new_children_count, fail_count
            )
        logger.warning(
            "explore(): All expansions complete => total_expansions=%d across %d generations.",
            total_expansions, self.genetic_num_iters
        )
        all_nodes = self._gather_tree_schedules(mcts_root)
        logger.warning(
            "explore(): Gathered %d total nodes (with schedules) from the MCTS tree.",
            len(all_nodes)
        )
        new_population = []
        for node in all_nodes:
            if node.schedule is not None:
                wl = self._commit_workload_cached(node.schedule)
                measured_flag = (wl is not None) and (wl in self._measured_workloads)
                new_population.append((node.schedule, measured_flag))
        logger.warning(
            "explore(): Returning a new population of size=%d (some may be measured).",
            len(new_population)
        )
        return new_population


    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Select a node for expansion or simulation using UCB-based traversal.
        """
        current = node
        while True:
            if current.parent is None: 
                if self.mcts_use_root_prior_selection:
                    current = self._select_root_child(current)
                    if current is None:
                        return None
                else:
                    unvisited = [child for child in current.children if child.visits == 0]
                    if unvisited:
                        current = unvisited[0]
                    else:
                        current = self._select_by_ucb(current)
                        if current is None:
                            return None
                continue    
            if current.visits == 0:
                return current
            if len(current.children) < self._max_children_for_node(current):
                return current
            unvisited_children = [child for child in current.children if child.visits == 0]
            if unvisited_children:
                return unvisited_children[0]
            next_child = self._select_by_ucb(current)
            if next_child is None:
                return None
            current = next_child

    def _select_root_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        if not node.children:
            return None
        schedules = [child.schedule for child in node.children if child.schedule is not None]
        priors_by_hash: Dict[int, float] = {}
        if schedules:
            priors = self._predict_normalized_score(schedules)
            prior_norm = self._normalize_score_list(priors)
            for child_sch, prior in zip(schedules, prior_norm):
                priors_by_hash[tvm.ir.structural_hash(child_sch.mod)] = prior
        best_child = None
        best_score = -float("inf")
        c = self._get_dynamic_ucb_constant()
        parent_visits = max(1, node.visits)
        for child in node.children:
            if child.schedule is not None:
                prior = priors_by_hash.get(tvm.ir.structural_hash(child.schedule.mod), 0.0)
            else:
                prior = 0.0
            if child.visits > 0:
                exploit = 0.75 * (child.total_value / child.visits) + 0.25 * prior
            else:
                exploit = prior
            explore = math.sqrt(math.log(parent_visits + 1.0) / (child.visits + 1.0))
            score = exploit + c * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def _normalize_score_list(scores: List[float]) -> List[float]:
        if not scores:
            return []
        lo = min(scores)
        hi = max(scores)
        if hi - lo <= 1e-12:
            return [0.0 for _ in scores]
        return [(score - lo) / (hi - lo) for score in scores]

    def _max_children_for_node(self, node: MCTSNode) -> int:
        if node.parent is None:
            return max(self.mcts_max_children_per_node, len(node.children))
        return self.mcts_max_children_per_node

    def _select_by_ucb(self, node: MCTSNode) -> Optional[MCTSNode]:
        best_child = None
        best_score = -float("inf")
        c = self._get_dynamic_ucb_constant()
        for ch in node.children:
            if ch.visits == 0:
                return ch
            exploit = ch.total_value / ch.visits
            explore = math.sqrt(max(1e-12, math.log(node.visits) / ch.visits))
            score = exploit + c * explore
            if score > best_score:
                best_score = score
                best_child = ch
        return best_child

    def _history_label(self, ancestor_distance: int) -> str:
        if ancestor_distance == 0:
            return "Current Schedule"
        if ancestor_distance == 1:
            return "Immediate Parent Schedule"
        if ancestor_distance == 2:
            return "Grandparent Schedule"
        return f"Ancestor Schedule (-{ancestor_distance})"

    def _build_schedule_snapshot(self, label: str, schedule: Schedule, score: float) -> str:
        try:
            mod_str = schedule.mod.script()
        except Exception:  # pylint: disable=broad-except
            mod_str = "<failed to script IR>"
        trace_str = str(schedule.trace)
        return (
            f"{label}:\n"
            f"{label}'s IR:\n{mod_str}\n\n"
            f"{label}'s Trace:\n{trace_str}\n\n"
            f"{label}'s Predicted Score by TVM's default cost model XGBoost: {score}\n"
        )

    def _build_historical_perf(self, leaf: MCTSNode) -> Optional[str]:
        historical_perf_parts: List[str] = []
        current = leaf
        distance = 0
        while current is not None and distance < self.llm_history_depth:
            if current.schedule is not None:
                scores = self._predict_normalized_score([current.schedule])
                score = scores[0] if scores else 0.0
                historical_perf_parts.append(
                    self._build_schedule_snapshot(
                        label=self._history_label(distance),
                        schedule=current.schedule,
                        score=score,
                    )
                )
            current = current.parent
            distance += 1
        if not historical_perf_parts:
            return None
        return "\n\n".join(historical_perf_parts)

    def _expand(self, leaf: MCTSNode, rand_state: int) -> Optional[MCTSNode]:
        """
        Expand a leaf node by applying multiple random mutations to its schedule.
        """
        if len(leaf.children) >= self._max_children_for_node(leaf):
            return None
        if self.mcts_max_depth is not None and leaf.depth >= self.mcts_max_depth:
            return None
        if leaf.schedule is None:
            return None
        can_use_llm = (
            self.use_llm
            and self.llm_policy is not None
            and self.llm_budget > 0
            and (2 <= leaf.depth)
        )
        if not can_use_llm:
            logger.warning(
                "Not using LLM (either disabled, no budget, or depth/children constraints). "
                "Using cost-model-guided mutation tournament."
            )
            new_sch = self._try_mcts_mutation(leaf.schedule, rand_state)
            if not new_sch:
                logger.warning("At line 625.")
                return None
            child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
            leaf.children.append(child)
            return child
        else:
            logger.warning(
                "LLM usage is enabled. Gathering historical info up to %d ancestor levels.",
                self.llm_history_depth,
            )
            new_sch = None
            try:
                historical_perf = self._build_historical_perf(leaf)
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning("Failed to gather historical info for ancestor chain: %s", str(e))
                historical_perf = None

            logger.warning("Invoking LLM policy to pick a sequence of mutators.")
            possible_mutator_names = [str(m) for m in self._mutator_probs.keys()]
            mutator_probs_dict = {str(mut): prob for mut, prob in self._mutator_probs.items()}
            chosen_mutator_names = self.llm_policy.pick_mutators(
                mod=leaf.schedule.mod,
                available_mutators=possible_mutator_names,
                historical_perf=historical_perf,
                available_mutator_probs=mutator_probs_dict,
                extra_context=self._combined_llm_context(),
            )
            if chosen_mutator_names is not None and len(chosen_mutator_names) > 0:
                logger.warning("LLM returned mutator names: '%s'", chosen_mutator_names)
                temp_sch = leaf.schedule
                for name in chosen_mutator_names:
                    chosen_mutator = None
                    for mut, _prob in self._mutator_probs.items():
                        if str(mut) == name:
                            chosen_mutator = mut
                            break
                    if chosen_mutator is None:
                        logger.warning(
                            "LLM mutator name '%s' did not match any known mutator. Fallback to random for this step.",
                            name
                        )
                        chosen_mutator = self._pick_random_mutator(rand_state)
                    maybe_new = self._apply_mutator_with_retry(temp_sch, chosen_mutator, rand_state)
                    if maybe_new is None:
                        logger.warning("Failed applying mutator '%s'. Will not continue sequence.", name)
                        break
                    temp_sch = maybe_new

                new_sch = temp_sch
                self.llm_budget -= 1
                logger.warning("LLM budget decremented. Remaining: %d", self.llm_budget)
            else:
                logger.warning("LLM did not produce a valid mutator list => fallback to a random single mutation.")
                new_sch = self._try_mcts_mutation(leaf.schedule, rand_state)
            if not new_sch:
                logger.warning("Failed to create a new schedule from chosen mutators. Expansion returning None.")
                return None

            child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
            leaf.children.append(child)
            logger.warning(
                "Successfully expanded leaf using %s approach. New child node at depth %d.",
                "LLM-based" if chosen_mutator_names else "random",
                child.depth
            )
            return child


    def _simulate_node(self, node: MCTSNode, rand_state: int) -> float:
        """
        Run rollout/simulation from a newly expanded node
        """
        if (self.mcts_num_rollouts_per_expansion <= 1) and (self.mcts_num_threads <= 1):
            return self._rollout(node.schedule, node.depth, rand_state)

        results = []
        if (self.mcts_num_threads > 1) and (self.mcts_num_rollouts_per_expansion > 1):
            with ThreadPoolExecutor(max_workers=self.mcts_num_threads) as executor:
                futures = [
                    executor.submit(self._rollout, node.schedule, node.depth, rand_state)
                    for _ in range(self.mcts_num_rollouts_per_expansion)
                ]
                for f in as_completed(futures):
                    results.append(f.result())
        else:
            for _ in range(self.mcts_num_rollouts_per_expansion):
                results.append(self._rollout(node.schedule, node.depth, rand_state))
        if results:
            return sum(results) / len(results)
        return 0.0

    def _backprop(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate the 'value' up the tree, incrementing visits and
        adding 'value' to total_value.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _rollout(self, schedule: Schedule, depth: int, rand_state: int) -> float:
        new_sch = self._replay_schedule(schedule.trace, rand_state)
        if new_sch is None:
            return 0.0
        cur_depth = depth
        while (self.mcts_max_depth is None) or (cur_depth < self.mcts_max_depth):
            cur_depth += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                logger.warning("[_rollout] No mutator found (mut is None). Breaking from rollout loop.")
                break
            try:
                mutated_trace = mut.apply(new_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                mutated_trace = None
            if mutated_trace is None:
                logger.warning("[_rollout] mutated_trace is None after apply(). Stopping mutations.")
                break
            maybe_new = self._replay_schedule(mutated_trace, rand_state)
            if maybe_new is None:
                logger.warning("[_rollout] Replaying the mutated trace returned None. Stopping mutations.")
                break
            new_sch = maybe_new
            # logger.warning(
            #     f"[_rollout] Successfully replayed mutated trace. Now at rollout depth={cur_depth}."
            # )
        if not self._cost_model or not self._database:
            logger.warning(
                f"[_rollout] No cost_model or no database found. Returning random fallback score."
            )
            return random.random()  # fallback random
        arg_info = ArgInfo.from_entry_func(new_sch.mod, remove_preproc=True)
        candidate = MeasureCandidate(new_sch, arg_info)
        preds = self._cost_model.predict(self._ctx, [candidate])
        if preds:
            logger.warning(
                f"[_rollout] Final cost-model prediction. Returning this from rollout."
            )
            return max(0.0, preds[0])
        return 0.0

    def gather_unmeasured_leaves(self, node: MCTSNode) -> List[MCTSNode]:
        stack = [node]
        leaves = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None and not nd.children:
                wl = None
                wl = self._commit_workload_cached(nd.schedule)
                if wl not in self._measured_workloads:
                    leaves.append(nd)
            else:
                stack.extend(nd.children)
        return leaves

    def pick_unmeasured_best_leaves(self, root: MCTSNode, batch_size: int) -> List[Schedule]:
        leaves = self.gather_unmeasured_leaves(root)
        if not leaves:
            return []
        scored = []
        for nd in leaves:
            q_val = (nd.total_value / nd.visits) if nd.visits > 0 else 0.0
            scored.append((nd, q_val))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored[:batch_size]
        return [node.schedule for (node, _) in top_nodes]

    def _gather_tree_schedules(self, root: MCTSNode) -> List[MCTSNode]:
        stack = [root]
        out_nodes = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None:
                out_nodes.append(nd)
            stack.extend(nd.children)
        return out_nodes

    def find_node_by_workload(self, root: Optional[MCTSNode], workload: Workload) -> Optional[MCTSNode]:
        if root is None:
            return None
        stack = [root]
        while stack:
            node = stack.pop()
            if node.schedule is not None:
                node_workload = self._commit_workload_cached(node.schedule)
                if node_workload == workload:
                    return node
            stack.extend(node.children)
        return None


    def _replay_schedule(self, trace: Optional[Trace], rand_state: int) -> Optional[Schedule]:
        if not self._ctx or not self._ctx.mod:
            return None
        mod = self._ctx.mod
        if trace is None:
            try:
                sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            except (InvalidScheduleError, tvm.TVMError):
                return None
            sch.enter_postproc()
            if not self._apply_postprocs(sch):
                return None
            return sch
        try:
            sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None
        sch.enter_postproc()
        if not self._apply_postprocs(sch):
            return None
        return sch

    def _apply_postprocs(self, sch: Schedule) -> bool:
        if not self._postprocs:
            return True
        ffi_postproc = getattr(_ffi_api, "SearchStrategyApplyPostprocs", None)
        if ffi_postproc is not None:
            try:
                return bool(ffi_postproc(sch, self._postprocs))
            except Exception:
                pass
        for proc in self._postprocs:
            try:
                if not proc.apply(sch):
                    return False
            except (InvalidScheduleError, tvm.TVMError):
                return False
        return True


    def _pick_random_mutator(self, rand_state: int) -> Optional[Mutator]:
        if not self._mutator_probs:
            return None
        weighted_mutators: List[Tuple[Mutator, float]] = []
        total_p = 0.0
        for mut, base_prob in self._mutator_probs.items():
            failures = float(self._mutator_failure_count.get(mut, 0))
            successes = float(self._mutator_success_count.get(mut, 0))
            success_rate = (successes + 1.0) / (successes + failures + 2.0)
            adjusted_prob = base_prob * success_rate * self._transfer_factor(mut)
            if adjusted_prob > 0.0:
                weighted_mutators.append((mut, adjusted_prob))
                total_p += adjusted_prob
        if not weighted_mutators or total_p <= 0.0:
            weighted_mutators = list(self._mutator_probs.items())
            total_p = sum(prob for _, prob in weighted_mutators)
        rng = random.Random((rand_state or 1) + self._mutator_failure_count["total"])
        r = rng.random() * total_p
        s = 0.0
        for mut, p in weighted_mutators:
            s += p
            if r <= s:
                return mut
        return weighted_mutators[0][0]

    def _try_mcts_mutation(self, parent_sch: Schedule, rand_state: int) -> Optional[Schedule]:
        attempts = 0
        candidate_pool: List[Tuple[Schedule, Optional[Mutator], Workload]] = []
        candidate_hashes: Set[int] = set()
        max_attempts = max(1, self.genetic_max_fail_count) * self.mcts_expansion_candidates
        while attempts <= max_attempts and len(candidate_pool) < self.mcts_expansion_candidates:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                child_sch = self._replay_schedule(parent_sch.trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    shash = tvm.ir.structural_hash(child_sch.mod)
                    if wl not in self._seen_workloads and shash not in candidate_hashes:
                        candidate_hashes.add(shash)
                        candidate_pool.append((child_sch, None, wl))
                continue
            try:
                new_trace = mut.apply(parent_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                new_trace = None
            if new_trace is None:
                self._mutator_failure_count[mut] = self._mutator_failure_count.get(mut, 0) + 1
            else:
                child_sch = self._replay_schedule(new_trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    shash = tvm.ir.structural_hash(child_sch.mod)
                    if wl not in self._seen_workloads and shash not in candidate_hashes:
                        candidate_hashes.add(shash)
                        candidate_pool.append((child_sch, mut, wl))
        if not candidate_pool:
            return None
        chosen_sch, chosen_mut, chosen_wl = self._pick_best_expansion_candidate(candidate_pool)
        self._seen_workloads.add(chosen_wl)
        if chosen_mut is not None:
            self._mutator_success_count[chosen_mut] = (
                self._mutator_success_count.get(chosen_mut, 0) + 1
            )
            self._mutator_success_count["total"] += 1
        return chosen_sch

    def _pick_best_expansion_candidate(
        self,
        candidates: List[Tuple[Schedule, Optional[Mutator], Workload]],
    ) -> Tuple[Schedule, Optional[Mutator], Workload]:
        if len(candidates) == 1:
            return candidates[0]
        schedules = [sch for sch, _, _ in candidates]
        scores = self._predict_normalized_score(schedules)
        best_idx = 0
        best_score = -float("inf")
        for idx, score in enumerate(scores):
            if score > best_score:
                best_idx = idx
                best_score = score
        if self.verbose >= 2:
            logger.debug(
                "MCTS expansion tournament picked candidate %d/%d with predicted score %.4f",
                best_idx + 1,
                len(candidates),
                best_score,
            )
        return candidates[best_idx]

    def _apply_mutator_with_retry(
        self,
        parent_sch: tvm.tir.Schedule,
        chosen_mutator: Mutator,
        rand_state: int
    ) -> Optional[tvm.tir.Schedule]:
        attempts = 0
        while attempts <= self.genetic_max_fail_count:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            try:
                new_trace = chosen_mutator.apply(parent_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                new_trace = None
            if new_trace is None:
                self._mutator_failure_count[chosen_mutator] = (
                    self._mutator_failure_count.get(chosen_mutator, 0) + 1
                )
            else:
                child_sch = self._replay_schedule(new_trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        self._mutator_success_count[chosen_mutator] = (
                            self._mutator_success_count.get(chosen_mutator, 0) + 1
                        )
                        self._mutator_success_count["total"] += 1
                        return child_sch
        return None
    
    def _get_dynamic_ucb_constant(self) -> float:
        c0 = self.mcts_ucb_constant
        alpha = 0.99
        scaling = 150.0
        current_trials = 0
        if self._search_state is not None:
            current_trials = self._search_state.trial_count
        exponent = float(current_trials) / scaling
        c_dynamic = c0 * (alpha ** exponent)
        if self.verbose >= 2:
            logger.debug(
                f"[_get_dynamic_ucb_constant] trial_count={current_trials}, c_dynamic={c_dynamic:.4f}"
            )
        return c_dynamic


    def _commit_workload_cached(self, sch: Schedule) -> Optional[Workload]:
        if self._database is None:
            return None
        wl = getattr(sch, "_cached_wl", None)
        if wl is not None:
            return wl
        shash = tvm.ir.structural_hash(sch.mod)
        wl = self._workload_cache.get(shash)
        if wl is None:
            wl = self._database.commit_workload(sch.mod)
            self._workload_cache[shash] = wl
        sch._cached_wl = wl
        return wl

    def _predict_normalized_score(self, schedules: List[Schedule]) -> List[float]:
        if not schedules or not self._cost_model or not self._database:
            return [0.0] * len(schedules)
        cands = []
        for sch in schedules:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            cands.append(MeasureCandidate(sch, arg_info))
        scores = self._cost_model.predict(self._ctx, cands)
        return [max(0.0, sc) for sc in scores]

    @property
    def _measured_workloads(self) -> Set[Workload]:
        """
        The set of workload keys that have been actually measured on hardware.
        """
        if self._search_state is not None:
            return self._search_state.measured_workloads
        return set()

    @property
    def _seen_workloads(self) -> Set[Workload]:
        """
        The set of workload keys encountered in generated schedules.
        """
        if self._search_state is not None:
            return self._search_state.seen_workloads
        return set()


class MCTSTuningState:
    """
    MCTSTuningState tracks the MCTS root, population, number of trials used,
    best score, etc. The MCTSTuner performs expansions and rollouts; 
    MCTSTuningState decides how to handle each iteration.
    """

    def __init__(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"],
        cost_model: Optional["CostModel"],
        context: "TuneContext",
        tuner: MCTSTuner,
    ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_spaces = design_spaces
        self.database = database
        self.cost_model = cost_model
        self.context = context
        self.tuner = tuner
        self.tuner.attach_search_state(self)

        self.mod = context.mod
        self.workload_key = None
        if self.database and self.mod is not None:
            self.workload_key = self.database.commit_workload(self.mod)
            self.tuner._workload_key = self.workload_key

        self.trial_count = 0
        self.num_empty_iters = 0
        self.used_init_population = False
        self.population: List[Tuple[Schedule, bool]] = []
        self.mcts_root: Optional[MCTSNode] = None

        self.measured_workloads: Set[Workload] = set()
        self.seen_workloads: Set[Workload] = set()

        self.best_score_so_far = -float("inf")
        self.best_run_sec_so_far = float("inf")
        self.stale_iter_count = 0
        self.stale_diversity_count = 0
        self.diversity_history: List[float] = []
        self.score_history: List[float] = []
        self.dynamic_pop_size = self.tuner.population_size

        rs = context.rand_state
        self.rand_state = rs if rs is not None else 1
        if self.rand_state == 0:
            self.rand_state = 1

    def reset(self) -> None:
        """
        Called from MCTSSearch.post_tuning().
        """

    
    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        if self.tuner.verbose >= 1:
            logger.warning(
                "[DEBUG] Enter generate_measure_candidates: trial_count=%d, max_trials=%d",
                self.trial_count, self.max_trials
            )
        if self.trial_count >= self.max_trials:
            return None
        remaining = self.max_trials - self.trial_count
        batch_size = min(remaining, self.num_trials_per_iter)
        if batch_size <= 0:
            return None
        if not self.used_init_population:
            init_pop = self._init_population()
            if not init_pop:
                return None
            self.mcts_root = MCTSNode(schedule=None, parent=None, depth=0)
            for (sch, is_measured) in init_pop:
                child = MCTSNode(schedule=sch, parent=self.mcts_root, depth=1)
                self.mcts_root.children.append(child)
            self.population = init_pop
            self.used_init_population = True
            if self.tuner.verbose >= 1:
                logger.warning(
                    "generate_measure_candidates: MCTS: Initialized root with %d child schedules.",
                    len(init_pop)
                )

        self.population = self.tuner.explore(
            mcts_root=self.mcts_root,
            population=self.population,
            dynamic_pop_size=self.dynamic_pop_size,
            rand_state=self.rand_state,
        )
        if not self.population:
            self.num_empty_iters += 1
            logger.warning(
                "generate_measure_candidates: MCTS: explore() returned empty => empty iters=%d",
                self.num_empty_iters
            )
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                if self.tuner.verbose >= 1:
                    logger.warning("generate_measure_candidates: MCTS: Stopping early => repeated empty iters.")
                return None
            return None
        logger.warning(
            "generate_measure_candidates: MCTS: population size=%d before eps-greedy picking",
            len(self.population)
        )
        cands_sch = self._pick_unmeasured_eps_greedy(self.population, batch_size, self.rand_state)
        if not cands_sch:
            self.num_empty_iters += 1
            logger.warning(
                "generate_measure_candidates: MCTS: no unmeasured schedules => empty iters=%d",
                self.num_empty_iters
            )
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                if self.tuner.verbose >= 1:
                    logger.warning("generate_measure_candidates: stopping early => repeated empty iters.")
                return None
            return None
        logger.warning(
            "generate_measure_candidates: [DEBUG] Eps-greedy picked %d schedules for measurement (batch_size=%d).",
            len(cands_sch), batch_size
        )
        measure_cands: List[MeasureCandidate] = []
        for sch in cands_sch:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            measure_cands.append(MeasureCandidate(sch, arg_info))
        logger.warning(
                "generate_measure_candidates: [DEBUG] MCTS => returning %d cands; trial_count=%d, "
                "batch_size_requested=%d, used_init_population=%s",
                len(measure_cands),
                self.trial_count,
                batch_size,
                str(self.used_init_population),
            )
        return measure_cands


    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        if self.database is None:
            logger.warning("database is not defined, skipping MCTS measure update.")
            return

        num_measured_now = 0
        best_run_sec = float("inf")
        measured_feedback: List[Tuple[Workload, float]] = []
        for cand, res in zip(measure_candidates, results):
            sch = cand.sch
            mod = sch.mod
            wl = self.database.commit_workload(mod)
            if res.run_secs and all(t >= 0 for t in res.run_secs):
                run_sec = sum(res.run_secs) / len(res.run_secs)
                if run_sec < best_run_sec:
                    best_run_sec = run_sec
                self.measured_workloads.add(wl)
                self._mark_schedule_measured(sch)
                self.tuner._record_transfer_example(sch, run_sec)
                if self.tuner.mcts_use_measured_feedback:
                    measured_feedback.append((wl, run_sec))
                num_measured_now += 1
        self.trial_count += num_measured_now
        if self.tuner.mcts_use_measured_feedback and measured_feedback:
            reward_baseline = min(self.best_run_sec_so_far, best_run_sec)
            self._backprop_measured_feedback_batch(measured_feedback, reward_baseline)
        if best_run_sec < float("inf"):
            if best_run_sec < self.best_run_sec_so_far:
                self.best_run_sec_so_far = best_run_sec
            new_score = 1.0 / best_run_sec
            self.score_history.append(new_score)
            if new_score > self.best_score_so_far + 1e-12:
                self.best_score_so_far = new_score
                self.stale_iter_count = 0
            else:
                self.stale_iter_count += 1
                if self.stale_iter_count >= self.tuner.max_stale_iters and self.tuner.verbose >= 1:
                    logger.warning(
                        "notifu_runner_results: MCTS: No improvement => stopping early (stale_iter=%d).",
                        self.stale_iter_count
                    )
        else:
            self.score_history.append(0.0)

        # # Check population diversity
        # if self.population:
        #     pop_scores = self._predict_population_scores(self.population)
        #     diversity = self._check_population_diversity(pop_scores)
        #     if diversity < self.tuner.diversity_epsilon:
        #         self.stale_diversity_count += 1
        #         if self.tuner.verbose >= 1:
        #             logger.info(
        #                 "MCTS: Pop diversity=%.6f < threshold => stale_diversity_count=%d",
        #                 diversity, self.stale_diversity_count
        #             )
        #         if self.stale_diversity_count >= self.tuner.max_stale_diversity_iters:
        #             if self.tuner.verbose >= 1:
        #                 logger.info("MCTS: Population too homogeneous => stopping early.")
        #     else:
        #         self.stale_diversity_count = 0

        # Adaptive population resizing example
        #if self.population:
        #    if self.stale_diversity_count > 0:
        #        old_size = self.dynamic_pop_size
        #        self.dynamic_pop_size = max(10, int(self.dynamic_pop_size * 0.9))
        #        if self.tuner.verbose >= 1:
        #            logger.info(
        #                "MCTS: Adaptive pop resize: %d -> %d",
        #                old_size, self.dynamic_pop_size
        #            )
        #    else:
        #        self.dynamic_pop_size = min(
        #            self.tuner.population_size,
        #            self.dynamic_pop_size + 5
        #        )

        if self.tuner.verbose >= 1:
            logger.warning(
                "MCTS: notify_runner_results => measured=%d, total=%d, stale_iter=%d, div_stale=%d",
                num_measured_now, self.trial_count, self.stale_iter_count, self.stale_diversity_count
            )

    def _measured_reward(self, run_sec: float, reward_baseline: float) -> float:
        run_sec = float(run_sec)
        reward_baseline = float(reward_baseline)
        if run_sec <= 0.0:
            return 0.0
        if reward_baseline < float("inf"):
            reward = (reward_baseline / run_sec) ** self.tuner.mcts_measured_feedback_power
        else:
            reward = 1.0
        return max(0.0, min(3.0, reward))

    def _backprop_measured_feedback_batch(
        self,
        measured_feedback: List[Tuple[Workload, float]],
        reward_baseline: float,
    ) -> None:
        for workload, run_sec in measured_feedback:
            self._backprop_measured_feedback(workload, run_sec, reward_baseline)

    def _backprop_measured_feedback(
        self,
        workload: Workload,
        run_sec: float,
        reward_baseline: float,
    ) -> None:
        node = self.tuner.find_node_by_workload(self.mcts_root, workload)
        if node is None:
            return
        reward = self._measured_reward(run_sec, reward_baseline)
        if reward <= 0.0:
            return
        self.tuner._backprop(node, reward)
        if self.tuner.verbose >= 2:
            logger.debug(
                "MCTS measured feedback: workload=%s run_sec=%.6e reward=%.4f",
                workload,
                run_sec,
                reward,
            )

    def _init_population(self) -> List[Tuple[Schedule, bool]]:
        """
        Combine schedules from DB, transfer warm starts, and random design-space samples.
        """
        num_measured_wanted = int(self.tuner.population_size * self.tuner.init_measured_ratio)
        measured_from_db = self._pick_best_from_database(num_measured_wanted)
        need_unmeasured = max(
            self.tuner.population_size - len(measured_from_db),
            self.tuner.init_min_unmeasured
        )
        transfer_warm_start = self._sample_transfer_population(need_unmeasured)
        need_rand = max(0, need_unmeasured - len(transfer_warm_start))
        unmeasured_rand = self._sample_init_population(need_rand)
        logger.warning(
            "[MCTS init_pop] from DB: %d, from transfer: %d, from random: %d, population_size=%d, init_min_unmeasured=%d",
            len(measured_from_db),
            len(transfer_warm_start),
            len(unmeasured_rand),
            self.tuner.population_size,
            self.tuner.init_min_unmeasured
        )
        combined = [(sch, True) for sch in measured_from_db] + \
                   [(sch, False) for sch in transfer_warm_start] + \
                   [(sch, False) for sch in unmeasured_rand]
        random.shuffle(combined)
        if len(combined) > self.tuner.population_size:
            combined = combined[: self.tuner.population_size]
        for (sch, measured_flag) in combined:
            wl = self.tuner._commit_workload_cached(sch)
            self.seen_workloads.add(wl)
            if measured_flag:
                self.measured_workloads.add(wl)
        return combined

    def _pick_best_from_database(self, num: int) -> List[Schedule]:
        if num <= 0 or not self.database:
            return []
        out = []
        top_records = self.database.get_top_k(self.workload_key, num)
        for rec in top_records:
            sch = self._replay_schedule(rec.trace)
            if sch is not None:
                wl = self.tuner._commit_workload_cached(sch)
                # wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
        return out

    def _replay_schedule(self, trace: Optional[Trace]) -> Optional[Schedule]:
        if not trace or not self.context or not self.context.mod:
            return None
        mod = self.context.mod
        try:
            sch = Schedule(mod, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None
        sch.enter_postproc()
        for proc in self.tuner._postprocs:
            try:
                if not proc.apply(sch):
                    return None
            except (InvalidScheduleError, tvm.TVMError):
                return None
        return sch

    def _replay_schedule_json(self, trace_json: object) -> Optional[Schedule]:
        if trace_json is None or not self.context or not self.context.mod:
            return None
        mod = self.context.mod
        try:
            sch = Schedule(mod, debug_mask="all")
            Trace.apply_json_to_schedule(trace_json, sch)
        except (InvalidScheduleError, tvm.TVMError, TypeError, ValueError):
            return None
        sch.enter_postproc()
        for proc in self.tuner._postprocs:
            try:
                if not proc.apply(sch):
                    return None
            except (InvalidScheduleError, tvm.TVMError):
                return None
        return sch

    def _sample_init_population(self, num: int) -> List[Schedule]:
        out = []
        fails = 0
        n_spaces = len(self.design_spaces)
        while len(out) < num and fails < self.tuner.max_fail_count:
            idx = random.randint(0, n_spaces - 1)
            base_sch = self.design_spaces[idx]
            sch = self._replay_schedule(base_sch.trace)
            if sch is not None:
                wl = self.tuner._commit_workload_cached(sch)
                # wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
                    self.seen_workloads.add(wl)
                else:
                    fails += 1
            else:
                fails += 1
        return out

    def _sample_transfer_population(self, num: int) -> List[Schedule]:
        out: List[Schedule] = []
        if num <= 0:
            return out
        if (
            self.tuner.transfer_enable_max_trials > 0
            and self.max_trials > self.tuner.transfer_enable_max_trials
        ):
            return out
        if self.tuner.transfer_warmstart_limit > 0:
            num = min(num, self.tuner.transfer_warmstart_limit)
        for example in self.tuner._retrieved_transfer_examples:
            if len(out) >= num:
                break
            sch = self._replay_schedule_json(example.get("trace_json"))
            if sch is None:
                continue
            wl = self.tuner._commit_workload_cached(sch)
            if wl in self.seen_workloads:
                continue
            out.append(sch)
            self.seen_workloads.add(wl)
        return out
    
    def _pick_unmeasured_eps_greedy(
        self,
        schedules_with_flags: List[Tuple[tvm.tir.Schedule, bool]],
        total_needed: int,
        rand_state: int
    ) -> List[tvm.tir.Schedule]:
        logger.warning(
            "[DEBUG] _pick_unmeasured_eps_greedy called with total_needed=%d, eps_greedy=%.3f",
            total_needed, 0.05
        )
        unmeasured = []
        for (sch, measured_flag) in schedules_with_flags:
            if not measured_flag:
                unmeasured.append(sch)
        logger.warning("[DEBUG] Found %d unmeasured schedules.", len(unmeasured))
        if not unmeasured:
            return []
        preds = self.tuner._predict_normalized_score(unmeasured)
        logger.warning("[DEBUG] Computed cost-model predictions for %d unmeasured schedules.", len(preds))
        q_scores = [self._node_q_score(sch) for sch in unmeasured]
        pred_norm = self._normalize_scores(preds)
        q_norm = self._normalize_scores(q_scores)
        q_weight = self.tuner.mcts_measure_selection_q_weight
        scored = [
            (sch, (1.0 - q_weight) * pred + q_weight * q_score)
            for sch, pred, q_score in zip(unmeasured, pred_norm, q_norm)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.warning(
            "[DEBUG] Top schedule after sorting has selection score=%.4f if the list is non-empty.",
            scored[0][1] if scored else -1.0
        )
        n_total = min(total_needed, len(scored))
        n_rand = int(round(n_total * 0.05))
        n_top = n_total - n_rand
        logger.warning(
            "[DEBUG] Eps-greedy selection: total_needed=%d => n_top=%d, n_rand=%d",
            n_total, n_top, n_rand
        )
        top_part = scored[:n_top]
        leftover = scored[n_top:]
        random_schedules = []
        if leftover and n_rand > 0:
            random.seed(rand_state)
            n_rand = min(n_rand, len(leftover))
            random_part = random.sample(leftover, n_rand)
            random_schedules = [sch for (sch, _) in random_part]
        top_schedules = self._select_diverse_top_schedules(top_part, n_top)
        combined = top_schedules + random_schedules
        logger.warning(
            "[DEBUG] _pick_unmeasured_eps_greedy => returning %d schedules => %d top + %d random",
            len(combined), len(top_schedules), len(random_schedules)
        )
        return combined

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        lo = min(scores)
        hi = max(scores)
        if hi - lo <= 1e-12:
            return [0.0 for _ in scores]
        return [(score - lo) / (hi - lo) for score in scores]

    def _node_q_score(self, sch: Schedule) -> float:
        wl = self.tuner._commit_workload_cached(sch)
        node = self.tuner.find_node_by_workload(self.mcts_root, wl)
        if node is None or node.visits <= 0:
            return 0.0
        return node.total_value / node.visits

    def _select_diverse_top_schedules(
        self,
        scored: List[Tuple[Schedule, float]],
        n_top: int,
    ) -> List[Schedule]:
        if n_top <= 0:
            return []
        if self.tuner.mcts_measure_selection_diversity_weight <= 0.0:
            return [sch for sch, _ in scored[:n_top]]
        remaining = list(scored)
        selected: List[Tuple[Schedule, Set[str]]] = []
        while remaining and len(selected) < n_top:
            best_idx = 0
            best_score = -float("inf")
            for idx, (sch, base_score) in enumerate(remaining):
                hints = self._schedule_hint_set(sch)
                similarity = 0.0
                if selected:
                    similarity = max(_jaccard(hints, prior_hints) for _, prior_hints in selected)
                adjusted = (
                    base_score
                    - self.tuner.mcts_measure_selection_diversity_weight * similarity
                )
                if adjusted > best_score:
                    best_idx = idx
                    best_score = adjusted
            chosen_sch, _ = remaining.pop(best_idx)
            selected.append((chosen_sch, self._schedule_hint_set(chosen_sch)))
        return [sch for sch, _ in selected]

    @staticmethod
    def _schedule_hint_set(sch: Schedule) -> Set[str]:
        hints = set(_extract_trace_mutator_hints(str(sch.trace)).keys())
        if not hints:
            hints.add(f"trace_len_{len(sch.trace.insts) // 5}")
        return hints

    def _mark_schedule_measured(self, sch: Schedule):
        wl = self.database.commit_workload(sch.mod)
        self.measured_workloads.add(wl)
        for i, (pop_sch, was_measured) in enumerate(self.population):
            if pop_sch == sch and not was_measured:
                self.population[i] = (pop_sch, True)


    def _predict_population_scores(self, pop: List[Tuple[Schedule, bool]]) -> List[float]:
        schs = [p[0] for p in pop]
        if not schs:
            return []
        return self.tuner._predict_normalized_score(schs)

    def _check_population_diversity(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        mean_val = sum(scores) / len(scores)
        var = sum((s - mean_val) ** 2 for s in scores) / len(scores)
        cur_div = math.sqrt(var)
        self.diversity_history.append(cur_div)
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        avg_div = sum(self.diversity_history) / len(self.diversity_history)
        self.tuner.diversity_epsilon = 0.5 * avg_div
        return cur_div



@derived_object
class MCTSSearchPyFull(PySearchStrategy):
    def __init__(
        self,
        population_size: int = 3,
        init_measured_ratio: float = 0.0,
        init_min_unmeasured: int = 3,
        max_fail_count: int = 20,
        genetic_num_iters: int = 2,
        genetic_mutate_prob: float = 0.85,
        genetic_max_fail_count: int = 2,
        num_empty_iters_before_early_stop: int = 10,
        max_stale_iters: int = 60,
        diversity_epsilon: float = 1e-6,
        max_stale_diversity_iters: int = 30,
        trace_commit: bool = True,
        verbose: int = 2,
        # MCTS-specific:
        mcts_ucb_constant: float = 1.41,
        mcts_max_depth: Optional[int] = 500,
        mcts_max_children_per_node: int = 4,
        mcts_expansion_candidates: int = 4,
        mcts_use_measured_feedback: bool = True,
        mcts_measured_feedback_power: float = 1.0,
        mcts_use_root_prior_selection: bool = False,
        mcts_measure_selection_q_weight: float = 0.0,
        mcts_measure_selection_diversity_weight: float = 0.0,
        mcts_num_threads: int = 1,
        mcts_num_rollouts_per_expansion: int = 1,
        use_llm: bool = False,
        llm_budget: int = 1,
        llm_model_name: str = "",
        llm_history_depth: int = 4,
        llm_extra_prompt_context: str = "",
        custom_module_spec_path: str = "",
        custom_mutator_plugin_paths: str = "",
        transfer_memory_path: str = "",
        transfer_top_k: int = 3,
        transfer_min_similarity: float = 0.2,
        transfer_warmstart_limit: int = 1,
        transfer_enable_max_trials: int = 0,
    ) -> None:
        super().__init__()
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob
        self.genetic_max_fail_count = genetic_max_fail_count
        self.num_empty_iters_before_early_stop = num_empty_iters_before_early_stop
        self.max_stale_iters = max_stale_iters
        self.diversity_epsilon = diversity_epsilon
        self.max_stale_diversity_iters = max_stale_diversity_iters
        self.trace_commit = trace_commit
        self.verbose = verbose

        self.mcts_ucb_constant = mcts_ucb_constant
        self.mcts_max_depth = mcts_max_depth
        self.mcts_max_children_per_node = mcts_max_children_per_node
        self.mcts_expansion_candidates = mcts_expansion_candidates
        self.mcts_use_measured_feedback = mcts_use_measured_feedback
        self.mcts_measured_feedback_power = mcts_measured_feedback_power
        self.mcts_use_root_prior_selection = mcts_use_root_prior_selection
        self.mcts_measure_selection_q_weight = mcts_measure_selection_q_weight
        self.mcts_measure_selection_diversity_weight = mcts_measure_selection_diversity_weight
        self.mcts_num_threads = mcts_num_threads
        self.mcts_num_rollouts_per_expansion = mcts_num_rollouts_per_expansion
        self.use_llm = use_llm
        self.llm_budget = llm_budget
        self._ctx: Optional["TuneContext"] = None
        self._postprocs: List[Postproc] = []
        self._mutator_probs: Dict[Mutator, float] = {}
        self.state: Optional[MCTSTuningState] = None

        self.llm_model_name = llm_model_name
        self.llm_history_depth = llm_history_depth
        self.llm_extra_prompt_context = llm_extra_prompt_context.strip()
        self.custom_module_spec_path = custom_module_spec_path.strip()
        self.custom_mutator_plugin_paths = custom_mutator_plugin_paths.strip()
        self.transfer_memory_path = transfer_memory_path.strip()
        self.transfer_top_k = transfer_top_k
        self.transfer_min_similarity = transfer_min_similarity
        self.transfer_warmstart_limit = transfer_warmstart_limit
        self.transfer_enable_max_trials = transfer_enable_max_trials

    def _load_custom_mutator_plugins(self, context: "TuneContext") -> List[Tuple[Mutator, float]]:
        loaded: List[Tuple[Mutator, float]] = []
        for plugin_path in _split_plugin_paths(self.custom_mutator_plugin_paths):
            try:
                module = _import_custom_mutator_plugin(plugin_path)
                register = getattr(module, "register_mutators", None)
                if register is None:
                    register = getattr(module, "get_mutators", None)
                if register is None:
                    logger.warning(
                        "Custom mutator plugin '%s' has no register_mutators(context) function.",
                        plugin_path,
                    )
                    continue
                try:
                    raw_result = register(context)
                except TypeError:
                    raw_result = register()
                mutators = _normalize_custom_mutator_result(raw_result)
                for mutator, prob in mutators:
                    try:
                        mutator._initialize_with_tune_context(context)
                    except Exception as err:  # pylint: disable=broad-except
                        logger.warning("Custom mutator initialization failed for %s: %s", mutator, err)
                        continue
                    loaded.append((mutator, prob))
                logger.warning(
                    "Loaded %d custom mutator(s) from plugin '%s'.",
                    len(mutators),
                    plugin_path,
                )
            except Exception as err:  # pylint: disable=broad-except
                logger.warning("Failed to load custom mutator plugin '%s': %s", plugin_path, err)
        return loaded

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        self._ctx = context
        if context.space_generator is None:
            raise ValueError("TuneContext.space_generator must be defined.")
        if context.target is None:
            raise ValueError("TuneContext.target must be defined.")
        sg = context.space_generator
        self._postprocs = list(sg.postprocs) if sg.postprocs else []
        user_probs = sg.mutator_probs or {}
        for mut, prob_f in user_probs.items():
            p = float(prob_f.value)
            if p > 1e-12:
                self._mutator_probs[mut] = self._mutator_probs.get(mut, 0.0) + p
        target_kind = str(context.target.kind.name)
        if not self._mutator_probs:
            try:
                default_muts = Mutator.create(target_kind)
            except:
                default_muts = Mutator.create("llvm")
            if isinstance(default_muts, dict):
                for m, p2 in default_muts.items():
                    self._mutator_probs[m] = float(p2)
            elif isinstance(default_muts, list) and len(default_muts) > 0:
                p2 = 1.0 / len(default_muts)
                for m in default_muts:
                    self._mutator_probs[m] = p2
        for mutator, prob in self._load_custom_mutator_plugins(context):
            self._mutator_probs[mutator] = self._mutator_probs.get(mutator, 0.0) + prob
        total_p = sum(self._mutator_probs.values())
        if total_p > 1e-12:
            for k in self._mutator_probs:
                self._mutator_probs[k] /= total_p
        if self.verbose >= 1:
            logger.warning(
                "_initialize_with_tune_context: MCTSSearch: Using target=%s, found #mutators=%d, rand_state=%s",
                target_kind, len(self._mutator_probs), str(context.rand_state)
            )

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"],
        cost_model: Optional["CostModel"],
    ) -> None:
        if self.state is not None:
            raise ValueError("MCTSSearch.pre_tuning called without post_tuning after previous run")

        tuner = MCTSTuner(
            population_size=self.population_size,
            init_measured_ratio=self.init_measured_ratio,
            init_min_unmeasured=self.init_min_unmeasured,
            max_fail_count=self.max_fail_count,
            genetic_num_iters=self.genetic_num_iters,
            genetic_mutate_prob=self.genetic_mutate_prob,
            genetic_max_fail_count=self.genetic_max_fail_count,
            num_empty_iters_before_early_stop=self.num_empty_iters_before_early_stop,
            max_stale_iters=self.max_stale_iters,
            diversity_epsilon=self.diversity_epsilon,
            max_stale_diversity_iters=self.max_stale_diversity_iters,
            trace_commit=self.trace_commit,
            verbose=self.verbose,
            mcts_ucb_constant=self.mcts_ucb_constant,
            mcts_max_depth=self.mcts_max_depth,
            mcts_max_children_per_node=self.mcts_max_children_per_node,
            mcts_expansion_candidates=self.mcts_expansion_candidates,
            mcts_use_measured_feedback=self.mcts_use_measured_feedback,
            mcts_measured_feedback_power=self.mcts_measured_feedback_power,
            mcts_use_root_prior_selection=self.mcts_use_root_prior_selection,
            mcts_measure_selection_q_weight=self.mcts_measure_selection_q_weight,
            mcts_measure_selection_diversity_weight=(
                self.mcts_measure_selection_diversity_weight
            ),
            mcts_num_threads=self.mcts_num_threads,
            mcts_num_rollouts_per_expansion=self.mcts_num_rollouts_per_expansion,
            postprocs=self._postprocs,
            mutator_probs=self._mutator_probs,
            context=self._ctx,
            cost_model=cost_model,
            database=database,
            workload_key=None,
            use_llm=self.use_llm,
            llm_budget=self.llm_budget,
            llm_policy=LLMGuidancePolicy(
                model_name=self.llm_model_name,
                verbose=True,
                extra_prompt_context=self.llm_extra_prompt_context,
            ),
            llm_history_depth=self.llm_history_depth,
            llm_extra_prompt_context=self.llm_extra_prompt_context,
            custom_module_spec_path=self.custom_module_spec_path,
            target_kind=str(self._ctx.target.kind.name) if self._ctx and self._ctx.target else "",
            transfer_memory_path=self.transfer_memory_path,
            transfer_top_k=self.transfer_top_k,
            transfer_min_similarity=self.transfer_min_similarity,
            transfer_warmstart_limit=self.transfer_warmstart_limit,
            transfer_enable_max_trials=self.transfer_enable_max_trials,
        )
        self.state = MCTSTuningState(
            max_trials=max_trials,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=design_spaces,
            database=database,
            cost_model=cost_model,
            context=self._ctx,
            tuner=tuner,
        )
        if self.verbose >= 1:
            logger.warning(
                "MCTSSearch.pre_tuning => max_trials=%d, num_per_iter=%d, #design_spaces=%d",
                max_trials, num_trials_per_iter, len(design_spaces)
            )

    def post_tuning(self) -> None:
        if self.state:
            self.state.reset()
            self.state = None
        if self.verbose >= 1:
            logger.warning("MCTSSearch: Tuning finished in post_tuning().")

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        if not self.state:
            logger.warning("MCTSSearch.generate_measure_candidates called before pre_tuning.")
            return None
        return self.state.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        if self.state:
            self.state.notify_runner_results(measure_candidates, results)

    def clone(self) -> "MCTSSearchPyFull":
        return MCTSSearchPyFull(
            population_size=self.population_size,
            init_measured_ratio=self.init_measured_ratio,
            init_min_unmeasured=self.init_min_unmeasured,
            max_fail_count=self.max_fail_count,
            genetic_num_iters=self.genetic_num_iters,
            genetic_mutate_prob=self.genetic_mutate_prob,
            genetic_max_fail_count=self.genetic_max_fail_count,
            num_empty_iters_before_early_stop=self.num_empty_iters_before_early_stop,
            max_stale_iters=self.max_stale_iters,
            diversity_epsilon=self.diversity_epsilon,
            max_stale_diversity_iters=self.max_stale_diversity_iters,
            trace_commit=self.trace_commit,
            verbose=self.verbose,
            mcts_ucb_constant=self.mcts_ucb_constant,
            mcts_max_depth=self.mcts_max_depth,
            mcts_max_children_per_node=self.mcts_max_children_per_node,
            mcts_expansion_candidates=self.mcts_expansion_candidates,
            mcts_use_measured_feedback=self.mcts_use_measured_feedback,
            mcts_measured_feedback_power=self.mcts_measured_feedback_power,
            mcts_use_root_prior_selection=self.mcts_use_root_prior_selection,
            mcts_measure_selection_q_weight=self.mcts_measure_selection_q_weight,
            mcts_measure_selection_diversity_weight=(
                self.mcts_measure_selection_diversity_weight
            ),
            mcts_num_threads=self.mcts_num_threads,
            mcts_num_rollouts_per_expansion=self.mcts_num_rollouts_per_expansion,
            use_llm=self.use_llm,
            llm_budget=self.llm_budget,
            llm_model_name=self.llm_model_name,
            llm_history_depth=self.llm_history_depth,
            llm_extra_prompt_context=self.llm_extra_prompt_context,
            custom_module_spec_path=self.custom_module_spec_path,
            custom_mutator_plugin_paths=self.custom_mutator_plugin_paths,
            transfer_memory_path=self.transfer_memory_path,
            transfer_top_k=self.transfer_top_k,
            transfer_min_similarity=self.transfer_min_similarity,
            transfer_warmstart_limit=self.transfer_warmstart_limit,
            transfer_enable_max_trials=self.transfer_enable_max_trials,
        )
