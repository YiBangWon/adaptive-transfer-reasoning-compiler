#!/usr/bin/env python3
"""Smoke-test custom module specs and custom mutator plugin loading."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_repo_overrides() -> None:
    bootstrap_path = REPO_ROOT / "tools" / "bootstrap_reasoning_compiler_overrides.py"
    spec = importlib.util.spec_from_file_location("bootstrap_reasoning_compiler_overrides", bootstrap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bootstrap script from {bootstrap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_repo_overrides(REPO_ROOT)


def main() -> None:
    load_repo_overrides()

    from tvm.meta_schedule.search_strategy import mcts_search

    with tempfile.TemporaryDirectory(prefix="rc_custom_module_") as temp_dir:
        temp_root = pathlib.Path(temp_dir)
        spec_path = temp_root / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "modules": [
                        {
                            "name": "smoke_custom_cuda_module",
                            "target_kinds": ["cuda"],
                            "match_keywords": ["rmsnorm"],
                            "prompt_context": "Prefer row-wise cooperative reductions.",
                            "mutator_prior_by_category": {
                                "thread_binding": 0.5,
                                "tile_size": 0.25,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        signature = {
            "target_kind": "cuda",
            "keywords": ["rmsnorm"],
            "counts": {},
            "module_excerpt": 'with T.block("rmsnorm")',
        }
        matched = mcts_search._match_custom_module_specs(signature, str(spec_path))
        assert len(matched) == 1, matched

        prompt_context = mcts_search._build_custom_module_prompt_context(matched)
        assert "row-wise cooperative reductions" in prompt_context

        priors = mcts_search._build_custom_module_prior_by_category(matched)
        assert priors["thread_binding"] == 0.5
        assert priors["tile_size"] == 0.25

        plugin_path = temp_root / "plugin.py"
        plugin_path.write_text(
            """
from typing import Optional

from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Trace


@derived_object
class TraceCloneMutator(ms.mutator.PyMutator):
    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        pass

    def apply(self, trace: Trace, _) -> Optional[Trace]:
        return Trace(trace.insts, {})

    def clone(self):
        return TraceCloneMutator()


def register_mutators(context=None):
    return {TraceCloneMutator(): 0.25}
""",
            encoding="utf-8",
        )
        module = mcts_search._import_custom_mutator_plugin(str(plugin_path))
        normalized = mcts_search._normalize_custom_mutator_result(module.register_mutators())
        assert len(normalized) == 1, normalized
        assert abs(normalized[0][1] - 0.25) < 1e-12

        from tvm import meta_schedule as ms

        strategy = mcts_search.MCTSSearchPyFull(custom_mutator_plugin_paths=str(plugin_path))
        loaded = strategy._load_custom_mutator_plugins(context=ms.TuneContext(target="llvm"))
        assert len(loaded) == 1, loaded
        assert abs(loaded[0][1] - 0.25) < 1e-12

    print("custom module extension smoke test passed")


if __name__ == "__main__":
    main()
