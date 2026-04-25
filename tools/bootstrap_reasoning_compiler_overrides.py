#!/usr/bin/env python3
"""Overlay the repo-local REASONING COMPILER Python modules onto an installed TVM package."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

MODULE_REL_PATHS = {
    "tvm.meta_schedule.search_strategy.llm_guidance": "python/tvm/meta_schedule/search_strategy/llm_guidance.py",
    "tvm.meta_schedule.search_strategy.mcts_search": "python/tvm/meta_schedule/search_strategy/mcts_search.py",
}


def _load_module(module_name: str, source_path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_name} from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def apply_repo_overrides(repo_root: str | pathlib.Path | None = None) -> ModuleType:
    """Load the repo's search-strategy Python files on top of an installed TVM package."""

    import tvm.meta_schedule.search_strategy as search_strategy_pkg

    if repo_root is None:
        root = pathlib.Path(__file__).resolve().parents[1]
    else:
        root = pathlib.Path(repo_root).resolve()
    loaded_modules: dict[str, ModuleType] = {}
    for module_name, rel_path in MODULE_REL_PATHS.items():
        loaded_modules[module_name] = _load_module(module_name, root / rel_path)

    llm_module = loaded_modules["tvm.meta_schedule.search_strategy.llm_guidance"]
    mcts_module = loaded_modules["tvm.meta_schedule.search_strategy.mcts_search"]

    search_strategy_pkg.LLMGuidancePolicy = llm_module.LLMGuidancePolicy
    search_strategy_pkg.MCTSSearchPyFull = mcts_module.MCTSSearchPyFull
    search_strategy_pkg.llm_guidance = llm_module
    search_strategy_pkg.mcts_search = mcts_module
    return search_strategy_pkg


if __name__ == "__main__":
    pkg = apply_repo_overrides()
    print("Loaded repo overrides for:", pkg.MCTSSearchPyFull.__name__)
