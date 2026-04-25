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

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMGuidancePolicy:
    def __init__(
        self,
        model_name: str = "",
        verbose: bool = True,
        extra_prompt_context: str = "",
    ):
        self.model_name = model_name
        self.verbose = verbose
        self.extra_prompt_context = extra_prompt_context.strip()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def pick_mutators(
        self,
        mod,
        available_mutators: List[str],
        historical_perf: Optional[str] = None,
        available_mutator_probs: Optional[Dict[str, float]] = None,
        extra_context: Optional[str] = None,
    ) -> Optional[List[str]]:
        tir_text = self._get_tir_as_text(mod)
        system_prompt, user_prompt = self._build_prompt(
            tir_text=tir_text,
            available_mutators=available_mutators,
            historical_perf=historical_perf,
            mutator_probs=available_mutator_probs,
            extra_context=extra_context,
        )
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            if self.verbose:
                logger.warning("LLM raw response:\n%s", content)
            chosen_list = self._extract_mutators_list(content, available_mutators)
            if not chosen_list:
                if self.verbose:
                    logger.warning("LLM did not return a valid list of mutators.")
                return None
            return chosen_list
        except Exception as err:  # pylint: disable=broad-except
            logger.warning("OpenAI ChatCompletion failed: %s", str(err))
            return None

    def _get_tir_as_text(self, mod) -> str:
        try:
            return mod.script(show_meta=True)
        except Exception as err:  # pylint: disable=broad-except
            logger.warning("Failed to script the IRModule: %s", str(err))
            return "<IRModule scripting failed>"

    def _build_prompt(
        self,
        tir_text: str,
        available_mutators: List[str],
        historical_perf: Optional[str],
        mutator_probs: Optional[Dict[str, float]] = None,
        extra_context: Optional[str] = None,
    ) -> Tuple[str, str]:
        system_msg = (
            "You are an AI scheduling assistant integrated with TVM MetaSchedule. "
            "We are performing Monte Carlo Tree Search (MCTS) to find a strong "
            "schedule transformation sequence for the current TensorIR module.\n\n"
            "You will receive the current IRModule, a history of ancestor schedules "
            "with traces and scores, and the exact mutators that can be applied next. "
            "Use the history to reason about which transformations are synergistic, "
            "which transformations appear to stall, and what structural opportunities "
            "remain.\n\n"
            "Return only a JSON object with this schema:\n"
            '{"rationale": "brief explanation", "mutators": ["ExactMutatorName", "..."]}\n'
            "Rules:\n"
            "1. The mutator names must exactly match items from the provided list.\n"
            "2. You may repeat the same mutator multiple times.\n"
            "3. Keep the rationale short; do not expose hidden chain-of-thought.\n"
            "4. Do not output markdown fences or any text outside the JSON object.\n"
        )

        user_msg = (
            "=== Current IRModule (TensorIR) ===\n"
            f"```python\n{tir_text}\n```\n\n"
        )

        if historical_perf:
            user_msg += (
                "=== Historical Schedule Context ===\n"
                f"{historical_perf}\n\n"
            )

        if mutator_probs:
            ranked = sorted(mutator_probs.items(), key=lambda item: item[1], reverse=True)
            user_msg += "=== Available Mutators With Priors ===\n"
            user_msg += "\n".join(f"- {name}: {prob:.4f}" for name, prob in ranked)
            user_msg += "\n\n"
        else:
            user_msg += "=== Available Mutators ===\n"
            user_msg += "\n".join(f"- {name}" for name in available_mutators)
            user_msg += "\n\n"

        merged_context = extra_context or self.extra_prompt_context
        if merged_context:
            user_msg += (
                "=== Extra Hardware / Operator Context ===\n"
                f"{merged_context}\n\n"
            )

        user_msg += (
            "Pick a short sequence of mutators that is likely to improve the current "
            "schedule while respecting the observed history. Prefer sequences that make "
            "sense for the current target and avoid repeating mutators that look clearly "
            "counter-productive in the recent ancestry."
        )
        return system_msg, user_msg

    def _extract_mutators_list(
        self,
        model_text: str,
        valid_mutators: List[str],
    ) -> List[str]:
        parsed = self._extract_mutators_from_json(model_text, valid_mutators)
        if parsed:
            return parsed
        return self._extract_mutators_from_lines(model_text, valid_mutators)

    def _extract_mutators_from_json(
        self,
        model_text: str,
        valid_mutators: List[str],
    ) -> List[str]:
        valid_set = set(valid_mutators)
        candidates = [model_text.strip()]
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", model_text, flags=re.DOTALL)
        candidates.extend(fenced)
        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            mutators = payload.get("mutators")
            if not isinstance(mutators, list):
                continue
            chosen = [item for item in mutators if isinstance(item, str) and item in valid_set]
            if chosen:
                return chosen
        return []

    def _extract_mutators_from_lines(
        self,
        model_text: str,
        valid_mutators: List[str],
    ) -> List[str]:
        chosen_list: List[str] = []
        valid_set = set(valid_mutators)
        for line in model_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("mutators:"):
                remainder = stripped.split(":", 1)[1].strip()
                raw_names = [item.strip().strip('"').strip("'") for item in remainder.split(",")]
                for candidate in raw_names:
                    if candidate in valid_set:
                        chosen_list.append(candidate)
                if chosen_list:
                    return chosen_list
        for candidate in valid_mutators:
            if candidate in model_text:
                chosen_list.append(candidate)
        return chosen_list
