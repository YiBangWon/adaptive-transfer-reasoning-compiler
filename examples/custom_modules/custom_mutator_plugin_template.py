"""Template for adding operator-specific mutators to REASONING COMPILER.

Copy this file next to your experiment and pass its path through
MCTSSearchPyFull(custom_mutator_plugin_paths="..."). The concrete mutator below
is intentionally conservative: it shows the TVM PyMutator interface, but returns
None until you replace the body with a real trace rewrite for your operator.
"""

from __future__ import annotations

from typing import Optional

from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Trace


@derived_object
class MutateFusedRMSNormSWIGLURows(ms.mutator.PyMutator):
    """Example custom mutator hook for a fused RMSNorm + SwiGLU CUDA kernel."""

    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        self.target = context.target

    def apply(self, trace: Trace, _) -> Optional[Trace]:
        # Replace this with an operator-specific rewrite, for example:
        # - adjust split decisions for the row/hidden dimension,
        # - move the reduction block into a CTA-local schedule,
        # - or enforce a vector-load-friendly binding pattern.
        #
        # Return None when the mutation is not valid for the current trace.
        return None

    def clone(self):
        return MutateFusedRMSNormSWIGLURows()


def register_mutators(context: ms.TuneContext):
    """Return custom mutators and their unnormalized sampling weights."""

    return {
        MutateFusedRMSNormSWIGLURows(): 0.20,
    }
