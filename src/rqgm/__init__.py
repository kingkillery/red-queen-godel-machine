"""Red Queen Godel Machine (RQGM).

An independent, dependency-light implementation of the co-evolutionary
archive-search algorithm from "The Red Queen Godel Machine: Co-Evolving Agents
and Their Evaluators" (arXiv:2606.26294). RQGM searches over agent variants while
its evaluators co-evolve under controlled utility evolution: epoch-local frozen
evaluators, anchor-based best-belief replacement, and selective erasure of
evaluator-dependent utility records.

Public API surface:

* :class:`RQGMSearch`, :class:`RQGMConfig`, :class:`RQGMResult`,
  :func:`run_rqgm` -- run a search.
* :class:`Archive`, :class:`ArchiveNode`, :class:`UtilityRecord` -- the ledger.
* :class:`RoleSpec`, :class:`EvaluatorCandidate`, :class:`WorkspaceProvider`,
  :class:`EvaluatorSlotProvider` -- the provider extension points.
* :func:`best_belief`, :func:`posterior_mean` -- Beta scoring primitives.
* Mock and OpenAI-compatible LLM providers for demos and real runs.
"""

from __future__ import annotations

from .archive import Archive, ArchiveNode, UtilityRecord
from .beta import best_belief, posterior_mean
from .llm_providers import (
    AnchorItem,
    ChatModel,
    LabeledAnchorEvaluatorSlot,
    OpenAIChatModel,
    PromptEvolutionWorkspace,
    Sample,
    ScriptedChatModel,
)
from .mock_providers import MockEvaluatorSlotProvider, MockWorkspaceProvider
from .providers import (
    EvaluatorCandidate,
    EvaluatorSlotProvider,
    RoleSpec,
    WorkspaceProvider,
)
from .runner import build_providers, persist_result, result_to_dict, run_rqgm
from .search import (
    Replacement,
    RQGMConfig,
    RQGMResult,
    RQGMSearch,
    exponential_checkpoints,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Archive",
    "ArchiveNode",
    "UtilityRecord",
    "best_belief",
    "posterior_mean",
    "RoleSpec",
    "EvaluatorCandidate",
    "WorkspaceProvider",
    "EvaluatorSlotProvider",
    "RQGMSearch",
    "RQGMConfig",
    "RQGMResult",
    "Replacement",
    "exponential_checkpoints",
    "run_rqgm",
    "build_providers",
    "persist_result",
    "result_to_dict",
    "MockWorkspaceProvider",
    "MockEvaluatorSlotProvider",
    "ChatModel",
    "OpenAIChatModel",
    "ScriptedChatModel",
    "Sample",
    "AnchorItem",
    "PromptEvolutionWorkspace",
    "LabeledAnchorEvaluatorSlot",
]
