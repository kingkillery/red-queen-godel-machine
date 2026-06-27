"""Public extension points for RQGM.

A user wires RQGM to their own domain by implementing two structural
:class:`typing.Protocol` interfaces:

* :class:`WorkspaceProvider` — defines the search nodes, how to expand a node
  into a child variant, and how to score a (node, role, task) under a frozen
  evaluator.
* :class:`EvaluatorSlotProvider` — supplies the incumbent and challenger
  evaluators for a slot, and scores each against fixed ground-truth *anchors*.

Implementations need only match the method signatures (structural typing); they
do not need to subclass anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .archive import Archive, ArchiveNode

__all__ = [
    "RoleKind",
    "RoleSpec",
    "EvaluatorCandidate",
    "WorkspaceProvider",
    "EvaluatorSlotProvider",
]

RoleKind = Literal["evaluator_independent", "evaluator_dependent"]


@dataclass
class RoleSpec:
    """A capability a node is measured on.

    ``evaluator_independent`` roles are scored against fixed ground truth.
    ``evaluator_dependent`` roles are scored by the (evolving) evaluator filling
    ``slot``; such records carry an erasure dependency on that slot.
    """

    name: str
    kind: RoleKind
    tasks: list[str]
    slot: int | None = None


@dataclass
class EvaluatorCandidate:
    """An evaluator that can fill a slot. ``state`` is provider-defined."""

    evaluator_id: str
    state: dict = field(default_factory=dict)


@runtime_checkable
class WorkspaceProvider(Protocol):
    """Defines the node space, expansion, and evaluation of an RQGM run."""

    def roles(self) -> list[RoleSpec]:
        """Return the roles every node is evaluated on (stable across the run)."""
        ...

    def seed(self) -> dict:
        """Return the workspace dict for the seed node."""
        ...

    def expand(self, parent: ArchiveNode) -> dict | None:
        """Propose a child workspace from ``parent``; ``None`` aborts expansion."""
        ...

    def evaluate(
        self,
        node: ArchiveNode,
        role: RoleSpec,
        task: str,
        evaluator: EvaluatorCandidate | None,
    ) -> int:
        """Return a binary outcome (1 success / 0 failure).

        ``evaluator`` is the slot's frozen evaluator for evaluator-dependent
        roles, and ``None`` for evaluator-independent roles.
        """
        ...


@runtime_checkable
class EvaluatorSlotProvider(Protocol):
    """Supplies and anchor-scores evaluators for a single slot."""

    slot: int

    def incumbent(self) -> EvaluatorCandidate:
        """Return the evaluator that fills the slot at the start of the run."""
        ...

    def challengers(self, archive: Archive) -> list[EvaluatorCandidate]:
        """Return candidate evaluators discovered in the archive so far."""
        ...

    def anchor_outcomes(self, evaluator: EvaluatorCandidate) -> tuple[int, int]:
        """Return ``(successes, failures)`` of ``evaluator`` on fixed anchors."""
        ...
