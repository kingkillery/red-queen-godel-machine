"""Deterministic mock providers for tests, demos, and CI.

These providers exercise the full RQGM machinery (node expansion, evaluator
challengers, anchor-based replacement, selective erasure) without any network or
external dependency. Randomness is derived from SHA-256 of stable string keys
(never the built-in ``hash``, whose per-process salt would break cross-process
CLI determinism), and the per-cell evaluation counter makes repeated evaluations
of the same cell vary while staying reproducible across runs.

The anchor table is rigged so a stricter, better-anchored reviewer always wins
on best-belief once a node evolves it, guaranteeing at least one evaluator
replacement (and therefore a selective erasure) within a modest budget.
"""

from __future__ import annotations

import hashlib
import random

from .archive import Archive, ArchiveNode
from .providers import EvaluatorCandidate, RoleSpec

__all__ = ["MockWorkspaceProvider", "MockEvaluatorSlotProvider"]


def _stable_rng(*parts: object) -> random.Random:
    key = ":".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    return random.Random(int.from_bytes(digest, "big"))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# Reviewer prompt lineage: later ids are stricter (accept less) but better anchored.
_NEXT_PROMPT = {"rev_e0": "rev_e1", "rev_e1": "rev_e2", "rev_e2": "rev_e2"}
_STRICTNESS = {"rev_e0": 0.3, "rev_e1": 0.5, "rev_e2": 0.7}
_ANCHOR = {"rev_e0": (7, 3), "rev_e1": (8, 2), "rev_e2": (9, 1)}


class MockWorkspaceProvider:
    """A toy coder/reviewer workspace whose quality drifts as nodes expand."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def roles(self) -> list[RoleSpec]:
        return [
            RoleSpec("coder", "evaluator_independent", ["t0", "t1", "t2", "t3"]),
            RoleSpec("reviewer", "evaluator_dependent", ["r0", "r1"], slot=0),
        ]

    def seed(self) -> dict:
        return {"quality": 0.5, "reviewer_prompt_id": "rev_e0"}

    def expand(self, parent: ArchiveNode) -> dict | None:
        rng = _stable_rng(self._seed, parent.node_id, "expand")
        if rng.random() < 0.1:
            return None  # an unviable proposal
        quality = _clamp(parent.workspace.get("quality", 0.5) + rng.uniform(-0.1, 0.2))
        prompt_id = parent.workspace.get("reviewer_prompt_id", "rev_e0")
        if rng.random() < 0.5:
            prompt_id = _NEXT_PROMPT[prompt_id]
        return {"quality": quality, "reviewer_prompt_id": prompt_id}

    def evaluate(
        self,
        node: ArchiveNode,
        role: RoleSpec,
        task: str,
        evaluator: EvaluatorCandidate | None,
    ) -> int:
        quality = node.workspace.get("quality", 0.5)
        probability = quality
        evaluator_id = "none"
        if evaluator is not None:
            evaluator_id = evaluator.evaluator_id
            probability -= evaluator.state.get("strictness", 0.0) * 0.3
        rng = _stable_rng(self._seed, node.node_id, role.name, task, evaluator_id)
        return int(rng.random() < _clamp(probability))


class MockEvaluatorSlotProvider:
    """Slot 0 reviewer provider with a fixed, strictly increasing anchor table."""

    def __init__(self, slot: int = 0) -> None:
        self.slot = slot

    def incumbent(self) -> EvaluatorCandidate:
        return EvaluatorCandidate("rev_e0", {"strictness": _STRICTNESS["rev_e0"]})

    def challengers(self, archive: Archive) -> list[EvaluatorCandidate]:
        found: set[str] = set()
        for node in archive.nodes.values():
            prompt_id = node.workspace.get("reviewer_prompt_id")
            if prompt_id and prompt_id != "rev_e0":
                found.add(prompt_id)
        return [
            EvaluatorCandidate(prompt_id, {"strictness": _STRICTNESS.get(prompt_id, 0.5)})
            for prompt_id in sorted(found)
        ]

    def anchor_outcomes(self, evaluator: EvaluatorCandidate) -> tuple[int, int]:
        return _ANCHOR.get(evaluator.evaluator_id, (0, 0))
