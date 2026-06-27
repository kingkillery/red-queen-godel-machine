"""Paper-claim conformance tests for RQGM mechanisms.

These tests exercise claims that are central to arXiv:2606.26294 and are not just
implementation details: checkpoint-only evaluator replacement, epoch-local frozen
evaluators, anchor-BB directionality/tie behavior, selective erasure, logarithmic
checkpoints, UCB-Air archive growth, and same-seed Thompson reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log2

from rqgm.archive import Archive, ArchiveNode, UtilityRecord
from rqgm.providers import EvaluatorCandidate, RoleSpec
from rqgm.search import RQGMConfig, RQGMSearch, exponential_checkpoints


@dataclass(frozen=True, slots=True)
class _AnchorTableSlot:
    """Slot provider with explicit incumbent/challenger anchor outcomes."""

    incumbent_outcomes: tuple[int, int]
    challenger_outcomes: tuple[int, int]
    challenger_id: str = "chal"
    slot: int = 0

    def incumbent(self) -> EvaluatorCandidate:
        return EvaluatorCandidate("inc")

    def challengers(self, archive: Archive) -> list[EvaluatorCandidate]:
        return [EvaluatorCandidate(self.challenger_id)]

    def anchor_outcomes(self, evaluator: EvaluatorCandidate) -> tuple[int, int]:
        if evaluator.evaluator_id == self.challenger_id:
            return self.challenger_outcomes
        return self.incumbent_outcomes


class _NoExpandWorkspace:
    """Two-role workspace that never expands and returns successful outcomes."""

    def roles(self) -> list[RoleSpec]:
        return [
            RoleSpec("coder", "evaluator_independent", ["anchor"]),
            RoleSpec("reviewer", "evaluator_dependent", ["judge"], slot=0),
        ]

    def seed(self) -> dict[str, int]:
        return {"generation": 0}

    def expand(self, parent: ArchiveNode) -> None:
        return None

    def evaluate(
        self,
        node: ArchiveNode,
        role: RoleSpec,
        task: str,
        evaluator: EvaluatorCandidate | None,
    ) -> int:
        return 1


class _AlwaysExpandWorkspace:
    """Single-role workspace that yields a fresh child for every gate opening."""

    def roles(self) -> list[RoleSpec]:
        return [RoleSpec("coder", "evaluator_independent", ["task"])]

    def seed(self) -> dict[str, int]:
        return {"generation": 0}

    def expand(self, parent: ArchiveNode) -> dict[str, int]:
        return {"generation": int(parent.workspace.get("generation", 0)) + 1}

    def evaluate(
        self,
        node: ArchiveNode,
        role: RoleSpec,
        task: str,
        evaluator: EvaluatorCandidate | None,
    ) -> int:
        return 1


def _run_no_expand(
    slot: _AnchorTableSlot,
    budget: int = 16,
    checkpoint_min: int = 8,
    seed: int = 0,
) -> tuple[RQGMSearch, object]:
    search = RQGMSearch(
        _NoExpandWorkspace(),
        {0: slot},
        RQGMConfig(budget=budget, checkpoint_min=checkpoint_min, seed=seed),
    )
    return search, search.run()


def test_replacements_happen_only_at_checkpoints() -> None:
    """A better challenger available from eval 0 still replaces only at checkpoint 8."""
    slot = _AnchorTableSlot((3, 7), (9, 1))
    search, result = _run_no_expand(slot, budget=16, checkpoint_min=8)
    checkpoints = set(exponential_checkpoints(16, 2, 8))

    assert [rep.at_eval for rep in result.replacements] == [8]
    assert all(rep.at_eval in checkpoints for rep in result.replacements)
    assert result.replacements[0].from_id == "inc"
    assert result.replacements[0].to_id == "chal"
    assert search.archive.records


def test_evaluator_frozen_within_epoch_when_no_replacement() -> None:
    """Without replacement, every dependent record carries the same evaluator tag."""
    search, result = _run_no_expand(_AnchorTableSlot((8, 2), (8, 2)), budget=16)

    dependent_tags = {
        rec.criterion_tags[0]
        for rec in search.archive.records
        if rec.dep == (0,)
    }
    assert dependent_tags == {"inc"}
    assert result.epochs == {0: 1}


def test_evaluator_swaps_at_first_checkpoint() -> None:
    """A strictly better challenger swaps the slot exactly at the first checkpoint."""
    _search, result = _run_no_expand(_AnchorTableSlot((3, 7), (9, 1)), budget=16)

    assert len(result.replacements) == 1
    assert result.replacements[0].at_eval == 8
    assert result.epochs == {0: 2}


def test_strictly_better_anchor_challenger_replaces_incumbent() -> None:
    """Anchor BB direction: better anchor evidence replaces incumbent."""
    _search, result = _run_no_expand(_AnchorTableSlot((3, 7), (9, 1)), budget=16)

    assert [(rep.from_id, rep.to_id) for rep in result.replacements] == [("inc", "chal")]


def test_strictly_worse_anchor_challenger_does_not_replace() -> None:
    """Anchor BB direction: worse anchor evidence cannot replace incumbent."""
    _search, result = _run_no_expand(_AnchorTableSlot((9, 1), (3, 7)), budget=16)

    assert result.replacements == []
    assert result.epochs == {0: 1}


def test_equal_anchor_bb_tie_keeps_incumbent() -> None:
    """Anchor-BB ties resolve to the incumbent."""
    _search, result = _run_no_expand(_AnchorTableSlot((8, 2), (8, 2)), budget=16)

    assert result.replacements == []
    assert result.epochs == {0: 1}


def test_selective_erasure_drops_stale_dependent_records_only() -> None:
    """After replacement, retained dependent records all carry the new evaluator tag."""
    search, result = _run_no_expand(_AnchorTableSlot((3, 7), (9, 1)), budget=16)
    replacement = result.replacements[0]
    dependent = [rec for rec in search.archive.records if rec.dep == (0,)]

    assert replacement.erased == 4
    assert dependent
    assert all(rec.criterion_tags[0] == "chal" for rec in dependent)
    assert result.records_retained == 16 - replacement.erased


def test_anchor_records_with_empty_dep_survive_erasure() -> None:
    """Evaluator-independent records (`dep == ()`) survive slot replacement."""
    search, result = _run_no_expand(_AnchorTableSlot((3, 7), (9, 1)), budget=16)
    replacement = result.replacements[0]
    anchors = [rec for rec in search.archive.records if rec.dep == ()]
    dependents = [rec for rec in search.archive.records if rec.dep == (0,)]

    assert anchors
    assert result.records_retained == len(anchors) + len(dependents)
    assert len(anchors) == 16 - len(dependents) - replacement.erased


def test_checkpoints_always_include_final_budget() -> None:
    """The checkpoint schedule always includes the final budget and is deduped."""
    for budget in (1, 5, 7, 8, 9, 16, 17, 64, 100, 128, 1000):
        checkpoints = exponential_checkpoints(budget, base=2, minimum=8)
        assert checkpoints[-1] == budget
        assert checkpoints == sorted(set(checkpoints))


def test_checkpoints_grow_only_logarithmically() -> None:
    """Checkpoint count is logarithmic in budget under exponential spacing."""
    for budget in (8, 9, 16, 17, 32, 64, 128, 256, 512, 1024):
        checkpoints = exponential_checkpoints(budget, base=2, minimum=8)
        expected_max = floor(log2(max(budget, 8) / 8)) + 2
        assert len(checkpoints) <= expected_max
        for left, right in zip(checkpoints, checkpoints[1:], strict=False):
            assert right > left
            assert right == budget or right == left * 2


def test_ucb_air_bounds_archive_size_for_always_expand() -> None:
    """Even an always-expand provider is bounded by the UCB-Air gate."""
    budget = 64
    alpha = 0.6
    search = RQGMSearch(
        _AlwaysExpandWorkspace(),
        {},
        RQGMConfig(budget=budget, alpha=alpha, seed=0),
    )

    result = search.run()

    assert result.archive_size == 13
    assert result.num_expansions == result.archive_size - 1
    assert result.archive_size <= int(budget**alpha) + 2
    assert result.archive_size < budget


def test_clade_thompson_sampling_is_seed_reproducible() -> None:
    """Same seed + same archive produces identical measured-cell samples."""
    archive = Archive()
    for index in range(9):
        parent = None if index == 0 else f"n{(index - 1) // 2}"
        archive.add_node(ArchiveNode(f"n{index}", parent))
        archive.add_record(UtilityRecord(f"n{index}", "coder", "task", int(index % 2 == 0)))
    roles = [RoleSpec("coder", "evaluator_independent", ["task"])]

    def sample_sequence() -> list[tuple[str, str, str]]:
        search = RQGMSearch(_NoExpandWorkspace(), {}, RQGMConfig(seed=7), archive=archive)
        sequence = []
        for _ in range(8):
            node_id = search._sample_clade({})
            role, task = search._least_measured_cell(node_id, roles, {})
            sequence.append((node_id, role.name, task))
        return sequence

    assert sample_sequence() == sample_sequence()
