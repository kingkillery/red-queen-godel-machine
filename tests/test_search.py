"""The RQGM search loop (paper Algorithm 1)."""

from rqgm.mock_providers import MockEvaluatorSlotProvider, MockWorkspaceProvider
from rqgm.providers import EvaluatorCandidate
from rqgm.search import RQGMConfig, RQGMSearch, exponential_checkpoints


def _search(budget: int = 128, seed: int = 0) -> RQGMSearch:
    return RQGMSearch(
        MockWorkspaceProvider(seed),
        {0: MockEvaluatorSlotProvider(0)},
        RQGMConfig(budget=budget, seed=seed),
    )


def test_exponential_checkpoints():
    assert exponential_checkpoints(128, 2, 8) == [8, 16, 32, 64, 128]
    assert exponential_checkpoints(100, 2, 8) == [8, 16, 32, 64, 100]  # budget appended
    assert exponential_checkpoints(5, 2, 8) == [5]  # budget below the minimum


def test_search_completes_with_an_erasing_replacement():
    search = _search(128, 0)
    result = search.run()
    assert result.num_evaluations == 128
    assert result.archive_size > 1
    erasing = [rep for rep in result.replacements if rep.erased > 0]
    assert erasing, "expected at least one replacement that erased dependent records"


def test_post_replacement_has_no_stale_dependent_records():
    search = _search(128, 0)
    result = search.run()
    final_evaluator = result.replacements[-1].to_id
    stale = [
        rec
        for rec in search.archive.records
        if 0 in rec.dep and rec.criterion_tags.get(0) != final_evaluator
    ]
    assert stale == []
    assert result.records_retained == len(search.archive.records)


def test_records_retained_accounts_for_erasure():
    search = _search(128, 0)
    result = search.run()
    total_erased = sum(rep.erased for rep in result.replacements)
    assert result.records_retained == 128 - total_erased


def test_best_node_is_a_decent_lineage():
    search = _search(128, 0)
    result = search.run()
    assert result.best_node_id in search.archive.nodes
    assert result.best_belief > 0.5  # search found a node clearly better than the prior


def test_deterministic_across_constructions():
    ids = {_search(128, 0).run().best_node_id for _ in range(3)}
    assert len(ids) == 1


def test_constant_anchor_keeps_incumbent():
    # Identical anchors for incumbent and challenger => ties resolve to incumbent.
    class TieSlot:
        slot = 0

        def incumbent(self):
            return EvaluatorCandidate("inc", {"strictness": 0.3})

        def challengers(self, archive):
            return [EvaluatorCandidate("chal", {"strictness": 0.3})]

        def anchor_outcomes(self, evaluator):
            return (8, 2)

    search = RQGMSearch(
        MockWorkspaceProvider(0), {0: TieSlot()}, RQGMConfig(budget=64, seed=0)
    )
    result = search.run()
    assert result.replacements == []
