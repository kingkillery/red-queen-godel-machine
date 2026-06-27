"""Utility-record ledger and selective erasure (paper Appendix F)."""

from rqgm.archive import Archive, ArchiveNode, UtilityRecord


def _chain() -> Archive:
    archive = Archive()
    archive.add_node(ArchiveNode("n0", None))
    archive.add_node(ArchiveNode("n1", "n0"))
    archive.add_node(ArchiveNode("n2", "n1"))
    return archive


def test_validity_honors_current_epoch():
    archive = _chain()
    archive.add_record(UtilityRecord("n0", "reviewer", "r0", 1, dep=(0,), criterion_tags={0: "e0"}))
    archive.add_record(UtilityRecord("n0", "reviewer", "r0", 1, dep=(0,), criterion_tags={0: "e1"}))
    # Only the record whose tag matches the current evaluator counts.
    assert archive.node_counts("n0", {0: "e0"}) == (1, 0)
    assert archive.node_counts("n0", {0: "e1"}) == (1, 0)


def test_erase_slot_drops_only_stale_and_keeps_anchor():
    archive = _chain()
    archive.add_record(UtilityRecord("n0", "coder", "t0", 1))  # anchor: dep == ()
    archive.add_record(UtilityRecord("n0", "reviewer", "r0", 1, dep=(0,), criterion_tags={0: "e0"}))
    archive.add_record(UtilityRecord("n0", "reviewer", "r0", 0, dep=(0,), criterion_tags={0: "e1"}))

    erased = archive.erase_slot(0, "e1")

    assert erased == 1  # only the e0-tagged dependent record
    assert len(archive.records) == 2
    assert any(r.dep == () for r in archive.records)  # anchor survived
    assert all(r.criterion_tags.get(0) in (None, "e1") for r in archive.records)


def test_clade_counts_sums_subtree():
    archive = _chain()
    archive.add_record(UtilityRecord("n1", "coder", "t0", 1))
    archive.add_record(UtilityRecord("n2", "coder", "t0", 0))
    archive.add_record(UtilityRecord("n0", "coder", "t0", 1))
    assert archive.clade_counts("n0", {}) == (2, 1)  # n0 + n1 + n2
    assert archive.clade_counts("n1", {}) == (1, 1)  # n1 + n2
    assert archive.clade_counts("n2", {}) == (0, 1)


def test_disjoint_slot_erase_commutes():
    # Rem. 1: erasures of distinct slots commute.
    def build() -> Archive:
        archive = _chain()
        archive.add_record(
            UtilityRecord("n0", "rev", "x", 1, dep=(0,), criterion_tags={0: "a0"})
        )
        archive.add_record(
            UtilityRecord("n0", "rev", "y", 1, dep=(1,), criterion_tags={1: "b0"})
        )
        archive.add_record(UtilityRecord("n0", "coder", "z", 1))  # anchor
        return archive

    forward = build()
    forward.erase_slot(0, "a1")
    forward.erase_slot(1, "b1")

    reverse = build()
    reverse.erase_slot(1, "b1")
    reverse.erase_slot(0, "a1")

    def key(records):
        return sorted((r.role, r.task, r.outcome, r.dep) for r in records)

    assert key(forward.records) == key(reverse.records)
    assert len(forward.records) == 1  # only the anchor survives both erasures


def test_balanced_utility_defaults_unmeasured_to_half():
    archive = _chain()
    # coder measured once (success -> posterior mean 2/3); reviewer unmeasured (0.5).
    archive.add_record(UtilityRecord("n0", "coder", "t0", 1))
    score = archive.balanced_utility("n0", ["coder", "reviewer"], {})
    assert abs(score - 7.0 / 12.0) < 1e-9
