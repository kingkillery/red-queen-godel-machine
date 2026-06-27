"""Utility-record ledger with selective erasure (paper Appendix F).

The archive is the load-bearing data structure of RQGM. Every evaluation is
appended as an immutable :class:`UtilityRecord` tagged with the evaluator
*slots* it depended on and which concrete evaluator filled each slot at the
time. All derived statistics (per-node S/F, clade metaproductivity, balanced
utility) are recomputed from the retained records, so:

* **Selective erasure** (Def. F.2) is a pure filter: when an evaluator in a slot
  is replaced, only records whose tag for that slot no longer matches the new
  evaluator are dropped. Anchor records (``dep == ()``) never depend on a slot
  and always survive (Prop. 2 / Rem. 2).
* **Validity** (Def. F.1): a record counts toward the current objective only if,
  for every slot it depends on, its recorded evaluator id equals the slot's
  current evaluator id.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .beta import posterior_mean

__all__ = ["UtilityRecord", "ArchiveNode", "Archive"]


@dataclass
class UtilityRecord:
    """One evaluation outcome and the evaluator context it depended on.

    ``dep`` lists the evaluator slots the outcome depended on (empty for
    evaluator-independent / anchor records). ``criterion_tags`` maps each
    depended-on slot to the concrete evaluator id that produced the outcome.
    ``epoch_vector`` records the per-slot epoch counters at recording time.
    """

    node_id: str
    role: str
    task: str
    outcome: int
    dep: tuple[int, ...] = ()
    criterion_tags: dict[int, str] = field(default_factory=dict)
    epoch_vector: tuple[int, ...] = ()


@dataclass
class ArchiveNode:
    """A node (agent variant) in the archive. Lineage only; never selection."""

    node_id: str
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    workspace: dict = field(default_factory=dict)
    train_feedback: dict = field(default_factory=dict)


class Archive:
    """Tree of nodes plus an append-only ledger of utility records."""

    def __init__(self) -> None:
        self.nodes: dict[str, ArchiveNode] = {}
        self.records: list[UtilityRecord] = []

    # -- structure ---------------------------------------------------------
    def add_node(self, node: ArchiveNode) -> ArchiveNode:
        self.nodes[node.node_id] = node
        if node.parent_id is not None:
            parent = self.nodes.get(node.parent_id)
            if parent is not None and node.node_id not in parent.children:
                parent.children.append(node.node_id)
        return node

    def add_record(self, record: UtilityRecord) -> None:
        self.records.append(record)

    def clade(self, node_id: str) -> set[str]:
        """Return ``node_id`` and all of its transitive descendants."""
        seen: set[str] = set()
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            if current in seen or current not in self.nodes:
                continue
            seen.add(current)
            queue.extend(self.nodes[current].children)
        return seen

    # -- validity ----------------------------------------------------------
    def _valid(self, rec: UtilityRecord, current_epoch: dict[int, str]) -> bool:
        """Def. F.1: valid iff every depended-on slot matches the current id."""
        for slot in rec.dep:
            if rec.criterion_tags.get(slot) != current_epoch.get(slot):
                return False
        return True

    # -- counts ------------------------------------------------------------
    def node_counts(self, node_id: str, current_epoch: dict[int, str]) -> tuple[int, int]:
        successes = failures = 0
        for rec in self.records:
            if rec.node_id != node_id or not self._valid(rec, current_epoch):
                continue
            if rec.outcome:
                successes += 1
            else:
                failures += 1
        return successes, failures

    def clade_counts(self, node_id: str, current_epoch: dict[int, str]) -> tuple[int, int]:
        members = self.clade(node_id)
        successes = failures = 0
        for rec in self.records:
            if rec.node_id not in members or not self._valid(rec, current_epoch):
                continue
            if rec.outcome:
                successes += 1
            else:
                failures += 1
        return successes, failures

    def role_count(self, node_id: str, role: str, current_epoch: dict[int, str]) -> int:
        total = 0
        for rec in self.records:
            if rec.node_id == node_id and rec.role == role and self._valid(rec, current_epoch):
                total += 1
        return total

    def role_task_counts(
        self, node_id: str, role: str, task: str, current_epoch: dict[int, str]
    ) -> tuple[int, int]:
        successes = failures = 0
        for rec in self.records:
            if (
                rec.node_id != node_id
                or rec.role != role
                or rec.task != task
                or not self._valid(rec, current_epoch)
            ):
                continue
            if rec.outcome:
                successes += 1
            else:
                failures += 1
        return successes, failures

    # -- erasure -----------------------------------------------------------
    def erase_slot(self, slot: int, new_evaluator_id: str) -> int:
        """Def. F.2: drop records depending on ``slot`` with a stale tag.

        Returns the number of erased records. Records that do not depend on
        ``slot`` (including anchor records with ``dep == ()``) are retained.
        """
        kept: list[UtilityRecord] = []
        erased = 0
        for rec in self.records:
            if slot in rec.dep and rec.criterion_tags.get(slot) != new_evaluator_id:
                erased += 1
            else:
                kept.append(rec)
        self.records = kept
        return erased

    # -- balanced utility --------------------------------------------------
    def balanced_utility(
        self, node_id: str, roles: list[str], current_epoch: dict[int, str]
    ) -> float:
        """U_j: mean over roles of the mean per-task posterior rate.

        Empty role/task cells default to ``0.5`` (the uninformed prior mean).
        """
        if not roles:
            return 0.5
        tasks_by_role: dict[str, set[str]] = {role: set() for role in roles}
        for rec in self.records:
            if rec.node_id != node_id or rec.role not in tasks_by_role:
                continue
            if self._valid(rec, current_epoch):
                tasks_by_role[rec.role].add(rec.task)
        role_scores: list[float] = []
        for role in roles:
            tasks = tasks_by_role[role]
            if not tasks:
                role_scores.append(0.5)
                continue
            task_means = [
                posterior_mean(*self.role_task_counts(node_id, role, task, current_epoch))
                for task in tasks
            ]
            role_scores.append(sum(task_means) / len(task_means))
        return sum(role_scores) / len(role_scores)
