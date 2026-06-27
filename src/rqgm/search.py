"""The RQGM search loop (paper Algorithm 1 / Appendix F).

One epoch per evaluator slot runs with a *frozen* evaluator. Search interleaves
UCB-Air-gated expansion with three-level (node -> role -> task) sampling, scoring
selection by Beta best-belief over clade metaproductivity (Thompson sampling).
At exponentially spaced checkpoints each slot's frozen evaluator may be replaced
by an anchor-best-belief challenger; on replacement the dependent utility records
are selectively erased and the slot's epoch advances.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .archive import Archive, ArchiveNode, UtilityRecord
from .beta import best_belief
from .providers import EvaluatorCandidate, EvaluatorSlotProvider, RoleSpec, WorkspaceProvider

__all__ = [
    "RQGMConfig",
    "Replacement",
    "RQGMResult",
    "exponential_checkpoints",
    "RQGMSearch",
]


@dataclass
class RQGMConfig:
    budget: int = 256
    epsilon: float = 0.05
    alpha: float = 0.6
    checkpoint_base: int = 2
    checkpoint_min: int = 8
    min_anchor_outcomes: int = 5
    min_node_evals: int = 5
    seed: int = 0


@dataclass
class Replacement:
    slot: int
    from_id: str
    to_id: str
    anchor_best_belief: float
    erased: int
    at_eval: int


@dataclass
class RQGMResult:
    best_node_id: str
    best_belief: float
    balanced_utility: float
    archive_size: int
    num_evaluations: int
    num_expansions: int
    epochs: dict[int, int]
    replacements: list[Replacement] = field(default_factory=list)
    records_retained: int = 0


def exponential_checkpoints(budget: int, base: int, minimum: int) -> list[int]:
    """Checkpoint counts ``minimum * base**q`` up to (and including) ``budget``.

    Exponential spacing bounds total checkpoint reprocessing to ``O(budget)``
    (Prop. 6). ``budget`` is always a checkpoint so a final replacement pass runs.
    """
    points: set[int] = set()
    q = 0
    while True:
        value = minimum * (base**q)
        if value > budget:
            break
        points.add(value)
        q += 1
    points.add(budget)
    return sorted(points)


class RQGMSearch:
    """Run RQGM over a :class:`WorkspaceProvider` and a set of slot providers."""

    def __init__(
        self,
        workspace: WorkspaceProvider,
        slots: dict[int, EvaluatorSlotProvider],
        config: RQGMConfig | None = None,
        archive: Archive | None = None,
    ) -> None:
        self.workspace = workspace
        self.slots = slots
        self.config = config or RQGMConfig()
        self.archive = archive or Archive()
        self._rng = random.Random(self.config.seed)
        self._next_index = 0

    # -- sampling helpers --------------------------------------------------
    def _sample_clade(self, current_epoch: dict[int, str]) -> str:
        """Thompson sample a node by its clade metaproductivity posterior."""
        best_id: str | None = None
        best_draw = -1.0
        for node_id in self.archive.nodes:
            successes, failures = self.archive.clade_counts(node_id, current_epoch)
            draw = self._rng.betavariate(1.0 + successes, 1.0 + failures)
            if draw > best_draw:
                best_draw = draw
                best_id = node_id
        assert best_id is not None  # archive always holds the seed node
        return best_id

    def _least_measured_cell(
        self, node_id: str, roles: list[RoleSpec], current_epoch: dict[int, str]
    ) -> tuple[RoleSpec, str]:
        """Pick the least-measured role, then its least-measured task."""
        role = min(roles, key=lambda r: self.archive.role_count(node_id, r.name, current_epoch))
        best_task = role.tasks[0]
        best_count: int | None = None
        for task in role.tasks:
            successes, failures = self.archive.role_task_counts(
                node_id, role.name, task, current_epoch
            )
            count = successes + failures
            if best_count is None or count < best_count:
                best_count = count
                best_task = task
        return role, best_task

    # -- checkpoint --------------------------------------------------------
    def _checkpoint(
        self,
        frozen: dict[int, EvaluatorCandidate],
        epoch: dict[int, int],
        current_epoch: dict[int, str],
        replacements: list[Replacement],
        at_eval: int,
    ) -> None:
        cfg = self.config
        for slot, provider in self.slots.items():
            incumbent = frozen[slot]
            candidates = [incumbent]
            for challenger in provider.challengers(self.archive):
                if all(challenger.evaluator_id != c.evaluator_id for c in candidates):
                    candidates.append(challenger)
            scored: list[tuple[EvaluatorCandidate, float]] = []
            for candidate in candidates:
                successes, failures = provider.anchor_outcomes(candidate)
                if successes + failures >= cfg.min_anchor_outcomes:
                    scored.append((candidate, best_belief(successes, failures, cfg.epsilon)))
            if not scored:
                continue
            # Incumbent is first when it qualifies, so a strict ">" keeps it on ties.
            winner, winner_bb = scored[0]
            for candidate, score in scored[1:]:
                if score > winner_bb:
                    winner, winner_bb = candidate, score
            if winner.evaluator_id != incumbent.evaluator_id:
                epoch[slot] += 1
                frozen[slot] = winner
                current_epoch[slot] = winner.evaluator_id
                erased = self.archive.erase_slot(slot, winner.evaluator_id)
                replacements.append(
                    Replacement(
                        slot=slot,
                        from_id=incumbent.evaluator_id,
                        to_id=winner.evaluator_id,
                        anchor_best_belief=winner_bb,
                        erased=erased,
                        at_eval=at_eval,
                    )
                )

    # -- main loop ---------------------------------------------------------
    def run(self) -> RQGMResult:
        cfg = self.config
        workspace = self.workspace
        archive = self.archive
        roles = workspace.roles()
        role_names = [r.name for r in roles]
        checkpoints = set(
            exponential_checkpoints(cfg.budget, cfg.checkpoint_base, cfg.checkpoint_min)
        )

        if not archive.nodes:
            archive.add_node(
                ArchiveNode(node_id="node_0000", parent_id=None, workspace=workspace.seed())
            )
        self._next_index = len(archive.nodes)

        epoch = {slot: 1 for slot in self.slots}
        frozen: dict[int, EvaluatorCandidate] = {
            slot: provider.incumbent() for slot, provider in self.slots.items()
        }
        current_epoch = {slot: cand.evaluator_id for slot, cand in frozen.items()}

        num_evaluations = 0
        num_expansions = 0
        replacements: list[Replacement] = []

        while num_evaluations < cfg.budget:
            # (1) UCB-Air-gated expansion.
            if num_evaluations**cfg.alpha >= len(archive.nodes):
                parent_id = self._sample_clade(current_epoch)
                child_workspace = workspace.expand(archive.nodes[parent_id])
                if child_workspace is not None:
                    child_id = f"node_{self._next_index:04d}"
                    self._next_index += 1
                    archive.add_node(
                        ArchiveNode(
                            node_id=child_id, parent_id=parent_id, workspace=child_workspace
                        )
                    )
                num_expansions += 1

            # (2) Three-level sampling and evaluation.
            node_id = self._sample_clade(current_epoch)
            role, task = self._least_measured_cell(node_id, roles, current_epoch)
            slot = role.slot
            evaluator = (
                frozen[slot]
                if role.kind == "evaluator_dependent" and slot is not None
                else None
            )
            outcome = int(workspace.evaluate(archive.nodes[node_id], role, task, evaluator))
            dep = (slot,) if evaluator is not None else ()
            tags = {slot: evaluator.evaluator_id} if evaluator is not None else {}
            archive.add_record(
                UtilityRecord(
                    node_id=node_id,
                    role=role.name,
                    task=task,
                    outcome=outcome,
                    dep=dep,
                    criterion_tags=tags,
                    epoch_vector=tuple(epoch[s] for s in sorted(self.slots)),
                )
            )
            num_evaluations += 1

            # (3) Checkpoint: maybe replace evaluators and erase dependents.
            if num_evaluations in checkpoints:
                self._checkpoint(frozen, epoch, current_epoch, replacements, num_evaluations)

        best_node_id = self._select_best(role_names, current_epoch)
        final_successes, final_failures = archive.node_counts(best_node_id, current_epoch)
        return RQGMResult(
            best_node_id=best_node_id,
            best_belief=best_belief(final_successes, final_failures, cfg.epsilon),
            balanced_utility=archive.balanced_utility(best_node_id, role_names, current_epoch),
            archive_size=len(archive.nodes),
            num_evaluations=num_evaluations,
            num_expansions=num_expansions,
            epochs=dict(epoch),
            replacements=replacements,
            records_retained=len(archive.records),
        )

    def _select_best(self, role_names: list[str], current_epoch: dict[int, str]) -> str:
        """Best-belief node among the sufficiently measured; balanced-utility fallback."""
        cfg = self.config
        best_node_id: str | None = None
        best_score = -1.0
        for node_id in self.archive.nodes:
            successes, failures = self.archive.node_counts(node_id, current_epoch)
            if successes + failures < cfg.min_node_evals:
                continue
            score = best_belief(successes, failures, cfg.epsilon)
            if score > best_score:
                best_score = score
                best_node_id = node_id
        if best_node_id is not None:
            return best_node_id
        if self.archive.nodes:
            return max(
                self.archive.nodes,
                key=lambda n: self.archive.balanced_utility(n, role_names, current_epoch),
            )
        return "node_0000"
