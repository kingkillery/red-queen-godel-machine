"""A minimal hand-written RQGM integration (no mock helpers, no LLM).

Shows the two protocols end to end:

* ``SkillWorkspace`` — nodes hold an integer "skill"; the ``solve`` role succeeds
  with probability ``skill / 10`` (halved on the hard task), and expansion nudges
  skill upward. The ``review`` role is evaluator-dependent on slot 0.
* ``FixedJudgeSlot`` — a single, stable evaluator with fixed anchors and no
  challengers, so no replacement happens (the simplest possible slot).

Run: ``python examples/custom_provider.py``.
"""

from __future__ import annotations

import random

from rqgm import EvaluatorCandidate, RoleSpec, RQGMConfig, RQGMSearch
from rqgm.archive import Archive, ArchiveNode

_rng = random.Random(0)


class SkillWorkspace:
    def roles(self) -> list[RoleSpec]:
        return [
            RoleSpec("solve", "evaluator_independent", ["easy", "hard"]),
            RoleSpec("review", "evaluator_dependent", ["r0"], slot=0),
        ]

    def seed(self) -> dict:
        return {"skill": 1}

    def expand(self, parent: ArchiveNode) -> dict | None:
        return {"skill": min(parent.workspace["skill"] + 1, 10)}

    def evaluate(self, node, role, task, evaluator) -> int:
        probability = node.workspace["skill"] / 10.0
        if task == "hard":
            probability *= 0.5
        if evaluator is not None:
            # An evaluator-dependent outcome could weight by the judge; here it is
            # just slightly stricter to illustrate the dependency.
            probability *= 0.9
        return int(_rng.random() < probability)


class FixedJudgeSlot:
    slot = 0

    def incumbent(self) -> EvaluatorCandidate:
        return EvaluatorCandidate("judge_v0")

    def challengers(self, archive: Archive) -> list[EvaluatorCandidate]:
        return []

    def anchor_outcomes(self, evaluator: EvaluatorCandidate) -> tuple[int, int]:
        return (5, 0)


def main() -> None:
    result = RQGMSearch(
        SkillWorkspace(), {0: FixedJudgeSlot()}, RQGMConfig(budget=120, seed=0)
    ).run()
    print("best node     :", result.best_node_id)
    print("best belief   :", round(result.best_belief, 4))
    print("archive size  :", result.archive_size)
    print("replacements  :", len(result.replacements), "(none expected: fixed judge)")


if __name__ == "__main__":
    main()
