"""Deterministic RQGM co-evolution with the built-in mock providers.

Run: ``python examples/quickstart_mock.py`` (no API key, no extra deps).
"""

from __future__ import annotations

from rqgm import RQGMConfig, run_rqgm


def main() -> None:
    result = run_rqgm("mock", config=RQGMConfig(budget=128, seed=0))
    print("best node       :", result.best_node_id)
    print("best belief     :", round(result.best_belief, 4))
    print("balanced utility:", round(result.balanced_utility, 4))
    print("archive size    :", result.archive_size)
    print("epochs          :", result.epochs)
    print("replacements    :")
    for rep in result.replacements:
        print(
            f"  slot {rep.slot}: {rep.from_id} -> {rep.to_id} "
            f"(erased {rep.erased} dependent records @ eval {rep.at_eval})"
        )
    print("records retained:", result.records_retained)


if __name__ == "__main__":
    main()
