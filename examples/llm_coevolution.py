"""LLM-driven prompt co-evolution against an OpenAI-compatible endpoint.

Requires the ``[llm]`` extra (``pip install 'red-queen-godel-machine[llm]'``) and
an API key. By default it uses ``OPENAI_API_KEY`` and the OpenAI endpoint; pass a
``base_url`` to target OpenRouter, a local server, or any OpenAI-compatible API.

Run: ``python examples/llm_coevolution.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from rqgm import AnchorItem, OpenAIChatModel, RQGMConfig, Sample, run_rqgm

DATA = Path(__file__).parent / "data"


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        samples.append(
            Sample(row.get("task_id", f"q{index}"), row["prompt_input"], row.get("answer"))
        )
    return samples


def load_anchor(path: Path) -> list[AnchorItem]:
    items: list[AnchorItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append(AnchorItem(row["artifact"], row["label"]))
    return items


def main() -> None:
    model = OpenAIChatModel("gpt-4o-mini")  # or OpenAIChatModel("...", base_url="...")
    result = run_rqgm(
        "llm",
        config=RQGMConfig(budget=48, seed=0),
        model=model,
        tasks=load_samples(DATA / "tasks.jsonl"),
        anchor=load_anchor(DATA / "anchor.jsonl"),
    )
    print("best node    :", result.best_node_id)
    print("best belief  :", round(result.best_belief, 4))
    print("epochs       :", result.epochs)
    print("replacements :", [(r.from_id, r.to_id) for r in result.replacements])


if __name__ == "__main__":
    main()
