"""Command-line interface for RQGM (``rqgm search`` / ``rqgm inspect``).

Uses only the standard library so the core package stays dependency-free. The
``llm`` provider constructs an :class:`OpenAIChatModel` and loads task/anchor
JSONL files; it requires the optional ``[llm]`` extra (the ``openai`` package).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm_providers import AnchorItem, OpenAIChatModel, Sample
from .runner import build_providers, persist_result, result_to_dict
from .search import RQGMConfig, RQGMSearch


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_samples(path: str) -> list[Sample]:
    samples: list[Sample] = []
    for index, row in enumerate(_load_jsonl(path)):
        samples.append(
            Sample(
                task_id=str(row.get("task_id", f"q{index}")),
                prompt_input=str(row.get("prompt_input", row.get("input", ""))),
                answer=row.get("answer"),
            )
        )
    return samples


def _load_anchor(path: str) -> list[AnchorItem]:
    items: list[AnchorItem] = []
    for row in _load_jsonl(path):
        label = "Accept" if str(row.get("label", "")).lower().startswith("a") else "Reject"
        items.append(AnchorItem(artifact=str(row.get("artifact", "")), label=label))
    return items


def _cmd_search(args: argparse.Namespace) -> int:
    config = RQGMConfig(
        budget=args.budget,
        epsilon=args.epsilon,
        alpha=args.alpha,
        checkpoint_base=args.checkpoint_base,
        seed=args.seed,
    )
    kwargs: dict = {}
    if args.provider == "llm":
        if not args.dataset or not args.anchor:
            print("error: --provider llm requires --dataset and --anchor", file=sys.stderr)
            return 2
        model = OpenAIChatModel(model=args.model or "gpt-4o-mini", base_url=args.base_url)
        kwargs = {
            "model": model,
            "tasks": _load_samples(args.dataset),
            "anchor": _load_anchor(args.anchor),
        }
    workspace, slots = build_providers(args.provider, config, **kwargs)
    search = RQGMSearch(workspace, slots, config)
    result = search.run()
    run_id = persist_result(result, search.archive, args.out) if args.out else None

    if args.json:
        print(json.dumps(result_to_dict(result, run_id), indent=2))
    else:
        if run_id:
            print(f"run_id          {run_id}")
        print(f"best_node       {result.best_node_id}")
        print(f"best_belief     {result.best_belief:.4f}")
        print(f"balanced_util   {result.balanced_utility:.4f}")
        print(f"archive_size    {result.archive_size}")
        print(f"evaluations     {result.num_evaluations}")
        print(f"expansions      {result.num_expansions}")
        print(f"epochs          {result.epochs}")
        print(f"replacements    {len(result.replacements)}")
        for rep in result.replacements:
            print(
                f"  slot {rep.slot}: {rep.from_id} -> {rep.to_id} "
                f"(BB={rep.anchor_best_belief:.4f}, erased={rep.erased}, @{rep.at_eval})"
            )
        print(f"records_retained {result.records_retained}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    summary = Path(args.root) / args.run_id / "summary.json"
    if not summary.exists():
        print(f"error: no run found at {summary}", file=sys.stderr)
        return 1
    print(summary.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rqgm", description="Red Queen Godel Machine search")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="run a co-evolutionary search")
    search.add_argument("--provider", choices=["mock", "llm"], default="mock")
    search.add_argument("--budget", type=int, default=256)
    search.add_argument("--epsilon", type=float, default=0.05)
    search.add_argument("--alpha", type=float, default=0.6)
    search.add_argument("--checkpoint-base", dest="checkpoint_base", type=int, default=2)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--out", default=None, help="directory to persist run artifacts")
    search.add_argument("--json", action="store_true", help="emit JSON summary")
    search.add_argument("--model", default=None, help="model id for --provider llm")
    search.add_argument(
        "--base-url", dest="base_url", default=None, help="OpenAI-compatible base url"
    )
    search.add_argument("--dataset", default=None, help="coder tasks JSONL for --provider llm")
    search.add_argument("--anchor", default=None, help="labeled anchor JSONL for --provider llm")
    search.set_defaults(func=_cmd_search)

    inspect = sub.add_parser("inspect", help="print a persisted run summary")
    inspect.add_argument("run_id")
    inspect.add_argument("--root", default="runs/rqgm", help="runs root directory")
    inspect.set_defaults(func=_cmd_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
