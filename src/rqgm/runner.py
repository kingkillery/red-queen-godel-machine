"""High-level helpers to build providers, run a search, and persist results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .archive import Archive
from .llm_providers import LabeledAnchorEvaluatorSlot, PromptEvolutionWorkspace
from .mock_providers import MockEvaluatorSlotProvider, MockWorkspaceProvider
from .providers import EvaluatorSlotProvider, WorkspaceProvider
from .search import RQGMConfig, RQGMResult, RQGMSearch

__all__ = ["build_providers", "persist_result", "result_to_dict", "run_rqgm"]


def build_providers(
    provider: str, config: RQGMConfig, **kwargs: Any
) -> tuple[WorkspaceProvider, dict[int, EvaluatorSlotProvider]]:
    """Construct a workspace + slot providers for a built-in ``provider`` name."""
    if provider == "mock":
        workspace: WorkspaceProvider = MockWorkspaceProvider(seed=config.seed)
        slots: dict[int, EvaluatorSlotProvider] = {0: MockEvaluatorSlotProvider(slot=0)}
        return workspace, slots
    if provider == "llm":
        model = kwargs["model"]
        tasks = kwargs["tasks"]
        reviewer_tasks = kwargs.get("reviewer_tasks") or tasks
        anchor = kwargs["anchor"]
        workspace = PromptEvolutionWorkspace(model, tasks, reviewer_tasks)
        slots = {0: LabeledAnchorEvaluatorSlot(model, anchor, slot=0)}
        return workspace, slots
    raise ValueError(f"unknown provider: {provider!r} (expected 'mock' or 'llm')")


def result_to_dict(result: RQGMResult, run_id: str | None = None) -> dict[str, Any]:
    """Serialize a result (and optional run id) to a JSON-friendly dict."""
    payload = asdict(result)
    if run_id is not None:
        payload = {"run_id": run_id, **payload}
    return payload


def _archive_dump(archive: Archive) -> dict[str, Any]:
    return {
        "nodes": {
            node_id: {
                "parent_id": node.parent_id,
                "children": node.children,
                "workspace": node.workspace,
            }
            for node_id, node in archive.nodes.items()
        },
        "num_records": len(archive.records),
    }


def persist_result(result: RQGMResult, archive: Archive, out_dir: str | Path) -> str:
    """Write ``summary.json``, ``archive.json``, ``replacements.json``; return run id."""
    run_id = "rqgm_" + uuid4().hex[:12]
    base = Path(out_dir) / run_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(
        json.dumps(result_to_dict(result, run_id), indent=2), encoding="utf-8"
    )
    (base / "replacements.json").write_text(
        json.dumps([asdict(r) for r in result.replacements], indent=2), encoding="utf-8"
    )
    (base / "archive.json").write_text(
        json.dumps(_archive_dump(archive), indent=2), encoding="utf-8"
    )
    return run_id


def run_rqgm(
    provider: str = "mock",
    *,
    config: RQGMConfig | None = None,
    out_dir: str | Path | None = None,
    **kwargs: Any,
) -> RQGMResult:
    """Build providers, run an RQGM search, optionally persist, and return the result."""
    config = config or RQGMConfig()
    workspace, slots = build_providers(provider, config, **kwargs)
    search = RQGMSearch(workspace, slots, config)
    result = search.run()
    if out_dir is not None:
        persist_result(result, search.archive, out_dir)
    return result
