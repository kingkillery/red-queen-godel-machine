"""End-to-end CLI behavior, including cross-process determinism."""

import json
import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rqgm.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_mock_json_keys_and_cross_process_determinism():
    first = _run(["search", "--provider", "mock", "--budget", "64", "--seed", "0", "--json"])
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    for key in (
        "best_node_id",
        "best_belief",
        "balanced_utility",
        "archive_size",
        "num_evaluations",
        "num_expansions",
        "epochs",
        "replacements",
        "records_retained",
    ):
        assert key in payload

    second = _run(["search", "--provider", "mock", "--budget", "64", "--seed", "0", "--json"])
    assert second.returncode == 0, second.stderr
    # Stable-hash mock providers => identical best node from a separate process.
    assert json.loads(second.stdout)["best_node_id"] == payload["best_node_id"]


def test_cli_persist_and_inspect(tmp_path):
    out = tmp_path / "runs"
    run = _run(
        ["search", "--provider", "mock", "--budget", "32", "--seed", "1"]
        + ["--json", "--out", str(out)]
    )
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["run_id"].startswith("rqgm_")

    inspect = _run(["inspect", payload["run_id"], "--root", str(out)])
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(inspect.stdout)["best_node_id"] == payload["best_node_id"]
