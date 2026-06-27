"""LLM provider logic, exercised offline via a scripted model (no network)."""

from rqgm.archive import ArchiveNode
from rqgm.llm_providers import (
    AnchorItem,
    LabeledAnchorEvaluatorSlot,
    PromptEvolutionWorkspace,
    Sample,
    ScriptedChatModel,
)
from rqgm.providers import EvaluatorCandidate


def test_coder_exact_match_scoring():
    model = ScriptedChatModel(lambda system, user: "4" if user.startswith("2+2") else "wrong")
    workspace = PromptEvolutionWorkspace(
        model, [Sample("q0", "2+2=", "4"), Sample("q1", "9-1=", "8")], []
    )
    node = ArchiveNode("n0", None, workspace=workspace.seed())
    coder = workspace.roles()[0]
    assert workspace.evaluate(node, coder, "q0", None) == 1  # "4" matches
    assert workspace.evaluate(node, coder, "q1", None) == 0  # "wrong" != "8"


def test_reviewer_judge_parsing():
    def fn(system: str, user: str) -> str:
        if "judge" in system.lower():
            return "Accept" if user.strip() == "ok" else "Reject"
        return user.replace("make:", "").strip()  # coder turns the task into an artifact

    workspace = PromptEvolutionWorkspace(
        ScriptedChatModel(fn), [], [Sample("r0", "make:ok"), Sample("r1", "make:bad")]
    )
    node = ArchiveNode("n0", None, workspace=workspace.seed())
    reviewer = workspace.roles()[1]
    judge = EvaluatorCandidate("e0", {"prompt": "You are a judge. Accept or Reject."})
    assert workspace.evaluate(node, reviewer, "r0", judge) == 1
    assert workspace.evaluate(node, reviewer, "r1", judge) == 0


def test_anchor_outcomes_scoring():
    model = ScriptedChatModel(lambda system, user: "Accept" if user.strip() == "good" else "Reject")
    slot = LabeledAnchorEvaluatorSlot(
        model,
        [
            AnchorItem("good", "Accept"),  # judged Accept, gold Accept -> success
            AnchorItem("bad", "Reject"),  # judged Reject, gold Reject -> success
            AnchorItem("good", "Reject"),  # judged Accept, gold Reject -> failure
        ],
    )
    assert slot.anchor_outcomes(EvaluatorCandidate("e0", {"prompt": "j"})) == (2, 1)


def test_expand_rewrites_a_prompt():
    workspace = PromptEvolutionWorkspace(
        ScriptedChatModel(["REVISED PROMPT"]),
        [Sample("q0", "x", "y")],
        [Sample("r0", "z")],
    )
    parent = ArchiveNode("node_0000", None, workspace=workspace.seed())
    child = workspace.expand(parent)
    assert child is not None
    assert child != parent.workspace
    assert "REVISED PROMPT" in (child["coder_prompt"], child["reviewer_prompt"])


def test_challengers_dedupe_and_exclude_seed():
    model = ScriptedChatModel(["unused"])
    slot = LabeledAnchorEvaluatorSlot(model, [])
    archive_nodes = {
        "a": ArchiveNode("a", None, workspace={"reviewer_prompt": slot.seed_reviewer_prompt}),
        "b": ArchiveNode("b", None, workspace={"reviewer_prompt": "evolved judge prompt"}),
        "c": ArchiveNode("c", None, workspace={"reviewer_prompt": "evolved judge prompt"}),
    }

    class _Archive:
        nodes = archive_nodes

    challengers = slot.challengers(_Archive())
    assert len(challengers) == 1  # seed excluded, duplicate collapsed
    assert challengers[0].state["prompt"] == "evolved judge prompt"
