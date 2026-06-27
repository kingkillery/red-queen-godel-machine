"""Generic, OpenAI-compatible LLM providers using prompt evolution.

This is the publishable "real" provider pattern: a node's workspace holds a
coder prompt and a reviewer prompt. The coder role is scored against verifiable
answers (evaluator-independent); the reviewer role is scored by the frozen judge
prompt's Accept/Reject verdict (evaluator-dependent). The meta-agent rewrites one
of the prompts to expand the archive (paper Appendix E observes the gains were
prompt rewrites). Evaluators are grounded by a labeled anchor set.

``ChatModel`` is any callable ``(system, user) -> str``. ``OpenAIChatModel``
works against any OpenAI-compatible endpoint; ``ScriptedChatModel`` is a
deterministic stand-in for tests and examples (no network).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .archive import Archive, ArchiveNode
from .providers import EvaluatorCandidate, RoleSpec

__all__ = [
    "ChatModel",
    "OpenAIChatModel",
    "ScriptedChatModel",
    "Sample",
    "AnchorItem",
    "PromptEvolutionWorkspace",
    "LabeledAnchorEvaluatorSlot",
    "META_PROMPT",
    "SEED_CODER_PROMPT",
    "SEED_REVIEWER_PROMPT",
]

META_PROMPT = (
    "You are a meta-optimizer that improves prompts. Rewrite the prompt below to "
    "be more specific and effective at its task while staying concise. Respond "
    "with ONLY the rewritten prompt and nothing else."
)
SEED_CODER_PROMPT = "Solve the task. Respond with ONLY the final answer, no explanation."
SEED_REVIEWER_PROMPT = (
    "You are a strict judge. Decide whether the answer is correct and well-formed. "
    "Respond with exactly one word: Accept or Reject."
)


@runtime_checkable
class ChatModel(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split()).strip(" .!?\u3002")


def _parse_accept(text: str) -> bool:
    low = text.strip().lower()
    if "reject" in low:
        return False
    return "accept" in low


class OpenAIChatModel:
    """A ``ChatModel`` backed by any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self._client: object | None = None

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "openai is required for OpenAIChatModel; install with "
                    "pip install 'red-queen-godel-machine[llm]'"
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"missing API key: set ${self.api_key_env}")
            self._client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def __call__(self, system: str, user: str) -> str:
        client = self._ensure_client()
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class ScriptedChatModel:
    """Deterministic ``ChatModel`` for tests/examples.

    ``responses`` is either a callable ``(system, user) -> str`` or a sequence of
    canned replies returned in round-robin order.
    """

    def __init__(self, responses: Sequence[str] | Callable[[str, str], str]) -> None:
        self._responses = responses
        self._index = 0

    def __call__(self, system: str, user: str) -> str:
        if callable(self._responses):
            return self._responses(system, user)
        reply = self._responses[self._index % len(self._responses)]
        self._index += 1
        return reply


@dataclass
class Sample:
    task_id: str
    prompt_input: str
    answer: str | None = None


@dataclass
class AnchorItem:
    artifact: str
    label: Literal["Accept", "Reject"]


class PromptEvolutionWorkspace:
    """Workspace whose nodes carry a coder prompt and a reviewer (judge) prompt."""

    def __init__(
        self,
        model: ChatModel,
        tasks: list[Sample],
        reviewer_tasks: list[Sample],
        seed_coder_prompt: str = SEED_CODER_PROMPT,
        seed_reviewer_prompt: str = SEED_REVIEWER_PROMPT,
    ) -> None:
        self.model = model
        self.tasks = tasks
        self.reviewer_tasks = reviewer_tasks
        self.seed_coder_prompt = seed_coder_prompt
        self.seed_reviewer_prompt = seed_reviewer_prompt
        self._tasks_by_id = {t.task_id: t for t in tasks}
        self._reviewer_by_id = {t.task_id: t for t in reviewer_tasks}

    def roles(self) -> list[RoleSpec]:
        return [
            RoleSpec("coder", "evaluator_independent", [t.task_id for t in self.tasks]),
            RoleSpec(
                "reviewer",
                "evaluator_dependent",
                [t.task_id for t in self.reviewer_tasks],
                slot=0,
            ),
        ]

    def seed(self) -> dict:
        return {
            "coder_prompt": self.seed_coder_prompt,
            "reviewer_prompt": self.seed_reviewer_prompt,
        }

    def expand(self, parent: ArchiveNode) -> dict | None:
        workspace = dict(parent.workspace)
        digest = int.from_bytes(hashlib.sha256(parent.node_id.encode()).digest()[:8], "big")
        target = "reviewer_prompt" if digest % 2 else "coder_prompt"
        current = workspace.get(target, "")
        user = (
            f"Task type: {target}\n\nCurrent prompt:\n{current}\n\n"
            f"Recent feedback: {parent.train_feedback or 'none'}"
        )
        try:
            revised = self.model(META_PROMPT, user)
        except Exception:
            return None
        revised = revised.strip()
        if not revised or revised == current:
            return None
        workspace[target] = revised
        return workspace

    def evaluate(
        self,
        node: ArchiveNode,
        role: RoleSpec,
        task: str,
        evaluator: EvaluatorCandidate | None,
    ) -> int:
        coder_prompt = node.workspace.get("coder_prompt", self.seed_coder_prompt)
        try:
            if role.name == "coder":
                sample = self._tasks_by_id[task]
                output = self.model(coder_prompt, sample.prompt_input)
                return int(_normalize(output) == _normalize(sample.answer or ""))
            sample = self._reviewer_by_id[task]
            artifact = self.model(coder_prompt, sample.prompt_input)
            judge_prompt = (
                (evaluator.state.get("prompt") if evaluator else None)
                or self.seed_reviewer_prompt
            )
            decision = self.model(judge_prompt, artifact)
            return int(_parse_accept(decision))
        except Exception:
            return 0


class LabeledAnchorEvaluatorSlot:
    """Slot provider that grounds reviewer prompts against labeled anchors."""

    def __init__(
        self,
        model: ChatModel,
        anchor: list[AnchorItem],
        slot: int = 0,
        seed_reviewer_prompt: str = SEED_REVIEWER_PROMPT,
    ) -> None:
        self.model = model
        self.anchor = anchor
        self.slot = slot
        self.seed_reviewer_prompt = seed_reviewer_prompt

    def incumbent(self) -> EvaluatorCandidate:
        return EvaluatorCandidate("anchor_e0", {"prompt": self.seed_reviewer_prompt})

    def challengers(self, archive: Archive) -> list[EvaluatorCandidate]:
        seen: dict[str, str] = {}
        for node in archive.nodes.values():
            prompt = node.workspace.get("reviewer_prompt")
            if not prompt or prompt == self.seed_reviewer_prompt:
                continue
            seen.setdefault(_stable_id(prompt), prompt)
        return [
            EvaluatorCandidate(candidate_id, {"prompt": prompt})
            for candidate_id, prompt in sorted(seen.items())
        ]

    def anchor_outcomes(self, evaluator: EvaluatorCandidate) -> tuple[int, int]:
        prompt = evaluator.state.get("prompt", self.seed_reviewer_prompt)
        successes = failures = 0
        for item in self.anchor:
            try:
                decision = self.model(prompt, item.artifact)
            except Exception:
                return (0, 0)
            accepted = _parse_accept(decision)
            if accepted == (item.label == "Accept"):
                successes += 1
            else:
                failures += 1
        return successes, failures
