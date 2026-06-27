# Contributing

Thanks for your interest in improving the Red Queen Gödel Machine implementation.

## Development setup

```bash
git clone <your-fork-url>
cd red-queen-godel-machine
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev,llm]"
```

## Checks

Run these before opening a pull request:

```bash
pytest -q        # tests must pass (LLM tests run offline via ScriptedChatModel)
ruff check       # lint must be clean
ruff format      # optional: apply formatting
```

The core package (`rqgm.beta`, `rqgm.archive`, `rqgm.providers`, `rqgm.search`,
`rqgm.mock_providers`) must stay **dependency-free**. Do not add third-party
runtime imports to it — keep `openai` confined to `rqgm.llm_providers` behind the
lazy import and the `[llm]` extra.

## Adding a provider

Most integrations are new providers, not changes to the core. Implement the two
structural protocols in `rqgm.providers`:

- `WorkspaceProvider` — `roles()`, `seed()`, `expand(parent)`, and
  `evaluate(node, role, task, evaluator)`.
- `EvaluatorSlotProvider` — a `slot` attribute, `incumbent()`,
  `challengers(archive)`, and `anchor_outcomes(evaluator)`.

Keep evaluations **binary** (1/0) and keep `anchor_outcomes` grounded in fixed
truth — the safety of evaluator co-evolution depends on it. Add a deterministic
test (a scripted/mock model is fine) for any new provider.

## Tests

Every test should defend an observable contract: the Beta math, erasure
semantics, a search invariant, the CLI surface, or provider scoring. Avoid
network in tests — use `ScriptedChatModel` for LLM paths.
