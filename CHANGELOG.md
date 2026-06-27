# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-27

### Added

- Initial implementation of the Red Queen Gödel Machine search algorithm
  (arXiv:2606.26294).
- `rqgm.beta`: Beta best-belief (`BB_ε`) scoring and posterior mean, pure stdlib.
- `rqgm.archive`: utility-record ledger with epoch-aware validity and selective
  erasure.
- `rqgm.providers`: `WorkspaceProvider` and `EvaluatorSlotProvider` protocols.
- `rqgm.search`: the RQGM loop — UCB-Air expansion, three-level sampling,
  Thompson selection over clade best-belief, exponential checkpoints, and
  anchor-based evaluator replacement.
- `rqgm.mock_providers`: deterministic providers for tests and demos.
- `rqgm.llm_providers`: OpenAI-compatible prompt-evolution provider with a
  labeled-anchor evaluator slot and a scripted model for offline tests.
- `rqgm.runner` and the `rqgm` command-line interface (`search`, `inspect`).
- Examples, paper-mapping docs, and a test suite covering the Beta math,
  selective erasure, the search loop, the CLI, and the LLM providers.
