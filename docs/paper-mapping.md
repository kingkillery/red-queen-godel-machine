# Paper → code mapping

This implementation follows *The Red Queen Gödel Machine: Co-Evolving Agents and
Their Evaluators* (arXiv:2606.26294). The table maps the paper's concepts to the
code in `src/rqgm/`.

| Paper concept | Symbol / section | Code |
| --- | --- | --- |
| Best-belief score | `BB_ε(a) = I⁻¹_ε(1+S, 1+F)`, §4 | `rqgm.beta.best_belief` |
| Regularized incomplete Beta / inverse | `I_x(a,b)`, `I⁻¹_q` | `rqgm.beta.regularized_incomplete_beta`, `rqgm.beta.beta_ppf` |
| Posterior mean (working belief) | Prop. 1 | `rqgm.beta.posterior_mean` |
| Utility record `z=(node,role,task,outcome,dep,κ,j)` | App. F | `rqgm.archive.UtilityRecord` |
| Record validity under current evaluators | Def. F.1 | `rqgm.archive.Archive._valid` |
| Selective erasure on replacement | Def. F.2, Prop. 2, Rem. 2 | `rqgm.archive.Archive.erase_slot` |
| Clade metaproductivity (CMP) | §5 | `rqgm.archive.Archive.clade_counts` |
| Balanced utility `U_j` | §5 | `rqgm.archive.Archive.balanced_utility` |
| Epochs & frozen evaluators | §3 | `rqgm.search.RQGMSearch.run` (`epoch`, `frozen`, `current_epoch`) |
| Three-level sampling (node → role → task) | §5 | `RQGMSearch._sample_clade`, `RQGMSearch._least_measured_cell` |
| Thompson sampling over CMP | §5 | `RQGMSearch._sample_clade` (`betavariate`) |
| UCB-Air expansion gate | §5 | `RQGMSearch.run` (`N**alpha >= len(nodes)`) |
| Exponential checkpoints (ρ = 2) | Prop. 6 | `rqgm.search.exponential_checkpoints` |
| Anchor best-belief replacement (ties → incumbent) | §4, Prop. 4 | `RQGMSearch._checkpoint` |
| Evaluator anchors (fixed ground truth) | §4, safety | `EvaluatorSlotProvider.anchor_outcomes` |
| Best-belief node selection / balanced-utility fallback | Alg. 1 | `RQGMSearch._select_best` |

## Notes on faithful-but-pragmatic choices

- **Binary outcomes.** The ledger and scoring assume success/failure outcomes, as
  in the paper's acceptance metric.
- **Prompt evolution as the public self-modification.** App. E observes that the
  measured gains came from prompt rewrites; the shipped LLM provider
  (`rqgm.llm_providers`) therefore evolves coder/reviewer prompts rather than
  editing code, keeping the public package safe and dependency-light.
- **Determinism.** Mock providers derive randomness from SHA-256 of stable keys so
  results reproduce across processes; the search RNG is seeded by `RQGMConfig.seed`.
