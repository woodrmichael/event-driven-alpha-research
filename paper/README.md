# Study and implementation

[Read the final paper](final_paper.pdf) · [View the presentation](../presentation/final_presentation.pdf)

The paper contains the full experimental design, results, uncertainty, and limitations. The public code covers the final volatility comparison and the option-execution mechanics.

| Module | Role |
| --- | --- |
| `final_ranking_increment.py` | Feature groups, point-in-time checks, train-fitted Ridge/TF-IDF pipelines, and matched evaluation |
| `final_ranking_core.py` | Expanding chronological splits, embargo and duplicate purges, metrics, clustered inference, and claim decisions |
| `final_ranking_inference.py` | Paired comparisons and multiple-testing adjustment |
| `execution_options_engine.py` | Exact-contract quote selection, common multi-leg states, fill evidence, and bid/ask cashflows |
| `week4_local_data.py`, `week5_event_taxonomy.py`, `week6_volatility_causality.py` | Required input, event-taxonomy, feature-definition, and realized-variance helpers |

These modules are in [src/event_driven_alpha/analysis](../src/event_driven_alpha/analysis/). Their original names are retained to preserve imports. Model and execution logic matches the research snapshot; the public CLI defaults to the final expanded-options study. Tests verify source identity while allowing that one default-path change.

The [frozen specification](../configs/final_ranking_increment_marketdata_20260809_v1/research_specification.yaml) records the study's target, folds, estimators, seeds, and decision rule. Its hash is retained. The research CLI requires the historical inputs; the synthetic example runs without them.

The sample depends on historical data coverage, the evaluation is retrospective, and the option results describe simulated execution rather than actual broker fills.
