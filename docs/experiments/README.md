# Experiments

One write-up per decision made on evidence. Each records the question, the two
configurations, the numbers with their confidence intervals, the cost, and the decision.
Reports live in `data/samples/eval-*.json` (git-ignored, reproducible with `make dataset`
plus the `make eval*` targets) and every run is tracked in MLflow (`make mlflow-ui`).

| # | Question | Decision |
|---|---|---|
| [0001](0001-rerank-on-vs-off.md) | Does an LLM rerank of the fused candidates improve answers? | No measurable difference; off by default |
| [0002](0002-lora-finetune-plan.md) | Would a LoRA fine-tune of a small model beat prompting for extraction? | Planned, not run |

Rules: the judge is never the model under test; A/B decisions are made on paired bootstrap
intervals over the same items, never on two point estimates; a decision is recorded even
when the answer is "no difference".
