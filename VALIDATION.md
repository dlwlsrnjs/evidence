# Validation status

- CPU tests passed: bundled-manifest hashes/counts, image-filename split checks, fixed balanced POPE subset, statistic invariance counterexample, image-cluster CI, default matched settings, paired comparison, summary CLI, overwrite guard.
- `bash -n run_requested.sh` and Python compilation passed.
- Prepared data: historical training pool 2,665; submitted evaluation 250; pool-image-disjoint sensitivity subset 61; matched training pool 2,365; official POPE subset 252 per regime on the same 42 images.
- Actual LLaVA base GPU smoke inference completed for two questions using BF16 Transformers + CPU offload. The saved margins differ from historical Unsloth scores (up to approximately 0.25 nats on this tiny check); do not mix loader outputs. PEC adapter smoke inference is being checked separately. This is a loading/scoring smoke check, not a scientific evaluation result.
- No new full GPU evaluation or training result is bundled in this repository.
