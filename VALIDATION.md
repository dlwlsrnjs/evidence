# Validation status

- CPU tests passed: bundled-manifest hashes/counts, image-filename split checks, fixed balanced POPE subset, statistic invariance counterexample, image-cluster CI, default matched settings, paired comparison, summary CLI, overwrite guard.
- `bash -n run_requested.sh` and Python compilation passed.
- Packaged original_250 image/question/label identities and order exactly match the submitted evaluation file.
- Prepared data: historical training pool 2,665; submitted evaluation 250; pool-image-disjoint sensitivity subset 61; matched training pool 2,365; official POPE subset 252 per regime on the same 42 images.
- Actual GPU smoke inference completed for LLaVA-1.5-7B base AND the existing PEC seed-0 adapter: two questions each, full and gray endpoints, BF16 Transformers + CPU offload, no quantization. All output files completed.
- The HF base margins differ from historical Unsloth scores by up to approximately 0.25 nats on this tiny check. PEC margins differed by up to approximately 0.375 nats on these two questions. Do not mix loader outputs or infer full-dataset equivalence from this smoke test.
- CPU offload was substantially slower than dedicated-GPU inference; use a free GPU for the 25-hour full experiment plan.
- Full endpoint/random evaluations, Qwen2.5 GPU smoke, and new GPU training have NOT been run as part of this package validation. No new scientific result is claimed from a two-question smoke test.
