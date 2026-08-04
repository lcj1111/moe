# Data freeze status

The current machine-local smoke seed is copied to `data/manifests/sharegpt_seed.json` by `scripts/freeze_baseline.sh` and verified by SHA-256. It contains 64 records and is frozen for service smoke and screening only.

The formal 256-sample calibration set and benchmark evaluation manifests are intentionally not fabricated from the 64-record smoke seed. They must be sourced from a pinned dataset revision, then stored with the dataset revision, sample indices, tokenizer revision, chat template, seed, and SHA-256 before formal quantization or quality results.
