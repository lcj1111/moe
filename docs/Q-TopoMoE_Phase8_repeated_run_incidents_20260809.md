# Phase 8 repeated-run pre-measurement incidents (2026-08-09)

No result from the three directories below is admitted to the repeated
comparison. Each attempt stopped before a workload cell was measured, kept its
failure metadata and logs, and released all selected GPUs.

| Attempt | Evidence on gpu-111 | Gate result | Root cause | Resolution |
|---|---|---|---|---|
| v1 | `/data/models/test/qtopomoe_phase8_repeated_v1` | rejected before health | runner default selected legacy vLLM 0.26.0 instead of the frozen 0.26.1rc build | exact cleanroom vLLM version and entrypoint SHA Gate, commit `6e790b3` |
| v2 | `/data/models/test/qtopomoe_phase8_repeated_v2` | rejected during engine startup | absolute venv entrypoint did not prepend its `bin` directory to `PATH`; FlashInfer JIT could not find the already-installed `ninja` | frozen venv PATH activation plus ninja path/SHA Gate, commit `c941881` |
| v3 | `/data/models/test/qtopomoe_phase8_repeated_v3` | FP8 rejected after a successful 32-request warmup | backend Gate expected quotes that the official log does not emit; actual line was `Using TRITON Fp8 MoE backend` | stable exact substring plus forbidden `Unknown SF transformation` Gate, commit `e4ee0b8` |

Before the v3 FP8 text-Gate rejection, NVFP4 EP8, W4A16 EP4 and NVFP4 EP4
each completed 12/12 cells with zero request failures. They are intentionally
not merged into the final dataset because v4 is a clean, single-plan restart.

The admitted run root is `/data/models/test/qtopomoe_phase8_repeated_v4`.
Its schedule records frozen vLLM and ninja hashes before launching any service.
