# Q-TopoMoE Phase 2: Frozen Open Calibration Set

Date: 2026-08-04  
Host: gpu-111  
Status: calibration input PASS; W4A16 execution not started yet

## Candidate assessment

The selected source is `Salesforce/wikitext`, subset `wikitext-2-raw-v1`,
split `train`. The Hub page documents the dataset as Wikipedia-derived
language-modeling text and lists CC BY-SA/GFDL licensing. A fixed Hub commit
is used rather than a mutable `main` pointer. WikiText is general English
text, not an instruction/chat benchmark; it is suitable for weight and MoE
expert calibration, but it must not be used to claim chat-quality gains.

This is more controllable for the present reproduction than C4: C4 is ODC-BY,
is derived from Common Crawl, and is many terabytes in the Hub card. The
selection is therefore feasible, small enough to archive locally, and easy to
rebuild. License obligations still apply to any redistribution.

## Frozen source and preprocessing

| Field | Value |
|---|---|
| Dataset repository | `Salesforce/wikitext` |
| Revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Config / split | `wikitext-2-raw-v1` / `train` |
| Source parquet SHA-256 | `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` |
| Source rows | 36,718 |
| Selection seed | 42 |
| Records | 256 |
| Target tokens per record | 2,048 before chat wrapping |
| Chat-wrapped token range | 2,067–2,544 |
| Calibration JSONL SHA-256 | `142e9aa4a9821caec5fb10537485d7cf99cbc1bdb7336bf1cb8ab65b5c5e76cf` |

The generator shuffles source row IDs with `random.Random(42)`, packs real
rows until the target token budget is reached, and stores the exact source row
IDs in every record. It never duplicates the 64-record smoke seed.

## Tokenizer and template freeze

Tokenizer root:

`/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`

| File / value | SHA-256 |
|---|---|
| `tokenizer_config.json` | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `vocab.json` | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `merges.txt` | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| Qwen3.6 chat template | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |

The full sample-index list and template text are tracked in
`docs/Q-TopoMoE_Phase2_WikiText_calibration_manifest.json`.

## Server locations and checks

```text
/data/models/test/qtopomoe_wikitext/wikitext-2-raw-v1-train.parquet
/data/models/test/qtopomoe_calibration/wikitext2_raw_train_256.jsonl
/data/models/test/qtopomoe_calibration/wikitext2_raw_train_256.manifest.json
```

The formal preflight passed with 256 records, the Qwen3.6 MoE metadata,
8 visible GPUs, and the pinned quantization environment. The raw parquet and
JSONL remain server-side because the repository ignores large/raw data files;
the pinned revision, source hash, preprocessing script, sample indices, and
output hash are tracked so the files can be regenerated.

Next command after final review:

```bash
cd /home/k8s-ops/moe
source env/project.env
/data/models/test/qtopomoe_quant_env/bin/python \
  quantization/quantize_w4a16.py \
  --model "$QTOPOMOE_BF16_MODEL" \
  --calibration "$QTOPOMOE_CALIBRATION_JSONL" \
  --output /data/models/test/qtopomoe_w4a16 \
  --samples 256 --max-seq-length 4096
```

Quantization output is not yet claimed. It still requires the Runbook 6.4
coverage audit and 6.5 real SGLang/vLLM load, completion, metrics, and
concurrency validation.
