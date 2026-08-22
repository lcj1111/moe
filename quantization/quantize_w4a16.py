#!/usr/bin/env python3
"""Run the guarded Qwen3.6 MoE W4A16 PTQ recipe.

The script refuses to start unless a pinned calibration manifest contains the
full requested sample budget. It uses llmcompressor's supported MoE
linearization context so fused 3-D expert weights are quantized as individual
Linear modules and saved as compressed-tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from datasets import Dataset
from accelerate.hooks import remove_hook_from_module
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from llmcompressor.modifiers.gptq import GPTQModifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, tokenizer) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(payload, list):
        raise ValueError("calibration manifest must be a JSON list or JSONL")

    rows = []
    for item in payload:
        if isinstance(item, str):
            text = item
        elif item.get("text"):
            text = str(item["text"])
        elif item.get("conversations"):
            messages = []
            for turn in item["conversations"]:
                source = turn.get("from", "user")
                role = "assistant" if source in {"gpt", "assistant", "bot"} else "user"
                messages.append({"role": role, "content": str(turn.get("value", ""))})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            raise ValueError("each calibration record needs text or conversations")
        rows.append({"text": text})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not model_path.is_dir() or not calibration_path.is_file():
        raise SystemExit("model and calibration paths must exist")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rows = load_rows(calibration_path, tokenizer)
    if len(rows) < args.samples:
        raise SystemExit(
            f"refusing to quantize: calibration has {len(rows)} records, "
            f"requires {args.samples}"
        )
    rows = rows[: args.samples]
    dataset = Dataset.from_list(rows)

    # Keep router/shared-expert gating, embeddings, and lm_head in high
    # precision. Expert gate_proj remains quantizable; only routing gates are
    # excluded by name.
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "re:.*linear_attn.*",
            "re:.*router.*",
            "re:.*shared_expert_gate.*",
            "re:.*embed_tokens.*",
            "lm_head",
        ],
        block_size=128,
        dampening_frac=0.01,
        offload_hessians=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    with load_quantizable_moe(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # Transformers/Accelerate installs functools.partial forward hooks for
        # device_map="auto". Remove those hooks before llmcompressor wraps
        # Linear.forward; compressed-tensors 0.17.1 expects a bound method and
        # cannot access __func__ on Accelerate's partial. The calibration
        # pipeline installs its own compressed-tensors offload hooks later.
        if hasattr(model, "hf_device_map"):
            remove_hook_from_module(model, recurse=True)
            delattr(model, "hf_device_map")
        oneshot(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            recipe=recipe,
            num_calibration_samples=args.samples,
            max_seq_length=args.max_seq_length,
            text_column="text",
            data_collator="truncation",
            pad_to_max_length=False,
            shuffle_calibration_samples=False,
            moe_calibrate_all_experts=True,
            pipeline="independent",
            sequential_offload_device="cpu",
            dataloader_num_workers=0,
            save_compressed=True,
            output_dir=output_path,
        )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "format": "compressed-tensors",
        "scheme": "W4A16",
        "model": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "calibration_records": args.samples,
        "max_seq_length": args.max_seq_length,
        "moe_calibrate_all_experts": True,
        "targets": "Linear",
        "ignore": recipe.ignore,
        "git_commit": commit,
    }
    (output_path / "qtopomoe_quant_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
