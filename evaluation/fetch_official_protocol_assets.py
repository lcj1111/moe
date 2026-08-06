#!/usr/bin/env python3
"""Fetch pinned few-shot assets for official-like quality protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
import urllib.parse
import urllib.request

MMLU_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_VALIDATION = "data/validation-00000-of-00001.parquet"
CEVAL_REPO = "ceval/ceval-exam"
CEVAL_PARQUET_REVISION = "8267189d6ba0d516d414a98919558032958c4466"
USER_AGENT = "qtopomoe-official-protocol-freezer/1.0"


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = pathlib.Path(handle.name)
    temporary.replace(path)


def resolve_url(repo: str, revision: str, path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    records = []

    mmlu_url = resolve_url(MMLU_REPO, MMLU_REVISION, MMLU_VALIDATION)
    mmlu_data = request_bytes(mmlu_url)
    mmlu_path = output / "mmlupro_validation.parquet"
    atomic_write(mmlu_path, mmlu_data)
    records.append({
        "dataset": "mmlu_pro", "revision": MMLU_REVISION,
        "source_path": MMLU_VALIDATION, "output_path": mmlu_path.name,
        "bytes": len(mmlu_data), "sha256": sha256(mmlu_data), "url": mmlu_url,
    })

    api_url = (
        f"https://huggingface.co/api/datasets/{CEVAL_REPO}/tree/"
        f"{CEVAL_PARQUET_REVISION}?recursive=true&expand=false"
    )
    tree = json.loads(request_bytes(api_url))
    def fetch_ceval_split(tree: list[dict], split: str) -> list[dict]:
        files = sorted(
            (
                item for item in tree
                if item.get("type") == "file"
                and f"/{split}/" in item.get("path", "")
                and item["path"].endswith(".parquet")
            ),
            key=lambda item: item["path"],
        )
        if len(files) != 52:
            raise SystemExit(
                f"expected 52 C-Eval {split} files, got {len(files)}"
            )
        return files

    for split in ("dev", "test"):
        split_files = fetch_ceval_split(tree, split)
        for item in split_files:
            source_path = item["path"]
            subject = source_path.split("/", 1)[0]
            url = resolve_url(CEVAL_REPO, CEVAL_PARQUET_REVISION, source_path)
            data = request_bytes(url)
            if len(data) != int(item["size"]):
                raise RuntimeError(f"size mismatch for {source_path}")
            destination = output / f"ceval_{split}" / f"{subject}.parquet"
            atomic_write(destination, data)
            records.append({
                "dataset": "ceval", "revision": CEVAL_PARQUET_REVISION,
                "source_path": source_path, "split": split,
                "output_path": str(destination.relative_to(output)),
                "bytes": len(data), "sha256": sha256(data), "url": url,
            })

    mmlu_test_url = resolve_url(MMLU_REPO, MMLU_REVISION, "data/test-00000-of-00001.parquet")
    mmlu_test_data = request_bytes(mmlu_test_url)
    mmlu_test_path = output / "mmlupro_test.parquet"
    atomic_write(mmlu_test_path, mmlu_test_data)
    records.append({
        "dataset": "mmlu_pro", "revision": MMLU_REVISION,
        "source_path": "data/test-00000-of-00001.parquet",
        "output_path": mmlu_test_path.name,
        "bytes": len(mmlu_test_data), "sha256": sha256(mmlu_test_data),
        "url": mmlu_test_url,
    })

    manifest = {
        "schema_version": "qtopomoe.official_protocol_assets.v1",
        "files": records,
    }
    manifest_path = output / "official_protocol_assets_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(records), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
