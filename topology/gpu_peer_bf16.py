#!/usr/bin/env python3
import argparse
import json
import platform
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = torch.cuda.device_count()
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "arch_list": torch.cuda.get_arch_list(),
        "device_count": count,
        "devices": [],
        "peer_access": [],
        "bf16": [],
    }

    if count != 8:
        raise RuntimeError(f"expected 8 GPUs, found {count}")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        result["devices"].append({
            "index": i,
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(i)),
            "total_memory": props.total_memory,
        })
        with torch.cuda.device(i):
            torch.manual_seed(1000 + i)
            a = torch.randn((2048, 2048), device=i, dtype=torch.bfloat16)
            b = torch.randn((2048, 2048), device=i, dtype=torch.bfloat16)
            c = a @ b
            torch.cuda.synchronize(i)
            result["bf16"].append({
                "index": i,
                "dtype": str(c.dtype),
                "shape": list(c.shape),
                "finite": bool(torch.isfinite(c).all().item()),
                "checksum": float(c.float().mean().item()),
            })
            del a, b, c

    peer_fn = getattr(torch.cuda, "can_device_access_peer", None)
    for i in range(count):
        row = []
        for j in range(count):
            if i == j:
                row.append(True)
            elif peer_fn is None:
                row.append(None)
            else:
                row.append(bool(peer_fn(i, j)))
        result["peer_access"].append(row)

    result["all_bf16_pass"] = all(x["finite"] for x in result["bf16"])
    result["all_off_diagonal_peer_false"] = all(
        result["peer_access"][i][j] is False
        for i in range(count) for j in range(count) if i != j
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(output),
        "device_count": count,
        "all_bf16_pass": result["all_bf16_pass"],
        "all_off_diagonal_peer_false": result["all_off_diagonal_peer_false"],
    }, indent=2))
    return 0 if result["all_bf16_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
