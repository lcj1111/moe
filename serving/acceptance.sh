#!/usr/bin/env bash
# 作用：对已启动推理服务执行健康、模型、smoke 和结果准入。
set -euo pipefail

: "${ROOT_URL:?ROOT_URL is required, e.g. http://127.0.0.1:31001}"
: "${SERVED_NAME:?SERVED_NAME is required}"
: "${OUT_DIR:?OUT_DIR is required}"

API_URL="${API_URL:-${ROOT_URL%/}/v1}"

mkdir -p "$OUT_DIR"
curl -fsS --max-time 30 "${ROOT_URL%/}/health" > "$OUT_DIR/health.txt"
curl -fsS --max-time 30 "${API_URL%/}/models" > "$OUT_DIR/models.json"
curl -fsS --max-time 120 "${API_URL%/}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 - "$SERVED_NAME" <<'PY'
import json, sys
model = sys.argv[1]
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "只回答数字：6乘以7等于多少？"}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
  )" > "$OUT_DIR/completion.json"
curl -fsS --max-time 30 "${ROOT_URL%/}/metrics" > "$OUT_DIR/metrics.prom"

python3 - "$OUT_DIR" "$SERVED_NAME" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
served = sys.argv[2]
models = json.loads((out / "models.json").read_text())
ids = {x.get("id") for x in models.get("data", [])}
if served not in ids:
    raise SystemExit(f"model discovery failed: {served!r} not in {sorted(ids)!r}")
completion = json.loads((out / "completion.json").read_text())
choices = completion.get("choices") or []
if not choices or not choices[0].get("message", {}).get("content"):
    raise SystemExit("completion response has no assistant content")
if (out / "metrics.prom").stat().st_size == 0:
    raise SystemExit("metrics response is empty")
summary = {
    "health": True,
    "model_discovery": True,
    "completion": True,
    "metrics": True,
    "served_name": served,
    "completion_text": choices[0]["message"]["content"],
}
(out / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY
