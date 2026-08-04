# Q-TopoMoE Phase 2 W4A16 run status

Updated: 2026-08-04 16:44 CST  
Status: RUNNING (not a completed quantization result)

The retry uses the frozen input without modification:

- model: `/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`
- calibration: `/data/models/test/qtopomoe_calibration/wikitext2_raw_train_256.jsonl`
- calibration SHA-256: `142e9aa4a9821caec5fb10537485d7cf99cbc1bdb7336bf1cb8ab65b5c5e76cf`
- samples / max sequence length: `256 / 4096`
- command PID: `3851061`
- log: `/data/models/test/qtopomoe_quant_runs/w4a16_20260804_retry2.log`
- output target: `/data/models/test/qtopomoe_w4a16`

Observed state at update time:

- llmcompressor initialized `GPTQModifier` and selected `SequentialPipeline`;
- model calibration reached subgraph `(2/41)` at approximately 28%;
- no OOM or traceback after the recursive Accelerate-hook fix;
- no W4A16 checkpoint has been written yet.

Monitor with:

```bash
ps -p 3851061 -o pid,stat,etime,%cpu,%mem,cmd
tail -f /data/models/test/qtopomoe_quant_runs/w4a16_20260804_retry2.log
```

Completion still requires checkpoint existence, the 30,720-expert-scale
coverage audit, and real SGLang/vLLM load and service validation.
