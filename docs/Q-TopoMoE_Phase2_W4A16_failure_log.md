# W4A16 execution log

The first real W4A16 attempt used 256 frozen calibration records and loaded the
35B checkpoint, but failed before calibration at compressed-tensors hook
initialization:

```text
AttributeError: 'functools.partial' object has no attribute '__func__'
```

Cause: `device_map="auto"` in Transformers/Accelerate installs partial forward
hooks, while compressed-tensors 0.17.1 expects a bound method when wrapping a
Linear module. No checkpoint was written and all GPU memory returned to idle.

The entrypoint now removes Accelerate hooks recursively after loading and lets
the llmcompressor calibration pipeline install its own compressed-tensors
offload hooks. The fix is in commit `0f473f1`; the next retry additionally
includes the recursive hook removal and is expected to re-run the same frozen
input without changing its hash or sample indices.
