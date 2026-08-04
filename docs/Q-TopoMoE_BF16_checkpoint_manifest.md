# BF16 checkpoint manifest

本机 BF16 checkpoint 已补齐，项目环境变量固定为：

```bash
QTOPOMOE_BF16_MODEL=/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master
```

用户提供的 ModelScope 模型根目录为：
`/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B`。

执行前核验：

- snapshot：`master`
- checkpoint 分片：26 个 `model-*.safetensors`
- 总大小：约 67 GB
- `config.json` SHA256：`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`
- `model.safetensors.index.json` SHA256：`41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`

BF16 矩阵运行前仍需执行服务端真实加载、模型发现、completion 和 metrics 验收；本文件只记录 checkpoint 身份，不代表服务已经通过。
