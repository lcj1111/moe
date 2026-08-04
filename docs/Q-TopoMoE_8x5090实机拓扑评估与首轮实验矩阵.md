# Q-TopoMoE 八卡 RTX 5090 实机评估

本文件的预检版已被服务器实测结果取代。请以同目录中的 `Q-TopoMoE_gpu111_phase0实测分析.md` 为准。

关键修正：原先根据静态拓扑提出的 PIX-TP2 假设未通过实测；本机无 CUDA P2P，且 NODE 双卡在中大消息 NCCL AllReduce 中显著快于 PIX 双卡。当前推荐的 TP2 组合为 `(0,2)`、`(1,3)`、`(4,6)`、`(5,7)`。
