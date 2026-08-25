# Q-TopoMoE 数据与产物清单

本清单记录 `configs/`、`docs/` 与 `data/` 下所有已跟踪的机器可读产物。集中为一个文件而不是每个数据文件各放一份 README，既减少文件数量，也不隐藏来源信息。

## 条目阅读方式

- **冻结输入 / manifest**：由复现或评测命令消费；不得原地编辑。路径与 SHA-256 共同锁定输入。
- **实验计划**：可执行或半可执行的控制数据；必须保留路径，因为脚本和 Runbook 可能直接引用。
- **证据 / 结果**：记录的测量、审计、Gate 或摘要；除非 manifest 明确指向，否则不是运行时输入。
- **归档证据**：仅用于历史比较；为保证可审计性保留，不得用作当前 Gate。
- 原始 benchmark 大文件不进入 Git；需要时由 manifest 记录预期位置和哈希。

数据文件名保留日期和版本，因为它们是来源信息而不是冗余。人类可读的阶段归档见 `docs/results/phase*.md`；没有改名或删除任何机器可读路径。

本次提交完成后的机器可读产物数量： **143**

| 路径 | 类型 / 数据说明 | 复现指引 | SHA-256 |
|---|---|---|---|
| `configs/communication/nccl_cost_db.json` | 硬件 / 通信测量 | 用于拓扑、P2P、NCCL 或成本决策的证据；不是模型 checkpoint。 | `50140B47B46E7B6A887697153968E0C36D268C282B363AC15BEF6A1DD71BD468` |
| `configs/communication/nccl_cost_db_nop2p_20260803.json` | 硬件 / 通信测量 | 用于拓扑、P2P、NCCL 或成本决策的证据；不是模型 checkpoint。 | `1EBD35492D3949A2B1DF7007C20F58EFB164D931548CEE459B59AA639B2BFA69` |
| `configs/evaluation/full_set_official_protocol_v1.manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `FCEB6623AA2B05F36BD70924C9977116CB1CB4E50620A6D85649F066CC1C973B` |
| `configs/evaluation/full_set_official_v1.manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `135C8B93AC46A19B460ADB6D49DFD93D12411931E32C3516F38350F9C4474F3F` |
| `configs/evaluation/official_like_smoke_v2.jsonl` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `549C1E3B3E131E8FF586DBF17AE666FB8468B75B57362E206CFBCE2A4FCB6D0F` |
| `configs/evaluation/official_like_smoke_v2.manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `29AF6C231283BBEAE27A2F6F520ED4FCA9D0092934FA0257E67AA1932ADE7A4D` |
| `configs/evaluation/quality_formal_v1.jsonl` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `54597F02CB06A6579F07EFB5D488B5C2FAD2A2CA71F9011377F3C7F7E787DBB6` |
| `configs/evaluation/quality_sets_manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `4158771761E932DD2DC71325C736629EE1D4B7824B5582AC21D73C3B4FD108FE` |
| `configs/evaluation/quality_smoke_v1.jsonl` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `69BDDD50711A0447E9B9594850FDEB1EDEB9AB65864297656A1F6E1085F792AD` |
| `configs/experiments/eplb_gpu111_nvfp4.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `7D8F6747CD117F06544E35FCA8943B2CFA4EC53F59449461C5C03DCD6EBFD723` |
| `configs/experiments/phase1_service.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `D71A15D9B1E0BE4D14351C576D8BC2E601B073677BAB44FB1A348EC302D8B30B` |
| `configs/experiments/phase8_cache_arrival_pilot_fp8_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `BB114284C7688A72E813729894FAFC5F94830F0D8FACDD4E549F9FCC546BB88D` |
| `configs/experiments/phase8_cache_arrival_pilot_fp8_v2.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `33342AE0D4DC492A4DDB2E7FDE2F95ED33FA2FAC2956C6ED19EB8E693F7EE585` |
| `configs/experiments/phase8_cache_arrival_pilot_fp8_v3.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `BCFE2529FDD7BCB3205067EBA1640ECEC11F66E143C630C6B390CA475A889632` |
| `configs/experiments/phase8_capacity_prepass_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `4646F80FF0C84189B119728737BAB13F613872293C6C3B6F36E840F964292B68` |
| `configs/experiments/phase8_formal_controlled_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `2E42194CE47A6B1E397B7CC9AA7CB9551FA1A82025037F5CC93D16E18879A93E` |
| `configs/experiments/phase8_formal_controlled_v2.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `847C511DBE4330A722233C4793B6092577BB76FAEBAF08CCA3CF5191DB13939A` |
| `configs/experiments/phase8_formal_controlled_v3.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `C545516027C505FF43111044E1DED1BC2ED63881357F5B6CBB2053B45FA31FE3` |
| `configs/experiments/phase8_pareto_repeated_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `2FC2593317A54B4F49E35FC0C6E31BFA7280A18991734BB2075126B6936EAFE6` |
| `configs/experiments/topology_gpu111.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `46CAB1DF44FD63101C629066021EC213FFB90B9DD19D556152F97EDA241CC0D7` |
| `configs/kernels/phase4_benchmark_plan.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `5153F56411505F39A4BA3CE2CF20D69FA663F55D7C17FEDB1AAF89B7C4DF9126` |
| `configs/kernels/phase4_kernel_db.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `6AE396BFF2EC91156A8C2B9BBF60B7EE14AF5D198F42B3C593680E69B82302E8` |
| `configs/models/registry.yaml` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `6A908BFFD5F75C8AFE8FDF13C00F6EEC2C74DE09C7D4C6D12C19EEE7E767CDD1` |
| `configs/strategies/phase8_candidates.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `77BB9EA579554BC1A941255888F5E2DFB8327831BB54BFC6EA7AA3FC3C89EFA4` |
| `configs/strategies/phase8_candidates_all_formats_screened.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `CEC38841CD27DFBB87D184C8FFB45F32FBC35B21DDF479BD5B03FF174179C703` |
| `configs/strategies/phase8_candidates_nvfp4_redhat.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `DC4039833E1ED5ED0E0C4685E04A110FDA8A015EB43EBDC0B3D17570F916CC24` |
| `configs/strategies/phase8_candidates_repeated_pareto.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `CF2FB1E23025F8AA2456A5086BD365C768A20B8CA66D04C8B0BC6F06A772161A` |
| `configs/strategies/phase8_observation_nvfp4_redhat.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `81DFFE54DA2A93911D245FAC4715C124F43B72A632D0BB9548E6E3FA466B6C2A` |
| `configs/strategies/phase8_observations.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `5696B7C1CF02FD8D9610C286B2C5E0538AD63D6E32BA452FBF9EEB7FC4B76D99` |
| `configs/strategies/phase8_observations_all_formats_repeated.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `5F176527274306BC9D52692FE4599279E9851E026BF669F3A5B30CD9B9957157` |
| `configs/strategies/phase8_observations_nvfp4_redhat_screen.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0F6D4A2DE33D76D8D133BB38ECB10169E63990281D66389C014F6DDB4BE87B1D` |
| `configs/workloads/m_buckets.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `908825B5A20DF36BE72597E63489C6D84F18B982977EAFD4623076C03DCD27AA` |
| `configs/workloads/phase8_c8_supplement.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `159F7FA64E01000D16FEEEDAF86671C96FEC02163409DF48D8891FB679A36E19` |
| `configs/workloads/phase8_cache_arrival_pilot_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `5B8A08A265B8EE0288AF6AC8C6E67571E882F5AD163D28D9FC93DE31E0DAC1D8` |
| `configs/workloads/phase8_cache_arrival_pilot_v2.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `B22FB545996A0C0B400F84085D668BE3CF5412969F7FEAE27FB215FF3A8B4BC3` |
| `configs/workloads/phase8_cache_arrival_pilot_v3.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `4BDE01A7949829B604A58295A51DEA29E8E3EE6404D4111D829FF1D4B61C0739` |
| `configs/workloads/phase8_capacity_prepass_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `85E9A513450CD729A754587EED1C218CFD6A3EA6F536424EAB006DAA6C7A0B2B` |
| `configs/workloads/phase8_formal_controlled_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `3FC04A14F481CD008EE6ED6B556D020EDF2C3CEE4BF14E732D0808878710852E` |
| `configs/workloads/phase8_single_pass_screen.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `DE91DD4A05FA610D73D2E8554EA4FABB7D4959F0423DEFC057C1190089A33E74` |
| `configs/workloads/screening.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `2D99FB098BE0BA56B0281AAFA530DBC9CD434D70C70F2DF8D21B854D71CF9817` |
| `configs/workloads/smoke.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `E7901288D9FE4C44A321EF4F8AA568C64B50AC5F48B24E4001EE2452D62E93A5` |
| `data/manifests/sharegpt_seed.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `7979B05C9C1D32C4B1404F2A1BDFB8FD08D472016F32C8998227A1C020AD20E6` |
| `docs/Q-TopoMoE_Phase1_BF16_matrix_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `84CE1A34711AA68F03BB4FC6B5EE2BB6632FBE01345FCC5CA49459AE2765EEE7` |
| `docs/Q-TopoMoE_Phase1_FP8_matrix_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `167AA6DE7E341B5D48137D3C1CCF59234AE60EFC90DD2F0BBED68863BFCA7E7E` |
| `docs/Q-TopoMoE_Phase1_statistics_20260804.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `115E0BCFB9CB5953AB684C87F94174AB6C1437C3A1D8183298FE89A3FC142C64` |
| `docs/Q-TopoMoE_Phase2_NVFP4_audit_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `E5C9FE5F4FD2974B742D3F1261448F13DA758486B0D87CD6FB1DD3214B5A9212` |
| `docs/Q-TopoMoE_Phase2_NVFP4_official_like_v2_summary_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `931761A3137EF62EB784E6AD440A249D812C781ADCE9289D8A4B8AED69DDA3EF` |
| `docs/Q-TopoMoE_Phase2_NVFP4_selfgen_gate_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `88C475068AE379AE507CA5807B90CF358112659FD84C06FDD5104599E28EC95F` |
| `docs/Q-TopoMoE_Phase2_NVFP4_selfgen_official_like_v2_summary_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0512E3D1870374C6FE1816060F80BB98EDF2B3EFECECDF4D2BCC6AC47B445A2E` |
| `docs/Q-TopoMoE_Phase2_W4A16_audit_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `E13AD889ECF0279A29D04AA822CBD142CA4FB68959400D55EC450305A9EFF72E` |
| `docs/Q-TopoMoE_Phase2_W4A16_load_gate_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `13AAFF15111D7A8FFC6E6FFFEEF9A0A25526C4F59A37DBAE1D66B380D874FA9B` |
| `docs/Q-TopoMoE_Phase2_WikiText_calibration_manifest.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `C4CEBABC41BB8D72AEF99B9044FBF3D68ADB5DB711B4BD07F2A253C4C131C36D` |
| `docs/Q-TopoMoE_Phase2_official_like_v2_BF16_summary_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `5AE7EEF6555D2134C1A50BF6B27492967F3674398A59B64FDD4062C646912303` |
| `docs/Q-TopoMoE_Phase2_official_like_v2_BF16_vs_W4_compare_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `D60068D513706CCE27C95AEF6947B62D5B6C25E19E0FAE2AC672B54EE82E064D` |
| `docs/Q-TopoMoE_Phase2_official_like_v2_W4A16_g64_summary_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `A9C48DC7E5EF6DBD376BE5EE14A0BC1B9BC5C3097BF26231114219D61F8BBA89` |
| `docs/Q-TopoMoE_Phase2_official_like_v2_W4A16_summary_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `EAC8EFE05A796A34260F1D6CED2CCF736DD3C5FBDDC2215F2E35E1ABC456CBC9` |
| `docs/Q-TopoMoE_Phase3_route_drift_BF16_vs_NVFP4_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `63D3646C146C5E1AD689D3B9CAE1DA3FAFCAA76E731A2DF1DCF8A4E26009606C` |
| `docs/Q-TopoMoE_Phase3_NVFP4_fullset_merge_20260813.json` | 证据 / 结果 | NVFP4 24,374 条基础轮与 971 条续跑的合并审计、哈希、18 条显式未完成 ID 和 Gate。 | `0FE3433CD615D9D305C7594A8C0E649A3E5CEC3103588A6CA0BC26F0176625F0` |
| `docs/Q-TopoMoE_Phase3_FP8_fullset_base_20260813.json` | 证据 / 结果 | FP8 24,374 条基础轮审计、1,019 条截断 manifest 哈希及有限续跑启动 Gate。 | `4E3CA57C9E476B471E556896C839CD2FE61646C62C1A3E1FBCAD03C0C2AB97E4` |
| `docs/Q-TopoMoE_Phase3_FP8_merge_NVFP4_compare_20260813.json` | 证据 / 结果 | FP8 严格合并、26 条显式未完成审计，以及与 NVFP4 的 24,339 条共同分母对比。 | `898CD9A3A166F69F2088CF7BDCC4DCC4AE30CA64A4F03AC09176825313D0754B` |
| `docs/Q-TopoMoE_Phase3_BF16_FP8_NVFP4_质量总结_20260816.json` | 证据 / 结果 | BF16/FP8/NVFP4 严格合并、显式未完成处理和 24,330 条三格式共同分母最终质量结论。 | `ED60A3DB05C06F6CAAF54D57E338446C8172380F89205A8C273C33E009FFC383` |
| `docs/Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `9B8F761299223C99F87A7EE3E22E45E8B78E3C77094918FCA2AB29CC99B140D7` |
| `docs/Q-TopoMoE_Phase3c_full_official_w4_merged_summary_20260806.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `B1A1C06CFF29BBA2E6797B46A9FC0C507E435F891C22B6EEC8C67191FE7DA0C3` |
| `docs/Q-TopoMoE_Phase3c_pilot_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `2B216D209ADDD104E78832BD87A22346C2E4D05107C87D6858BE7D6B0CD85CF3` |
| `docs/Q-TopoMoE_Phase3c_pilot_thinking_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3B296D600C557BFE043082837A534564A7F02BE03D047CCF92F18B84F21C8B60` |
| `docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_phase8_missing_m_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3F3F357AB011CA720ED60AE8214F57AF86B0CB8999E08652126F611A6DF616F7` |
| `docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_real_m_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `FE206EA6ADDBD9644277414B56A37EBAE9A6E7998BD5EE84A1DA860C8D18973C` |
| `docs/Q-TopoMoE_Phase4_cutlass_moe_bf16_kernel_20260806_v3.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `690BBECB43E1E6EA4020003B9D362D3E0E441AEF9BBEB9107F83CAD5ADC22255` |
| `docs/Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806_v3.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `343100EEB69C1CDD152093168EA7502A1CF602FE24B1E61A3C2D928C985B7C40` |
| `docs/Q-TopoMoE_Phase4_triton_moe_fp8_kernel_20260806_v3.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `100D4AC3C5F3EE32310D0AF5F8E2E6C07CBAFA6628DB5C71F82967EBAC858530` |
| `docs/Q-TopoMoE_Phase4_triton_moe_fp8_missing_m_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3F5A067D28358762B4C72F9E4043ACE1AE219282DAB115F0349582CE1A4AECC1` |
| `docs/Q-TopoMoE_Phase4_triton_moe_w4a16_required_m_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `8866C5F79F1C4720D0BDC1A2F27AF739B86EFB321BA5DF9085A53A53B7C056B0` |
| `docs/Q-TopoMoE_Phase5_permute_quant_fusion_20260806.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `94C0DEDAAE80F01FBE41B6C8206FF958662B4B9A44E41F7F24F93EBF202BB328` |
| `docs/Q-TopoMoE_Phase5_prepare_ab_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `917031AE178A0158BD05DEFE5C89C327204476CE7A5F7D3626285666FD661877` |
| `docs/Q-TopoMoE_Phase6_NVFP4_EP_admission_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `B6E209671ABF6CB83B0C102171B293F996EBA3CD7D7002F43FC8211A517EBF90` |
| `docs/Q-TopoMoE_Phase6_matrix_4gpu_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3CABBCC155344C40EB971D0325C4CDA4A913D8F07032BE4B34F6F0A43BDCEAB5` |
| `docs/Q-TopoMoE_Phase6_matrix_8gpu_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0CAD04B7F24236851AFE4FA0B7913A4BD02439B7112A8991B1331D9CC8561735` |
| `docs/Q-TopoMoE_Phase7_NVFP4_eplb_plan_summary_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `624655A1581A0C7DADED331D5E7FEA3D56733FF1E6C08FCBEB013F1626970534` |
| `docs/Q-TopoMoE_Phase7_NVFP4_expert_migration_microbench_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `180B50EBBCBA0715B7001E0609CE1BFC82D3F466F5E65669B88E369973358770` |
| `docs/Q-TopoMoE_Phase7_NVFP4_expert_size_audit_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `17AD751A9F6E24B33DB09D497C8E40264AAFBE42BF555EBE5EC149DD59EF574A` |
| `docs/Q-TopoMoE_Phase7_migration_cost_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0E7AEA1921A18E11D0A46DD125B49F9DBB82EE18C81923341E3280532D4940E1` |
| `docs/Q-TopoMoE_Phase7_placement_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `98BC4A80BAEF5AB105F52AF71E2439B3CCB720BC74A81971D6C161F45D4A3E52` |
| `docs/Q-TopoMoE_Phase8_NVFP4_selector_replay_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `41995C5FAF18C34C62C6044C311E6256BA1A73EA79C3F3CB7CBA4C60AAAE066B` |
| `docs/Q-TopoMoE_Phase8_NVFP4_selector_screen_replay_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `657E0406A1F5A5F2122832636D6132D7C350C0864E25FAAF595E2FFB34788719` |
| `docs/Q-TopoMoE_Phase8_NVFP4_single_pass_screen_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `98A17BDC9272480E2E330491FB4B2CC362B213CF303ABB0C388378198DDFD3CA` |
| `docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `A6331F3C41B6755DCD1F065D7B73A60A6E22C07A88E5B242BB844D648453CD49` |
| `docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `2C106CCD3CF9870AAB92E8E5D5EFAF1CFE13AE7B232E1C79B6DD3DFAA0F9E722` |
| `docs/Q-TopoMoE_Phase8_cache_arrival_pilot_v3_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `F38683A65DC0A94886E8C8362FE6D65FBFF919F6193EE396182C4190DF5804A9` |
| `docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `73F4A63F8DA8EEEA750B9D0867B1D569A105D61E687E0E9B12255914E8379F79` |
| `docs/Q-TopoMoE_Phase8_capacity_audit_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `C5A1E98ACDF74B710A388D460ACC8E00817832F8A970E0815A3461257FD35005` |
| `docs/Q-TopoMoE_Phase8_capacity_prepass_launch_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `98BEE285FB4794FE7EAF6B50B40C83633875D62DA62CE516503F0A842DEE4214` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_launch_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `C9119AE46C6E41929FB47B216BCFE3C3F7486971493EBD841792B3D27B940BD4` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_v1_rejected_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `F626398F00BD3DE87A3C49BBAF65FFC9BAD4A87F1551F948EBC3E511F95F7BD0` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_v2_launch_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `1935E139B9B17C79B8C28DF591919F9D91EC3E54EA4BFED0D3396F44B198A177` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_v2_rejected_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `CCA18F1D0929B9ED623766E66E8B34F72BDFF6085FA7693DB967B3E32E3F87DC` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_v3_launch_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `1EA55715855038210E93856DCB443E2E048A6C0D0CF3F325A31EC55E99BCFC87` |
| `docs/Q-TopoMoE_Phase8_replay_20260807_p2p.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `2BAF8CB605DCDBDC2824F0C522C56304DF7205ECE2D028126D7FC26D55497E1C` |
| `docs/Q-TopoMoE_Phase8_service_calibration_cv_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `6ABA352C2BC7459DA5DF5369A8178F5AE522CA45A96670DE7D68D79E0EA0884D` |
| `docs/Q-TopoMoE_Phase8_uncalibrated_replay_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `902EE23694A525AAFAABEFB11D3F872ABD4598D65E4A52D64FFBD5DCBCCB32CF` |
| `docs/Q-TopoMoE_Qwen35_canonical_checkpoint_20260805.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `CAC08D7E7B91B9A263AF622184ABA18D12FC2AFE1A3D6DDB48A3C5BFA893F8E3` |
| `docs/Q-TopoMoE_Qwen35_canonical_tp1_gate_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `843ED722694985A4B8267F1DD97C4DB9A9D957545C1C0F05370025A1F85039EC` |
| `docs/Q-TopoMoE_Qwen35_canonical_tp2_smoke_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `85E1F7C55BDD40A6589BD9E2A636C018B5D91C977F6CFA25D4EC872DF6A2A235` |
| `docs/Q-TopoMoE_Qwen35_cleanroom_pins_20260805.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `B3A1E330611684D147076EADA122E243A1F2FFEAE09655DFD56F11319FD61531` |
| `docs/Q-TopoMoE_Qwen35_original_upstream_gate_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `B219E020F8D76083E9FDE75193203DC9D4B6B9E5E49DE033A19AED43BEE12A16` |
| `docs/Q-TopoMoE_W4A16_freeze_20260805.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `0F1F9B391E0AD5A49819125FE81B15446FD83B2E1900E07EDD89D2A8EA8EBBFD` |
| `docs/Q-TopoMoE_W4A16_reblock_manifest_20260807.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `01E4883942A772BB58AD29A717F4B9EFEF44C4714DFE4849A59A561D0EC8D51E` |
| `docs/Q-TopoMoE_quality_smoke_bf16_vllm_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `E2037D55CFBBBBB94651FAF9947AA4BF9E834DBAEF9F9804083193F423A4C334` |
| `docs/Q-TopoMoE_quality_smoke_compare_marlin_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `7B3EBC7BC95FC0B7909D1A6333F6A02B8350208E88DEC937AE662DBD92674D11` |
| `docs/Q-TopoMoE_quality_smoke_compare_triton_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `23032E02503850393383B12C565635FEA421671C43C023B3D3B27D1AE827A643` |
| `docs/Q-TopoMoE_quality_smoke_w4_sglang_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `EE4C52A650FBE267DABD21D23068432670FB6D376A5518D86F04FFE05BE07656` |
| `docs/Q-TopoMoE_quality_smoke_w4_vllm_marlin_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `D04424F2657029323457D7867544C085348DC867766B2E4C22BF426570C79A56` |
| `docs/Q-TopoMoE_quality_smoke_w4_vllm_triton_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0CAB91A7E2F06A3AB500853B1EA99296503E3E9D353D3091DC56109A19755DFD` |
| `docs/archive/Q-TopoMoE_Phase3c_pilot_official_protocol_bf16_tp4_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `0E9F16D5504EB79BDDF127018D99CCDD0F5E3B14891597ECF1C027E4C935BF04` |
| `docs/archive/Q-TopoMoE_Phase3c_pilot_official_protocol_summary_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `55EA61F167E18B7719764A16CEBAF38B5B8E5ADE2CEA260AEE482D61164E3384` |
| `docs/archive/Q-TopoMoE_Phase3c_pilot_thinking_bf16_tp4_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `E7AF1B6A8F28C2741C2814227159B11CD35F43B1D43E7248450B617DF7C9634C` |
| `docs/archive/Q-TopoMoE_Phase3c_pilot_thinking_summary_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `6ED076136084DA11FC50755519AA017D76976F6040B51F54231AA9BC7CE89027` |
| `docs/archive/Q-TopoMoE_Phase4_cutlass_moe_bf16_kernel_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `AD4BCF94471161E224A469AB8C13A3A7DCB4F91309321DF9212E8116AB4BCA36` |
| `docs/archive/Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `17F5EEECCF4375F36A5FBE2C347CE7D6E32713B7CF7290D31550CF0D7FCA0641` |
| `docs/archive/Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806_v2.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `02E4B1C41B25487BCA62093F57FCAE6FDB24D2243DFDEC45E19C8EBB91C53A52` |
| `docs/archive/Q-TopoMoE_Phase4_triton_moe_fp8_kernel_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `2C4F9BF27FD74CA4CF8D664096D73154A170E455A432702E13939F206422A815` |
| `docs/archive/Q-TopoMoE_Phase6_matrix_4gpu_20260807.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `5D54642AF6E6361DFDEE28195323B159DA24177E02567906E8BBD6C4C88E3478` |
| `docs/archive/Q-TopoMoE_Phase7_migration_cost_20260807.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `395EC6648F408AC6FD2C221AFEEDE462A9F44B4AC8222D1718703857B4014A4B` |
| `docs/archive/Q-TopoMoE_Phase8_replay_20260804.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `7461CB1F97B27D78B0E66E0963421644FE6561190BE2DD3A15C325F05F972DFB` |
| `docs/archive/Q-TopoMoE_Phase8_replay_20260806.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `5F6A88A80389F624B12D709E3109C818A66D73F988FA48CFDAC3D5B511CA3202` |
| `docs/archive/Q-TopoMoE_Phase8_replay_20260806_v2.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `23DA85BDFA44343BB862BAB5A7A6023120FB3595CC35D59B8D80880EB1B23E5E` |
| `docs/archive/Q-TopoMoE_Phase8_replay_20260806_v3.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `23BE5824AF7123058BCBB9A974E246111EF007D3EB650A1BB7D111727F88450D` |
| `docs/archive/Q-TopoMoE_Phase8_replay_20260806_v4.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `3B54FCB112DA7B0302C5B49C24CDB929F8E5DC0BA85AB3B6ED1D2EA38BD1B9F1` |
| `docs/archive/Q-TopoMoE_Phase8_replay_20260806_v5.json` | 归档证据 | 仅用于历史对照；没有新运行结果时不得提升为 accepted Gate。 | `2B7D5E9AB42AE3E8A926E9CD12908C6ECD4B08C8685F9EB9658010D48A2DF8D0` |

## 合并映射

| 新的人类可读文件 | 合并的源报告 |
|---|---|
| `docs/results/phase1_service_baseline.md` | Phase 1 BF16/FP8 服务报告与统计叙述 |
| `docs/results/phase2_quantization_quality.md` | Phase 2 量化、W4A16/NVFP4、校准、backend 与质量 Gate 叙述 |
| `docs/results/phase3_route_and_official_eval.md` | Phase 3 route trace、漂移、full-set 协议与 NVFP4/EPLB 叙述 |
| `docs/results/phase4_to_phase7_engineering.md` | Phase 4–7 kernel、融合、通信与 EPLB 进展叙述 |
| `docs/results/phase8_benchmark_history.md` | Phase 8 校准、回放、容量与故障叙述 |

旧拆分稿已从当前目录删除，当前入口只保留上表五份合并报告。需要追溯合并前文本时使用
Git 历史；日常阅读和复现不要再引用旧文件名。
# 2026-08-12 新增机器可读产物

| 路径 | 数据说明 | SHA-256 |
|---|---|---|
| `configs/strategies/phase8_observations_formal_controlled_v1.json` | 108-cell 正式 selector 观测；包含 M 桶、路由、prefix-cache、到达模式、冻结请求率与四候选五重复中位数/区间 | `C2E3E0F891A0047D39A737F20318E37963C790936328E1A575F262267AC20C24` |
| `docs/Q-TopoMoE_Phase7_migration_microbench_20260812.json` | 2026-08-12 专家迁移微基准；`all_verified=true` | `1267D44D657E85685B8B3CB6837CB4B74E2379186C51EC78CBB606E23C5343BA` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_combined_20260812.json` | 四候选、108-cell、五重复、10,000 次 bootstrap 正式聚合 | `8A61710AD796CEC3309A664DA2E53B6A58B93594E498D195A5371E3B00AA428D` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_merge_manifest_20260812.json` | GPU0–3 与 EP8 分拆运行的来源 schedule、SHA-256 与无损合并方式 | `A350F225375528B851A5682FA80FBD069A646E3394A3F4B04319113BCF32737F` |
| `docs/Q-TopoMoE_Phase8_formal_selector_nearest_gate_20260812.json` | 留 W1/W2/W3/W4 族交叉验证；median regret 通过、p95 regret 失败 | `44B266B46E6C4623A9D12D01D4B378588DCA77FEAA791F17CF5EEA3FD03978E0` |

# 2026-08-16 新增机器可读产物

| 路径 | 数据说明 | SHA-256 |
|---|---|---|
| `configs/experiments/phase8_selector_independent_v1.json` | Phase 8 selector 独立 Gate 的冻结实验计划；训练 Gate 通过前不得启动。 | `6CC4EECDB3AF60EAF57BEA1D468B8CD23693A54D01DEA4A12E3381FED5D11E13` |
| `configs/strategies/nvfp4_runtime_eplb_plan_gpu111.json` | gpu-111 NVFP4 在线 EPLB 的 placement-plan 与迁移控制参数。 | `384AE812C218CDC1934962D9E677E7094B2920930E3C31F2FAC18CBA8123A4CE` |
| `configs/strategies/phase8_selector_state_v2.template.json` | selector v2 训练模板；状态为草稿，只有训练 Gate 通过后才能产生冻结配置。 | `F23F2AB12877A065B8300B0BCAC479DF0FCA6EFAA5EB5036CE1355B6F96B096C` |
| `configs/strategies/phase8_selector_state_v3.template.json` | selector v3 有序区间规则模板；只允许训练侧拟合，独立结果读取前冻结。 | `F65B8A9411FEF0C53B389ACEF774077187C9B62E2F1D146A7D27DEFF1F4C3D5E` |
| `configs/workloads/phase8_selector_independent_design_v1.json` | 与训练参数指纹不重叠的独立 workload 设计规则。 | `66058F64D0EC270FC55E7F6402EA88C6FA40A152CBDA0AA5A3153629DDF29008` |
| `configs/workloads/phase8_selector_independent_v1.json` | 按冻结设计规则生成的45-cell独立 workload；2026-08-23 已完成四候选、五重复的900次正式测量。 | `B87110A400AB3B28FEF478ED635B791E3B12DFBAAD41159804E8E2B156DD4728` |
| `docs/Q-TopoMoE_Phase8_selector_v2_predecision_audit_20260816.json` | 108 个决策前窗口的状态阶段、遥测覆盖和可用性审计；Gate 已接受。 | `FDC83EC5DE1F41EB61A528B4BCC229619116BEF93CCCBD1CEA6ED1E5F7DA85DE` |
| `docs/Q-TopoMoE_Phase8_selector_v2_training_gate_failed_20260816.json` | 正式扩展搜索所得最佳配置及失败状态；不得作为生产 selector 加载。 | `BB736986E7307AA861B95F323BB780ABDD83BF6F86DD7E48554EA641CCDF92D2` |
| `docs/Q-TopoMoE_Phase8_selector_v2_fit_report_20260816.json` | 2,500 组训练侧组合、留族交叉验证、最差样本和 Gate 判定的完整报告。 | `69B72779A406CE53ADC9E112BC6A6AA8105BDFD9AE0159DBF5AA0358FF6AF68B` |
| `docs/Q-TopoMoE_Phase8_selector_v3_frozen_20260816.json` | v3 训练 Gate 通过后的冻结 selector；明确记录独立测试结果未读取。 | `77836F21A37132C9FCD9EB8CB348B1890AE321905AD42D81A90CB696712BEFB4` |
| `docs/Q-TopoMoE_Phase8_selector_v3_fit_report_20260816.json` | v3 训练侧指标、留族审计、最差样本和 Gate 判定的完整报告。 | `7D60B1178B4F9D542FE8E29BBD7D036BC58F5E1A588A35CCEC3ECBEA59237F8F` |

# 2026-08-23 新增机器可读产物

| 路径 | 数据说明 | SHA-256 |
|---|---|---|
| `docs/Q-TopoMoE_Phase8_selector_independent_gate_failed_20260823.json` | 冻结 selector 在45-cell、四候选、五重复独立集上的完整 Gate 与逐 workload 决策；p95 regret 11.50%，Gate 失败。 | `6F1D79571CB3366FDC581C6B518A7D25ED1AEAAD89D4FBFD05E08EF2FFDA95E4` |
| `docs/Q-TopoMoE_Phase8_selector_independent_manifest_20260823.json` | 900次正式测量的完整性、输入/输出哈希、Gate 摘要和 SGLang 恢复状态。 | `37F2BAE8810E68E0160CC4DD24D633A1401C75489BE3624C7B839767D4243DCA` |
| `configs/strategies/phase8_selector_gate_policy_v2_20260823.json` | 独立结果揭晓后批准的12%后续准入政策；保留原始10% Gate，并记录本次指标在新政策下的机器可读判定与有限 canary 边界。 | `89F03F1B23747AD515C9A9B0B9FDF71FEFAF63967B0181D059EBDFF7B2C19D36` |
| `docs/results/phase8_selector_posthoc_12pct_policy_20260823.md` | 12%后续准入政策的中文说明；区分原始 Gate 与当前运行决策，列明有限 canary 的后续门槛。 | `3A6565552DEF76526AD0EB01DDA3065CA54FD2FF2EBACE8E0B7D72E090D37722` |
| `configs/experiments/phase8_selector_limited_canary_v1.json` | 有限 canary 的冻结计划；三阶段使用匹配请求流，并在稳定段结束后通过带哈希文件显式激活 placement-plan。 | `DFE31BE5B7AEBA587F6E686F83EFE025F5FA6272C659B066CCE25BA83EA2486C` |
| `docs/Q-TopoMoE_Phase8_selector_limited_canary_rejected_20260823.json` | 有效有限 canary 的机器 Gate；384 请求零失败，但恢复 p99 为稳定段的136.36%，超过105%门槛，状态为 rejected。 | `8F03B73D99E6FA9969D0CE9743C6B5E0A73C9E22E9E298F2E4794F5608027FEC` |
| `docs/Q-TopoMoE_Phase8_selector_limited_canary_manifest_20260823.json` | 有限 canary 原始目录、代码提交、服务回滚状态、逐文件字节数与 SHA-256，以及无效尝试的隔离路径。 | `17D8662C4EFF642831F20260ED38626158E2A02261FC938943718114591F7D8F` |
| `docs/results/phase8_selector_limited_canary_rejected_20260823.md` | 有限 canary 中文拒绝报告；记录全部 Gate、延迟、编排修复、证据可靠性边界和后续禁止项。 | `2657A4BB4035B5840ED1B5E2FFA0ACEFB5CFDE29EBCBB35A49D2B9F59CB957E1` |
| `configs/experiments/phase8_selector_one_shot_canary_base_v1.json` | one-shot placement 五轮配对 canary 的基础计划；定义代表负载、显式激活、固定输出和 8-rank 哈希 Gate。 | `2CEFC83513A4ED7F63B32CFF32F900B65972A29DD0EB4B90DA2CFD4F6AA03AA8` |
| `configs/experiments/phase8_selector_one_shot_repeated_v1.json` | one-shot placement 五轮调度与聚合 Gate；结果曾 accepted，但固定阶段顺序的预热偏差已在最终报告中注明。 | `AE0521F0479EA0229BD612B9998DAB5EBD96625B02DCBDF7014E8D38A6825A80` |
| `configs/experiments/phase8_selector_closed_loop_acceptance_v1.json` | 自动闭环首次预注册计划；历史诊断发现误用了跨层汇总 rank CV，不作为最终验收口径。 | `DDD44081588ABC5163CF340D446B682B30C81D593420C6B3552571BC38B431C9` |
| `configs/experiments/phase8_selector_closed_loop_acceptance_v2.json` | 自动闭环第二次计划；修正逐层 expert-load CV，诊断出单窗口 p99 回滚参照过敏。 | `42FE638999EEDAE82A511371193C866F602B8AB1B5D11FC542BBDD45FACC9824` |
| `configs/experiments/phase8_selector_closed_loop_acceptance_v3.json` | 自动闭环第三次计划；冻结迁移前 p99 EMA，短负载在 cooldown 第9窗真实回滚。 | `8288C36A62CB3785CA84B986C88EA4563076E0AB068EC5AE9134B124D252CEB4` |
| `configs/experiments/phase8_selector_closed_loop_acceptance_v4.json` | 最终自动闭环计划；负载严格对齐五轮代表负载，原 Gate 不变，最终状态 rejected。 | `7C131A3F9554056932BF129598E93E3D521EC64AC363F69B17D1EB3BD5C0BE2E` |
| `docs/Q-TopoMoE_Phase8_selector_closed_loop_execution_summary_20260823.json` | 数据说明：合并记录 v1–v4 的目的、结果、服务器目录、产物哈希、代码提交和最终部署边界。 | `F3D8D196A4DA30EC6738D3639AA55FF7F34D8849FAE39182AB88BF8012B75970` |
| `docs/Q-TopoMoE_Phase8_selector_closed_loop_final_gate_rejected_20260823.json` | 最终机器 Gate；逐窗口记录 expert/rank CV、p99、控制器动作和 SGLang 恢复状态，状态 rejected。 | `09A29EC83FB63630091C3BD6585EEB7184F4E185251C3E28159A39B873AB1D15` |
| `docs/results/phase8_selector_closed_loop_final_20260823.md` | 自动闭环中文最终报告；说明五轮顺序偏差、四次诊断、最终拒绝理由与下一候选要求。 | `39B13289D9C767DE9F41CAA8C3FBE6D44FE7F75338DB82B243E7B4BCAA36E85C` |
| `configs/experiments/phase8_warm_placement_capture_v1.json` | 冻结旧 Gate 后的 identity 暖态逐层专家计数采集计划；正式有效结果来自独立 v3 输出目录。 | `D5F55D1A5200FD877E4F2666272FC87ED6C0F93C841C3E4ED86E6DD65FC4DB27` |
| `configs/experiments/phase8_warm_placement_abab_v1.json` | 新候选同进程 A/B/A/B 冻结计划；迁移提交段先预热且不计入测量。 | `6B59A067002D6280E37607A4E451BE0171212464B6CC92AFE6CEE2128A85C18E` |
| `configs/strategies/phase8_warm_swap_008_slots_per_layer_v1.json` | 从48个暖态窗口生成的最小迁移量候选；文件内状态是生成时快照，当前准入必须同时核对最终 accepted 摘要、map 哈希与运行时补丁，不能孤立加载。 | `3386D1457A62F4C2A9CB5CDC56B71225C8F07FAE73063A0E51C142F4C0B1F29E` |
| `configs/experiments/phase8_warm_placement_quality_equivalence_v1.json` | 首次 116 题质量 A/B/A 冻结计划；候选轮出现 8 条截断后按 Gate 停止，用于保留缺陷证据。 | `8A0BDEA1D2DE348E1EC9FFF8AA2E4257EB3B6302B7A9A82422594109D58AA237` |
| `configs/experiments/phase8_warm_placement_nvfp4_aux_regression_v2.json` | NVFP4 Marlin 两个辅助尺度迁移修复后的 8 条定向 A/B/A 回归计划，结果 accepted。 | `CC6B752C4FBD804AA9A77CACEB7820278AA66673B4B88A03BFEA9E261D6FF637` |
| `configs/experiments/phase8_warm_placement_quality_equivalence_v3.json` | 修复后完整 116 题质量 A/B/A 冻结计划；三轮零失败零截断，最终质量 Gate accepted。 | `4B3B56628E474195150A271D70BAB0901B56C42BFA8489624C36C28EDAE52C57` |
| `configs/experiments/phase8_route_stability_diagnostic_v1.json` | 14.28% 逻辑专家分布变化的一级预注册诊断；并行记录旧口径、记录时逻辑计数和 map 槽位代际，分离短输出与长生成轨迹。 | `5BCBD62A14378E4F2DD8269A8369B0174631EF981D655AA1347693418758042C` |
| `configs/experiments/phase8_route_stability_diagnostic_v2.json` | v1 后冻结的热态确认差异计划；继承 v1 全部阈值，在 identity A1 前增加热缓存 settle，并固定成因分类边界。 | `51F8E3B3E21A49EE12049A298A2789B3FECC267BD9DBACC721663730678305B4` |
| `configs/strategies/phase8_route_stability_gate_policy_v2_20260825.json` | 当前有效路由稳定性政策；上限由历史 0.5% 调整为 1%，现有短臂 0.045% 和长臂 0.718% 均通过，仅结束本诊断阶段。 | `DE15EA79DA27B5DF968478B4EF7116C976DBE5905BFE10D4ECCB22B16859DBE7` |
| `configs/experiments/phase8_warm_placement_limited_canary_v1.json` | 新暖态 placement 候选的有限 canary 冻结计划；使用显式激活文件，要求零失败、计划哈希一致和恢复 p99≤基线 105%，既有 SGLang 按用户要求不纳入本轮管理或 Gate。 | `0A2B085E5738096055132B81798799114F42870CE63E4A81284C90B6D67A9EE3` |
| `configs/experiments/phase8_warm_placement_closed_loop_acceptance_v1.json` | 有限 canary 接受后的自动闭环冻结计划；验收三窗口 trigger、十窗口 cooldown、三窗口 placement rollback、8-rank 提交、决策开销和回滚后 p99。 | `403F55BA6059FF8E92B264F4C350B5C57360F9A89A49C3E61FB61D472D0B6392` |
| `docs/Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json` | 暖态候选、迁移修复、质量 Gate、两级路由诊断与当前 1% 准入边界的合并机器摘要；项目仍在进行。 | `877A0CFBE66C73F32E698EE5C912F1CABD9BF0664E9098F44564D2634F6EB241` |
| `docs/results/phase8_warm_placement_regeneration_20260823.md` | 合并中文报告；保留历史 0.5% 拒绝证据，记录当前 1% Gate 接受，并链接到最终验收。 | `B924EC635B959D08F3967C7D6EBE2B2C47B68E773EF170EC11BB45B8C4DEE6BC` |
| `docs/Q-TopoMoE_Phase8_warm_placement_final_acceptance_20260825.json` | 当前 1% 路由 Gate、有限 canary、自动闭环及无效尝试隔离的最终机器摘要；Phase 8 技术验证状态为 accepted。 | `F04331C7365B44EC1DAB8CE1756AC9FE051A167FCC4CD5CDED19316EA8868BC6` |
| `docs/results/phase8_warm_placement_final_acceptance_20260825.md` | Phase 8 暖态 placement 最终中文验收报告；记录 384 请求 canary 和 20 窗口自动闭环结果，并明确生产部署未执行。 | `3ED65D754ACA553C8D977A39C05D878A6DAADD895419221CD563F66B58C116BD` |
| `docs/Q-TopoMoE_release_manifest_20260825.json` | 技术验收版 release manifest；绑定推荐模型路径、候选文件与 map、运行时补丁、当前政策及最终 Gate，并声明生产部署尚未执行。 | `B9949115BB1830949DE289F79ADDBA92F83FE7903D77BEB97C638723394CEE23` |
| `docs/Q-TopoMoE_项目发布与生产部署清单_20260825.md` | 最终中文交付清单；说明唯一推荐组合、证据语义、发布检查，以及生产小流量、扩量、闭环启用和回滚步骤。 | `D5CFD65D50AD5C24E28081E588F0ABE843BEACD0FDBFCD4D3A3D68F4C90AC5F2` |
