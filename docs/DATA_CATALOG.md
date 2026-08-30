# Q-TopoMoE 数据与产物清单

本清单记录 `configs/`、`docs/` 与 `data/` 下所有已跟踪的机器可读产物。集中为一个文件而不是每个数据文件各放一份 README，既减少文件数量，也不隐藏来源信息。

## 条目阅读方式

- **冻结输入 / manifest**：由复现或评测命令消费；不得原地编辑。路径与 SHA-256 共同锁定输入。
- **实验计划**：可执行或半可执行的控制数据；必须保留路径，因为脚本和 Runbook 可能直接引用。
- **证据 / 结果**：记录的测量、审计、Gate 或摘要；除非 manifest 明确指向，否则不是运行时输入。
- **归档证据**：仅保留输入和测量口径有效的历史比较；不得用作当前 Gate。
- 原始 benchmark 大文件不进入 Git；需要时由 manifest 记录预期位置和哈希。

数据文件名保留日期和版本，因为它们是来源信息而不是冗余。人类可读的阶段归档见 `docs/results/phase*.md`；输入无效的测量及其派生路径已从仓库和本清单移除。

本次提交完成后的机器可读产物数量： **146**

| 路径 | 类型 / 数据说明 | 复现指引 | SHA-256 |
|---|---|---|---|
| `configs/communication/nccl_cost_db.json` | 硬件 / 通信测量 | 2026-08-07 P2P 生效后的唯一权威通信成本库；不是模型 checkpoint。 | `B4D0796CDB76DA88307C7F2A09FC27F3C09B7D3408367DF54A0018756FDFF611` |
| `configs/evaluation/full_set_official_protocol_v1.manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `FCEB6623AA2B05F36BD70924C9977116CB1CB4E50620A6D85649F066CC1C973B` |
| `configs/evaluation/full_set_official_v1.manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `135C8B93AC46A19B460ADB6D49DFD93D12411931E32C3516F38350F9C4474F3F` |
| `configs/evaluation/official_like_smoke_v2.jsonl` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `549C1E3B3E131E8FF586DBF17AE666FB8468B75B57362E206CFBCE2A4FCB6D0F` |
| `configs/evaluation/official_like_smoke_v2.manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `29AF6C231283BBEAE27A2F6F520ED4FCA9D0092934FA0257E67AA1932ADE7A4D` |
| `configs/evaluation/quality_formal_v1.jsonl` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `54597F02CB06A6579F07EFB5D488B5C2FAD2A2CA71F9011377F3C7F7E787DBB6` |
| `configs/evaluation/quality_sets_manifest.json` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `4158771761E932DD2DC71325C736629EE1D4B7824B5582AC21D73C3B4FD108FE` |
| `configs/evaluation/quality_smoke_v1.jsonl` | 冻结输入 / manifest | 由评测、smoke 或 full-set 命令消费；保持路径和字节内容不变。 | `69BDDD50711A0447E9B9594850FDEB1EDEB9AB65864297656A1F6E1085F792AD` |
| `configs/experiments/phase1_service.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `6552B034403B127D69AF1B95FA849AB4C41833DCDA846B3C86902403806FAD6D` |
| `configs/experiments/phase8_cache_arrival_pilot_fp8_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `BB114284C7688A72E813729894FAFC5F94830F0D8FACDD4E549F9FCC546BB88D` |
| `configs/experiments/phase8_cache_arrival_pilot_fp8_v2.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `33342AE0D4DC492A4DDB2E7FDE2F95ED33FA2FAC2956C6ED19EB8E693F7EE585` |
| `configs/experiments/phase8_cache_arrival_pilot_fp8_v3.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `BCFE2529FDD7BCB3205067EBA1640ECEC11F66E143C630C6B390CA475A889632` |
| `configs/experiments/phase8_capacity_prepass_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `4646F80FF0C84189B119728737BAB13F613872293C6C3B6F36E840F964292B68` |
| `configs/experiments/phase8_formal_controlled_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `2E42194CE47A6B1E397B7CC9AA7CB9551FA1A82025037F5CC93D16E18879A93E` |
| `configs/experiments/phase8_formal_controlled_v2.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `978DC435E6B91EF0B26AEB1E69DF5C394E340657F74B71A39936B7101D3AFAB1` |
| `configs/experiments/phase8_formal_controlled_v3.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `6DF2EC472A0A54C4DF15F780BAFF11F3718951E15B5D35B1EEA6D3BBD77651F8` |
| `configs/experiments/phase8_pareto_repeated_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `2FC2593317A54B4F49E35FC0C6E31BFA7280A18991734BB2075126B6936EAFE6` |
| `configs/experiments/topology_gpu111.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `A279331908FBBE8FAD62C3777A2CAA31E3506D7D5CA1A3154DD79CEF96EE810C` |
| `configs/kernels/phase4_benchmark_plan.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `AEFEFDF358A1D0DFCB117FF8A79A7FB9A0D6B787E8F94A3B5A408BF90F350362` |
| `configs/kernels/phase4_kernel_db.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `FDF5423EC7315CBE31BEF0656EBBE147A5508DD0308F9CBE4E97A975B7577375` |
| `configs/models/registry.yaml` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `C25B5B6DEE8BE1960844B5484A820940BDA6ACB86D0DD7768482C9B3045F70E5` |
| `configs/strategies/phase8_candidates.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `77BB9EA579554BC1A941255888F5E2DFB8327831BB54BFC6EA7AA3FC3C89EFA4` |
| `configs/strategies/phase8_candidates_all_formats_screened.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `CEC38841CD27DFBB87D184C8FFB45F32FBC35B21DDF479BD5B03FF174179C703` |
| `configs/strategies/phase8_candidates_nvfp4_redhat.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `DC4039833E1ED5ED0E0C4685E04A110FDA8A015EB43EBDC0B3D17570F916CC24` |
| `configs/strategies/phase8_candidates_repeated_pareto.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `CF2FB1E23025F8AA2456A5086BD365C768A20B8CA66D04C8B0BC6F06A772161A` |
| `configs/strategies/phase8_observation_nvfp4_redhat.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `81DFFE54DA2A93911D245FAC4715C124F43B72A632D0BB9548E6E3FA466B6C2A` |
| `configs/strategies/phase8_observations.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `B6B419C206BEF387853DEE2D025AC146885EA33DB7F8300A0F220D9B470AF8E8` |
| `configs/strategies/phase8_observations_all_formats_repeated.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `06811187E2F695927045A3F2DD5EA1E205183BCFC996EFB04C6FFD396349E895` |
| `configs/strategies/phase8_observations_nvfp4_redhat_screen.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `8057E99E0010EFE61BA2AB4856D027DAF2534FFE9C8FB0483FFB5892B549BA7F` |
| `configs/workloads/m_buckets.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `C14D4FCB2282DE5E47A1D4207E151B79BD36419653EA45CB2DE8B382F7584B7B` |
| `configs/workloads/phase8_c8_supplement.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `159F7FA64E01000D16FEEEDAF86671C96FEC02163409DF48D8891FB679A36E19` |
| `configs/workloads/phase8_cache_arrival_pilot_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `5B8A08A265B8EE0288AF6AC8C6E67571E882F5AD163D28D9FC93DE31E0DAC1D8` |
| `configs/workloads/phase8_cache_arrival_pilot_v2.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `B22FB545996A0C0B400F84085D668BE3CF5412969F7FEAE27FB215FF3A8B4BC3` |
| `configs/workloads/phase8_cache_arrival_pilot_v3.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `4BDE01A7949829B604A58295A51DEA29E8E3EE6404D4111D829FF1D4B61C0739` |
| `configs/workloads/phase8_capacity_prepass_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `85E9A513450CD729A754587EED1C218CFD6A3EA6F536424EAB006DAA6C7A0B2B` |
| `configs/workloads/phase8_formal_controlled_v1.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `3FC04A14F481CD008EE6ED6B556D020EDF2C3CEE4BF14E732D0808878710852E` |
| `configs/workloads/phase8_single_pass_screen.json` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `DE91DD4A05FA610D73D2E8554EA4FABB7D4959F0423DEFC057C1190089A33E74` |
| `configs/workloads/screening.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `C857F87E70D6ADEA0FA02D51C72A165099A47A65854F0DF5BD7ED9871CC48058` |
| `configs/workloads/smoke.yaml` | 实验计划 | 阶段执行器或 benchmark 的控制数据；只能在引用的 Runbook 和环境 pin 下复现。 | `61591500780660CA3C73491A92134ED06A6FCD45A0B560AF0BA4689038E0CEA9` |
| `data/manifests/sharegpt_seed.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `C5DA9CDFDAD9054CA00F314571EBAC4EC50C711A676BF9C48C6BE2103BDE9389` |
| `docs/Q-TopoMoE_Phase1_BF16_matrix_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `84CE1A34711AA68F03BB4FC6B5EE2BB6632FBE01345FCC5CA49459AE2765EEE7` |
| `docs/Q-TopoMoE_Phase1_FP8_matrix_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `167AA6DE7E341B5D48137D3C1CCF59234AE60EFC90DD2F0BBED68863BFCA7E7E` |
| `docs/Q-TopoMoE_Phase2_NVFP4_audit_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `E5C9FE5F4FD2974B742D3F1261448F13DA758486B0D87CD6FB1DD3214B5A9212` |
| `docs/Q-TopoMoE_Phase2_NVFP4_official_like_v2_summary_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `931761A3137EF62EB784E6AD440A249D812C781ADCE9289D8A4B8AED69DDA3EF` |
| `docs/Q-TopoMoE_Phase2_NVFP4_selfgen_official_like_v2_summary_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0512E3D1870374C6FE1816060F80BB98EDF2B3EFECECDF4D2BCC6AC47B445A2E` |
| `docs/Q-TopoMoE_Phase2_W4A16_audit_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0FE342610861222196148ADAB6AE400DC6C908D85A9F8542ABB002F8BF1819D4` |
| `docs/Q-TopoMoE_Phase2_W4A16_load_gate_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `46F4E8B089A15A49560E8034E99BEE345F8A5F4A3B52DD60F5502BFBB62CDEAD` |
| `docs/Q-TopoMoE_Phase2_WikiText_calibration_manifest.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `01115B43EF5EE0541D7249D1666C5865466C14DB522F79DB5B7446CC3F431A34` |
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
| `docs/Q-TopoMoE_Phase3c_pilot_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `CA530010E541E2E7EBB22915F657EBF6BD7346E080FDF8B58C77C11059C4DED3` |
| `docs/Q-TopoMoE_Phase3c_pilot_thinking_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3E333E3A79EF023B58398E9AC4BAA3012A586AABE787A50C52B524FA8DDC01E0` |
| `docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_phase8_missing_m_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3F3F357AB011CA720ED60AE8214F57AF86B0CB8999E08652126F611A6DF616F7` |
| `docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_real_m_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `FE206EA6ADDBD9644277414B56A37EBAE9A6E7998BD5EE84A1DA860C8D18973C` |
| `docs/Q-TopoMoE_Phase4_cutlass_moe_bf16_kernel_20260806_v3.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `690BBECB43E1E6EA4020003B9D362D3E0E441AEF9BBEB9107F83CAD5ADC22255` |
| `docs/Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806_v3.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `343100EEB69C1CDD152093168EA7502A1CF602FE24B1E61A3C2D928C985B7C40` |
| `docs/Q-TopoMoE_Phase4_triton_moe_fp8_kernel_20260806_v3.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `100D4AC3C5F3EE32310D0AF5F8E2E6C07CBAFA6628DB5C71F82967EBAC858530` |
| `docs/Q-TopoMoE_Phase4_triton_moe_fp8_missing_m_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3F5A067D28358762B4C72F9E4043ACE1AE219282DAB115F0349582CE1A4AECC1` |
| `docs/Q-TopoMoE_Phase4_triton_moe_w4a16_required_m_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `8866C5F79F1C4720D0BDC1A2F27AF739B86EFB321BA5DF9085A53A53B7C056B0` |
| `docs/Q-TopoMoE_Phase5_permute_quant_fusion_20260806.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `4F96369CCF35CE02848D6CC76A3FC980DD70FCE08A61ED81FF6D163642AA7A0C` |
| `docs/Q-TopoMoE_Phase5_prepare_ab_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `917031AE178A0158BD05DEFE5C89C327204476CE7A5F7D3626285666FD661877` |
| `docs/Q-TopoMoE_Phase6_NVFP4_EP_admission_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `B6E209671ABF6CB83B0C102171B293F996EBA3CD7D7002F43FC8211A517EBF90` |
| `docs/Q-TopoMoE_Phase6_matrix_4gpu_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3CABBCC155344C40EB971D0325C4CDA4A913D8F07032BE4B34F6F0A43BDCEAB5` |
| `docs/Q-TopoMoE_Phase6_matrix_8gpu_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `52C3C14B6C5BA01F4AAAED8B7FD8C05F04D4639FCE17B13C4775CA20A1E02A44` |
| `docs/Q-TopoMoE_Phase7_NVFP4_expert_migration_microbench_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `180B50EBBCBA0715B7001E0609CE1BFC82D3F466F5E65669B88E369973358770` |
| `docs/Q-TopoMoE_Phase7_NVFP4_expert_size_audit_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `17AD751A9F6E24B33DB09D497C8E40264AAFBE42BF555EBE5EC149DD59EF574A` |
| `docs/Q-TopoMoE_Phase7_migration_cost_p2p_20260807.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `AC8562209679D2A0D71C77C5700CD69450BA810969F36CBFEC654701D237EFD3` |
| `docs/Q-TopoMoE_Phase8_NVFP4_selector_replay_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `3D7799A96A43F0B68D26C4A4B7B974EF825B064FFF135630878A0A1DC815BA3C` |
| `docs/Q-TopoMoE_Phase8_NVFP4_selector_screen_replay_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `E3F4144B71485FACD18902BB9CB5102FD1F28767AE8D37F4161A9BC52629A4EF` |
| `docs/Q-TopoMoE_Phase8_NVFP4_single_pass_screen_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `98A17BDC9272480E2E330491FB4B2CC362B213CF303ABB0C388378198DDFD3CA` |
| `docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `A6331F3C41B6755DCD1F065D7B73A60A6E22C07A88E5B242BB844D648453CD49` |
| `docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `2C106CCD3CF9870AAB92E8E5D5EFAF1CFE13AE7B232E1C79B6DD3DFAA0F9E722` |
| `docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `C5D9FFF5672D2C6F995E48C208468B8A61CB16A39930BF43802EB0F4C4471780` |
| `docs/Q-TopoMoE_Phase8_capacity_audit_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `C5A1E98ACDF74B710A388D460ACC8E00817832F8A970E0815A3461257FD35005` |
| `docs/Q-TopoMoE_Phase8_capacity_prepass_launch_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `98BEE285FB4794FE7EAF6B50B40C83633875D62DA62CE516503F0A842DEE4214` |
| `docs/Q-TopoMoE_Phase8_replay_20260807_p2p.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `2BAF8CB605DCDBDC2824F0C522C56304DF7205ECE2D028126D7FC26D55497E1C` |
| `docs/Q-TopoMoE_Phase8_uncalibrated_replay_20260810.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `480BA116F2313CE6A40775EAD0BA0B04A352ABB64B457E8E01538FB66074BB49` |
| `docs/Q-TopoMoE_Qwen35_canonical_checkpoint_20260805.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `CAC08D7E7B91B9A263AF622184ABA18D12FC2AFE1A3D6DDB48A3C5BFA893F8E3` |
| `docs/Q-TopoMoE_Qwen35_canonical_tp1_gate_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `843ED722694985A4B8267F1DD97C4DB9A9D957545C1C0F05370025A1F85039EC` |
| `docs/Q-TopoMoE_Qwen35_canonical_tp2_smoke_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `85E1F7C55BDD40A6589BD9E2A636C018B5D91C977F6CFA25D4EC872DF6A2A235` |
| `docs/Q-TopoMoE_Qwen35_cleanroom_pins_20260805.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `A8888F9B1B7F4428620CB4DD0ECB9E05EB723C420FDE08A2F48BD9486081A169` |
| `docs/Q-TopoMoE_W4A16_freeze_20260805.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `0B0258836394714237122029300B7BD00708C662C5126D45F37417FCC120698D` |
| `docs/Q-TopoMoE_W4A16_reblock_manifest_20260807.json` | 固定元数据 / manifest | 用于来源和完整性校验；加载引用资产前先验证 SHA-256。 | `01E4883942A772BB58AD29A717F4B9EFEF44C4714DFE4849A59A561D0EC8D51E` |
| `docs/Q-TopoMoE_quality_smoke_bf16_vllm_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `E2037D55CFBBBBB94651FAF9947AA4BF9E834DBAEF9F9804083193F423A4C334` |
| `docs/Q-TopoMoE_quality_smoke_compare_marlin_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `7B3EBC7BC95FC0B7909D1A6333F6A02B8350208E88DEC937AE662DBD92674D11` |
| `docs/Q-TopoMoE_quality_smoke_compare_triton_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `23032E02503850393383B12C565635FEA421671C43C023B3D3B27D1AE827A643` |
| `docs/Q-TopoMoE_quality_smoke_w4_sglang_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `EE4C52A650FBE267DABD21D23068432670FB6D376A5518D86F04FFE05BE07656` |
| `docs/Q-TopoMoE_quality_smoke_w4_vllm_marlin_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `D04424F2657029323457D7867544C085348DC867766B2E4C22BF426570C79A56` |
| `docs/Q-TopoMoE_quality_smoke_w4_vllm_triton_tp4_20260805.json` | 证据 / 结果 | 只读记录；解释结果时请结合对应阶段报告或交接文档。 | `0CAB91A7E2F06A3AB500853B1EA99296503E3E9D353D3091DC56109A19755DFD` |
## 人类可读聚合层

| 当前入口 | 作用 |
|---|---|
| `docs/results/phase1_service_baseline.md` | Phase 1 BF16/FP8 服务报告与统计叙述 |
| `docs/results/phase2_quantization_quality.md` | Phase 2 量化、W4A16/NVFP4、校准、backend 与质量 Gate 叙述 |
| `docs/results/phase3_route_and_official_eval.md` | Phase 3 route trace、漂移、full-set 协议与 NVFP4/EPLB 叙述 |
| `docs/results/phase3_fullset_quality_closeout_20260812.md` | BF16/FP8/NVFP4 full-set 严格合并和共同分母结论 |
| `docs/results/phase4_to_phase7_engineering.md` | Phase 4–7 kernel、融合、通信与 EPLB 进展叙述 |
| `docs/results/phase8_benchmark_history.md` | Phase 8 正式矩阵、校准和输入证据 |
| `docs/results/phase8_warm_placement_final_acceptance_20260825.md` | 当前 Phase 8 最终技术验收 |

`docs/results/` 只保留当前主线的聚合报告。输入无效或不参与当前发布的叙述性材料不纳入本清单。
# 2026-08-12 新增机器可读产物

| 路径 | 数据说明 | SHA-256 |
|---|---|---|
| `configs/strategies/phase8_observations_formal_controlled_v1.json` | 108-cell 正式 selector 观测；包含 M 桶、路由、prefix-cache、到达模式、冻结请求率与四候选五重复中位数/区间 | `4377F698C9941992312B32D0C5EA7D8CDDFA6F323DB70B3706A2F4B5061FBBBD` |
| `docs/Q-TopoMoE_Phase7_migration_microbench_20260812.json` | 2026-08-12 专家迁移微基准；`all_verified=true` | `1267D44D657E85685B8B3CB6837CB4B74E2379186C51EC78CBB606E23C5343BA` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_combined_20260812.json` | 四候选、108-cell、五重复、10,000 次 bootstrap 正式聚合 | `8A61710AD796CEC3309A664DA2E53B6A58B93594E498D195A5371E3B00AA428D` |
| `docs/Q-TopoMoE_Phase8_formal_controlled_merge_manifest_20260812.json` | GPU0–3 与 EP8 分拆运行的来源 schedule、SHA-256 与无损合并方式 | `A350F225375528B851A5682FA80FBD069A646E3394A3F4B04319113BCF32737F` |

# 2026-08-16 新增机器可读产物

| 路径 | 数据说明 | SHA-256 |
|---|---|---|
| `configs/experiments/phase8_selector_independent_v1.json` | Phase 8 selector 独立 Gate 的冻结实验计划；训练 Gate 通过前不得启动。 | `6CC4EECDB3AF60EAF57BEA1D468B8CD23693A54D01DEA4A12E3381FED5D11E13` |
| `configs/strategies/phase8_selector_state_v2.template.json` | selector v2 训练模板；状态为草稿，只有训练 Gate 通过后才能产生冻结配置。 | `F23F2AB12877A065B8300B0BCAC479DF0FCA6EFAA5EB5036CE1355B6F96B096C` |
| `configs/strategies/phase8_selector_state_v3.template.json` | selector v3 有序区间规则模板；只允许训练侧拟合，独立结果读取前冻结。 | `222D0F0D46EE310126CD2E18FC88BC8B099B21348E4D79BEB281BEA5541E4D94` |
| `configs/workloads/phase8_selector_independent_design_v1.json` | 与训练参数指纹不重叠的独立 workload 设计规则。 | `66058F64D0EC270FC55E7F6402EA88C6FA40A152CBDA0AA5A3153629DDF29008` |
| `configs/workloads/phase8_selector_independent_v1.json` | 按冻结设计规则生成的45-cell独立 workload；2026-08-23 已完成四候选、五重复的900次正式测量。 | `E281D2B4837EA2D1B1D1ED00211844FE12F446E8CE8E52A904C140FDA3649BD6` |
| `docs/Q-TopoMoE_Phase8_selector_v2_predecision_audit_20260816.json` | 108 个决策前窗口的状态阶段、遥测覆盖和可用性审计；Gate 已接受。 | `FDC83EC5DE1F41EB61A528B4BCC229619116BEF93CCCBD1CEA6ED1E5F7DA85DE` |
| `docs/Q-TopoMoE_Phase8_selector_v3_frozen_20260816.json` | v3 训练 Gate 通过后的冻结 selector；明确记录独立测试结果未读取。 | `77836F21A37132C9FCD9EB8CB348B1890AE321905AD42D81A90CB696712BEFB4` |
| `docs/Q-TopoMoE_Phase8_selector_v3_fit_report_20260816.json` | v3 训练侧指标、留族审计、最差样本和 Gate 判定的完整报告。 | `7D60B1178B4F9D542FE8E29BBD7D036BC58F5E1A588A35CCEC3ECBEA59237F8F` |

# 2026-08-23 新增机器可读产物

| 路径 | 数据说明 | SHA-256 |
|---|---|---|
| `docs/Q-TopoMoE_Phase8_selector_independent_gate_accepted_20260823.json` | 冻结 selector 在 45-cell、四候选、五重复独立集上的完整 Gate 与逐 workload 决策；p95 regret 为 11.50%，满足当前 12% 门槛。 | `9528F638CE93F30CBAA410E490875CB74415EB5D798EA82292A2B4EA37D654E8` |
| `docs/Q-TopoMoE_Phase8_selector_independent_manifest_20260823.json` | 900次正式测量的完整性、输入/输出哈希、Gate 摘要和 SGLang 恢复状态。 | `6403F9C23F8AD0DA9B812A97C6C7DB49CC0442DA9E3B1D97FC482716F23043E9` |
| `configs/strategies/phase8_selector_gate_policy_v2_20260823.json` | 当前 selector 准入政策；p95 regret 上限 12%，并要求 median regret、决策开销和不可行配置误选率同时达标。 | `7580C10300A56908343751212B7D64C69770C44D694758ADD4E86C545A6A7B38` |
| `configs/experiments/phase8_warm_placement_capture_v1.json` | identity 暖态逐层专家计数采集计划；正式有效结果来自独立 v3 输出目录。 | `B5A9679051F12B8C8AE5A56CA026A6102456E457EB348F0B5573D79A9CE62919` |
| `configs/strategies/phase8_warm_swap_008_slots_per_layer_v1.json` | 从48个暖态窗口生成的最小迁移量候选；文件内状态是生成时快照，当前准入必须同时核对最终 accepted 摘要、map 哈希与运行时补丁，不能孤立加载。 | `6EC449552D04041BEF086C31661CF832F3C60C301086399828C4201EC08F5740` |
| `configs/experiments/phase8_warm_placement_nvfp4_aux_regression_v2.json` | NVFP4 Marlin 两个辅助尺度迁移修复后的 8 条定向 A/B/A 回归计划，结果 accepted。 | `5596EB51E7F8F387D48607F291B7E9615F3C74887963214D12E3A8E2B4F3C9BC` |
| `configs/experiments/phase8_warm_placement_quality_equivalence_v3.json` | 修复后完整 116 题质量 A/B/A 冻结计划；三轮零失败零截断，最终质量 Gate accepted。 | `C7D8391C1C3955EFD792AB792F4DF2EE3530426B25DA0040E53C043D52600DC2` |
| `configs/strategies/phase8_route_stability_gate_policy_v2_20260825.json` | 当前路由稳定性政策；placement excess TV p95 上限 1%，短轨迹 0.045%、长轨迹 0.718%，均满足要求。 | `938493B11E9E8EFB11611EAB50299B22F64549D2736301BC1AE944EF9FD83EA0` |
| `configs/experiments/phase8_warm_placement_limited_canary_v1.json` | 新暖态 placement 候选的有限 canary 冻结计划；使用显式激活文件，要求零失败、计划哈希一致和恢复 p99≤基线 105%，既有 SGLang 按用户要求不纳入本轮管理或 Gate。 | `62E4E3EEA760CD99B9E0A1A69C47FADEF6BD86F062C67A3F559DCB4CE41C39FD` |
| `configs/experiments/phase8_warm_placement_closed_loop_acceptance_v1.json` | 有限 canary 接受后的自动闭环冻结计划；验收三窗口 trigger、十窗口 cooldown、三窗口 placement rollback、8-rank 提交、决策开销和回滚后 p99。 | `BEF8884DF0F16BD363E1019FA6283DFB7BA7EE5B03F0FF9448BAD07C0F9E70F1` |
| `docs/Q-TopoMoE_Phase8_warm_placement_final_acceptance_20260825.json` | 当前 1% 路由 Gate、有限 canary 与自动闭环的最终机器摘要；Phase 8 技术验证状态为 accepted。 | `CED91CC146C6109409777B43602EA0B852EB83993350D0383F4D6A0879880C9A` |
| `docs/results/phase8_warm_placement_final_acceptance_20260825.md` | Phase 8 暖态 placement 最终中文验收报告；记录 384 请求 canary 和 20 窗口自动闭环结果，并明确生产部署未执行。 | `7D56229DB3AF36CF9B507F33DDD576C58D24BF0D5B88EA9A4EF626FDD63896A3` |
| `docs/Q-TopoMoE_release_manifest_20260825.json` | 技术验收版 release manifest；绑定推荐模型路径、候选文件与 map、运行时补丁、当前政策及最终 Gate，并声明生产部署尚未执行。 | `12FE3BA6844509971891072A5F292FBBC5D8AD4726F90E12C16F13D626867FB0` |
| `docs/Q-TopoMoE_项目发布与生产部署清单_20260825.md` | 最终中文交付清单；说明唯一推荐组合、证据语义、发布检查，以及生产小流量、扩量、闭环启用和回滚步骤。 | `593958B4AF915C92820714D0AE1BBBFC2AD3F161E73FCE531E201B467E0F9FCB` |
