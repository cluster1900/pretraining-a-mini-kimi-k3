# v2 数据准备阶段报告台账

本文件是本地验收索引。每个阶段完成后，必须把服务器上的真实报告、报告 SHA256 和对应脚本版本记录到 `train/`，再进入下一阶段。合成回归不计作真实语料完成证据。

| 阶段 | 权威完成条件 | 本地报告 | 当前状态 |
|---|---|---|---|
| 统一格式转换 | 11 个来源 canonical 完成，来源、输入文件哈希、统计守恒通过 | `train/reports/real/canonical/*/COMPLETE.json` | 11/11 已归档；远端元数据链检查通过 |
| 清洗 | 11 个来源清洗报告，拒绝原因、统计守恒、上游哈希通过 | `train/reports/real/cleaned/*/COMPLETE.json` | 11/11 已归档；远端元数据链检查通过 |
| 去重 | 全局精确去重完成，训练/验证分组隔离 | `train/reports/real/deduped/*/COMPLETE.json` | 11/11 已归档；远端元数据链检查通过 |
| 13-gram 去污染 | 七项评测索引非空，11 个来源报告绑定同一索引 | `train/reports/decontaminated.json` | 待真实流水线终态 |
| tokenizer 编码 | 11 个来源编码完成，真实 tokenizer 指纹、EOS、统计守恒通过 | `train/reports/tokenized.json` | 待真实流水线终态 |
| uint32 shards | 小端 uint32、文件字节数/哈希/词数全部通过 | `train/reports/shards.json` | 待真实流水线终态 |
| manifest | 四份 manifest 与 AUDIT 哈希绑定，配比、来源、split 通过 | `train/reports/manifests.json` | 待真实流水线终态 |
| smoke test | 真实 manifest smoke 两次优化更新通过，有限性/梯度/显存报告存在 | `train/reports/smoke.json` | 待真实流水线终态 |

阶段报告由 `train/data/stage_audit.py` 和 `train/data/finalize_v2.py` 生成或核验；最终 smoke 由 `train/smoke_from_manifest.py` 生成。所有报告必须保留服务器报告中的 `script_sha256`、输入/输出哈希和生成时间。

已通过但不等于真实语料完成的验证：`v2-end-to-end-stage-chain.json` 合成回归、`stage-chain-through-deduped.json` 元数据链检查、`test_stage_audit.py` 七项防回归测试，以及 `test_smoke_audit.py` 五项绑定测试。

## 2026-09-16 归档与验收更新

`train/reports/real/STAGE_ARCHIVE_INDEX.json` 登记43份真实报告的SHA256、来源和统计；`train/reports/code-snapshots/` 保存四阶段按脚本SHA256命名的代码副本。归档核对每份报告的脚本哈希与本地快照一致。近重复去重仍为10/11；Dolma未完成，不能把精确去重11/11表述为全部去重完成。中间语料正文未复制到本地，也未在本地重做全量正文hash。

远端重新执行前三阶段元数据链审核通过，原始报告已复制到 `train/reports/verification/stage-chain-through-deduped.json`。新增来源审核门禁的真实tokenizer合成回归通过，报告为 `train/reports/verification/v2-end-to-end-source-gate.json`；该报告仅证明合成fixture，不证明生产数据终态。本地8项测试通过，覆盖缺失来源审核文件、未批准来源和缺少证据的拒绝路径。

验收代码 `stage_audit.py`、`finalize_v2.py`、合成回归和规划已成功同步服务器。生产来源审核仍未通过，OpenHermes子来源清单已复制到 `train/reports/verification/openhermes-subsource-inventory.json`，不得生成或声称通过训练准入审核。

2026-09-16 全量来源元数据盘点结果：1,001,551条原始记录中496,743条缺失source，414,062条仅含conversations；没有缺失source的记录含非空字符串dataset/dataset_name/subset/origin/source_name字段。仅凭已有字段不能完成来源审核，training_approved仍为false。报告已归档 `train/reports/verification/openhermes-metadata-inventory-20260916.json`，本地核对脚本SHA256、schema计数及总记录守恒通过。后续需要上游精确匹配证据或经过记录的来源过滤方案；本次没有更改生产语料。

## 2026-09-16 11:15 阶段推进

全局近重复去重已完成 11/11；`stage-chain-through-near-deduped.json` 已在远端通过并归档到 `train/reports/verification/`。13-gram 去污染正在运行，已完成 9/11，OpenWebMath 与 Dolma 仍在处理；已完成报告快照归档到 `train/reports/real/stage-snapshot-20260916-1115/`。
