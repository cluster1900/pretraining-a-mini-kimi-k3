# v2 数据准备阶段报告台账

本文件是验收索引。每个阶段完成后，控制器把服务器上的真实报告、报告 SHA256 和对应脚本版本记录到服务器项目 `train/reports/background/<run>/`，再进入下一阶段；后续检查时同步小型报告到本地 `train/`。合成回归不计作真实语料完成证据。

## 2026-09-26 supplement-v2 (CodeSearchNet 增补) 阶段完成与终态审计台账

为彻底填平 `code-python` 在 10B 固定配比下的 44.1M tokens 缺口，启动隔离运行根目录 `/data/mini-k3/data/prepared-v2-supplement-v2`（控制器 PID 51070，归档于 `train/reports/background/1790072703778497846-51070/`）。

截至 2026-09-24 22:15，**所有 11 个数据源的重度数据处理与 Token 编码阶段已 100% 全部完成**：
1. **统一格式转换 (canonical) & 清洗 (cleaned)**：`code-python` 成功整合基线 8 分片 + 补充 200 分片 + CodeSearchNet 4 分片（455,243 条优质 Python 函数），产出 1,438,545 条 clean 样本；其余 10 来源重绑通过。11/11 已归档。
2. **全局精确去重 (deduped)**：11 个数据源共保留 18,007,744 篇文档，其中 `code-python` 保留 1,274,083 篇。11/11 已归档。
3. **全局 MinHash 近去重 (near-deduped)**：2026-09-24 12:08 全部完成。`code-python` 滤除近重复 45,328 篇（保留 1,228,755 篇）；`fineweb-edu` 滤除 235,709 篇（保留 4,931,516 篇）；全部 11 来源通过。11/11 已归档。
4. **13-gram 评测基准去污染 (decontaminated)**：2026-09-24 18:55 全部完成，11 来源报告绑定统一 13-gram 索引。11/11 已归档。
5. **Tokenizer 编码 (tokenized)**：2026-09-24 22:15 全部完成。产出全部 uint32 shards 与 document ledgers。11/11 已归档。
6. **10B 预训练配比核验 (7/7 全部达标)**：总可用预训练训练 Token 达 **28,214,510,016 (28.21B)**。`code-python` 达 **1,288,200,254**（要求 1,225,000,878，**净盈余 +63.2M tokens**）；其余 6 领域均大幅盈余，彻底满足 10B 固定混合纯净无重复采样。
7. **终态 Manifest 审计 (finalize)**：2026-09-24 22:15 执行 `finalize_v2.py` 时在 `openassistant` 触发 `ValueError: Train/validation group overlap`。全库 16,097,968 个唯一分组经全量哈希扫描确认：其余 10 个数据源 0 冲突，仅有 `openassistant` 的 1 个对话树（`bfe63f8ebe9065b57ad3b71b1ad7e22cea8a12d59e99a72e9979024af633f914`）存在 3 条分支与 1 条验证集分支跨 split，等待执行收敛修复并生成最终 Manifest 与 Smoke。

| 阶段 | 权威完成条件 | 本地报告路径 | 当前状态 |
|---|---|---|---|
| 统一格式转换 (canonical) | 11 个来源 canonical 完成，来源/分片/统计守恒通过 | `train/reports/background/1790072703778497846-51070/canonical/*/COMPLETE.json` | 11/11 已归档并通过 |
| 清洗 (cleaned) | 11 个来源清洗报告，拒绝原因/守恒/上游哈希通过 | `train/reports/background/1790072703778497846-51070/cleaned/*/COMPLETE.json` | 11/11 已归档并通过 |
| 全局精确去重 (deduped) | 11 个来源精确去重完成，数据库哈希守恒 | `train/reports/background/1790072703778497846-51070/deduped/*/COMPLETE.json` | 11/11 已归档并通过 |
| 全局近去重 (near-deduped) | 11 个来源 MinHash-LSH 完成，Jaccard>=0.9 滤除 | `train/reports/background/1790072703778497846-51070/near-deduped/*/COMPLETE.json` | 11/11 已归档并通过 (09-24 12:08) |
| 13-gram 去污染 | 七项评测索引绑定，11 个来源过滤报告存在 | `train/reports/background/1790072703778497846-51070/decontaminated/*/COMPLETE.json` | 11/11 已归档并通过 (09-24 18:55) |
| Tokenizer 编码 | 11 个来源 uint32 / sft jsonl 编码完成，指纹一致 | `train/reports/background/1790072703778497846-51070/tokenized/*/COMPLETE.json` | 11/11 已归档并通过 (09-24 22:15) |
| Manifest 全量审计 | 4 份 manifest + AUDIT.json 生成，split 0 冲突 | `train/reports/manifests.json` | 待修复 openassistant 单组重叠并重跑 |
| 模型 Smoke 验证 | 真实 manifest smoke 梯度/损失有限/显存报告通过 | `train/reports/smoke.json` | 待 manifest 审计完成后执行 |

## 2026-09-17 用户批准审核子集后的重建（历史记录）

阶段报告由 `train/data/stage_audit.py` 和 `train/data/finalize_v2.py` 生成或核验；最终 smoke 由 `train/smoke_from_manifest.py` 生成。所有报告必须保留服务器报告中的 `script_sha256`、输入/输出哈希和生成时间。

已通过但不等于真实语料完成的验证：`v2-end-to-end-stage-chain.json` 合成回归、`stage-chain-through-deduped.json` 元数据链检查、`test_stage_audit.py` 七项防回归测试，以及 `test_smoke_audit.py` 五项绑定测试。

## 2026-09-16 归档与验收更新

`train/reports/real/STAGE_ARCHIVE_INDEX.json` 登记43份真实报告的SHA256、来源和统计；`train/reports/code-snapshots/` 保存四阶段按脚本SHA256命名的代码副本。归档核对每份报告的脚本哈希与本地快照一致。此前归档快照中的近重复去重为10/11、Dolma未完成；服务器当前续跑已补齐为11/11，但这不等于 manifest 或来源审核通过。中间语料正文未复制到本地，也未在本地重做全量正文hash。

远端重新执行前三阶段元数据链审核通过，原始报告已复制到 `train/reports/verification/stage-chain-through-deduped.json`。新增来源审核门禁的真实tokenizer合成回归通过，报告为 `train/reports/verification/v2-end-to-end-source-gate.json`；该报告仅证明合成fixture，不证明生产数据终态。本地8项测试通过，覆盖缺失来源审核文件、未批准来源和缺少证据的拒绝路径。

验收代码 `stage_audit.py`、`finalize_v2.py`、合成回归和规划已成功同步服务器。生产来源审核仍未通过，OpenHermes子来源清单已复制到 `train/reports/verification/openhermes-subsource-inventory.json`，不得生成或声称通过训练准入审核。

## 2026-09-17 门禁修复与当前状态

服务器已写入 `/data/mini-k3/data/prepared-v2/SOURCE_REVIEW.json`（SHA256 `72cc98ea7154f99c733dbb7e8728d760bb5a5adddc9b7ef9bb85e4cb2841bb58`）。该记录覆盖11个来源，其中10个有已登记证据，OpenHermes 保持 `training_approved=false`、整体 `status=pending`；因此它是阻断记录，不是训练准入证明。`continue_v2.py` 已改为在任何昂贵阶段前检查该门禁，失败状态写入 `PIPELINE_STATUS.json` 的 `stage=source_review`。当前六个数据阶段的完成标记仍为11/11，但没有生成 manifest、AUDIT 或真实数据 smoke，正式训练未启动。

2026-09-16 全量来源元数据盘点结果：1,001,551条原始记录中496,743条缺失source，414,062条仅含conversations；没有缺失source的记录含非空字符串dataset/dataset_name/subset/origin/source_name字段。仅凭已有字段不能完成来源审核，training_approved仍为false。报告已归档 `train/reports/verification/openhermes-metadata-inventory-20260916.json`，本地核对脚本SHA256、schema计数及总记录守恒通过。后续需要上游精确匹配证据或经过记录的来源过滤方案；本次没有更改生产语料。

## 2026-09-16 11:15 阶段推进

全局近重复去重已完成 11/11；`stage-chain-through-near-deduped.json` 已在远端通过并归档到 `train/reports/verification/`。此前 2026-09-16 11:15 快照中的 13-gram 去污染为 9/11（OpenWebMath 与 Dolma 仍在处理）；服务器当前已补齐为 11/11，快照仍保留在 `train/reports/real/stage-snapshot-20260916-1115/`。
