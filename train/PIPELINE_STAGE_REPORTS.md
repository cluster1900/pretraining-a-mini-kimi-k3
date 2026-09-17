# v2 数据准备阶段报告台账

本文件是验收索引。每个阶段完成后，控制器把服务器上的真实报告、报告 SHA256 和对应脚本版本记录到服务器项目 `train/reports/background/<run>/`，再进入下一阶段；后续检查时同步小型报告到本地 `train/`。合成回归不计作真实语料完成证据。

## 2026-09-17 用户批准审核子集后的重建（优先于下方历史快照）

用户已明确批准 Glaive/MetaMath 两个子集并授权后台执行，238,688条候选经原清洗规则保留238,672条。批准仅覆盖SHA256为 `3a6a1610d8ce9a62788aa9d0b5a170bf0b265587b2f89215c3179d0959b7d25d` 的审核版，原始全量未获批准。`SOURCE_REVIEW.json` 已改为schema 2并绑定审核输入、报告、脚本与canonical/cleaned报告哈希；批准记录SHA256为 `4a276b17f929a9c7442a87fdfb49e0790e35e69a98939f40ba948c1c005e6263`。

本轮11来源canonical→cleaned元数据链通过；服务器38项数据回归通过；真实tokenizer的11来源合成整链路回归通过。全局去重及所有下游重新生成，不能沿用下表旧全量11/11作为本轮完成证据。自动链路终点为manifest审计与真实manifest功能smoke，正式SFT仍需预训练checkpoint与SFT短跑。

启动证据：控制器PID101860及子进程101863已在SSH断开后继续执行全局精确去重，OpenHermes这一阶段保留238,672条（train236,270 / validation2,402）。服务器阶段归档路径 `/data/mini-k3/project/train/reports/background/1789617455875445126-101860/`。本地 `train/reports/verification/approved-20260917/INDEX.json` 记录本轮初始10份报告哈希；已核验批准与canonical/cleaned绑定一致。应用内自动跟进两次创建均因自动审批超时失败，当前仅服务器控制器自动推进与归档，尚无主动通知或代理自动修复。

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

`train/reports/real/STAGE_ARCHIVE_INDEX.json` 登记43份真实报告的SHA256、来源和统计；`train/reports/code-snapshots/` 保存四阶段按脚本SHA256命名的代码副本。归档核对每份报告的脚本哈希与本地快照一致。此前归档快照中的近重复去重为10/11、Dolma未完成；服务器当前续跑已补齐为11/11，但这不等于 manifest 或来源审核通过。中间语料正文未复制到本地，也未在本地重做全量正文hash。

远端重新执行前三阶段元数据链审核通过，原始报告已复制到 `train/reports/verification/stage-chain-through-deduped.json`。新增来源审核门禁的真实tokenizer合成回归通过，报告为 `train/reports/verification/v2-end-to-end-source-gate.json`；该报告仅证明合成fixture，不证明生产数据终态。本地8项测试通过，覆盖缺失来源审核文件、未批准来源和缺少证据的拒绝路径。

验收代码 `stage_audit.py`、`finalize_v2.py`、合成回归和规划已成功同步服务器。生产来源审核仍未通过，OpenHermes子来源清单已复制到 `train/reports/verification/openhermes-subsource-inventory.json`，不得生成或声称通过训练准入审核。

## 2026-09-17 门禁修复与当前状态

服务器已写入 `/data/mini-k3/data/prepared-v2/SOURCE_REVIEW.json`（SHA256 `72cc98ea7154f99c733dbb7e8728d760bb5a5adddc9b7ef9bb85e4cb2841bb58`）。该记录覆盖11个来源，其中10个有已登记证据，OpenHermes 保持 `training_approved=false`、整体 `status=pending`；因此它是阻断记录，不是训练准入证明。`continue_v2.py` 已改为在任何昂贵阶段前检查该门禁，失败状态写入 `PIPELINE_STATUS.json` 的 `stage=source_review`。当前六个数据阶段的完成标记仍为11/11，但没有生成 manifest、AUDIT 或真实数据 smoke，正式训练未启动。

2026-09-16 全量来源元数据盘点结果：1,001,551条原始记录中496,743条缺失source，414,062条仅含conversations；没有缺失source的记录含非空字符串dataset/dataset_name/subset/origin/source_name字段。仅凭已有字段不能完成来源审核，training_approved仍为false。报告已归档 `train/reports/verification/openhermes-metadata-inventory-20260916.json`，本地核对脚本SHA256、schema计数及总记录守恒通过。后续需要上游精确匹配证据或经过记录的来源过滤方案；本次没有更改生产语料。

## 2026-09-16 11:15 阶段推进

全局近重复去重已完成 11/11；`stage-chain-through-near-deduped.json` 已在远端通过并归档到 `train/reports/verification/`。此前 2026-09-16 11:15 快照中的 13-gram 去污染为 9/11（OpenWebMath 与 Dolma 仍在处理）；服务器当前已补齐为 11/11，快照仍保留在 `train/reports/real/stage-snapshot-20260916-1115/`。
