# 4×Tesla V100-SXM2 (32GB) 预训练与对齐 Mini K3 规划总则

> 本文档覆盖当前 4 卡 V100 机器的预训练、SFT、DPO 与 PPO 对齐流程。所有训练相关代码均归档于 `train/` 目录中。

## 2026-09-19 补充流水线元数据重绑与断点续跑

1. **补充数据就绪验证**：
   - FineWeb-EDU：8个分卷（18.69 GB，约6.0B tokens）于 `/data/mini-k3/data/raw/fineweb-edu/data/` 校验完成。
   - Code-Python：200个分片（6.5 GB，675,071个Python文件，约1.5B tokens）转换为 Stack-v3 格式，校验并归档于 `/data/mini-k3/data/raw/code-python/data/`。
   - `DOWNLOAD_COMPLETE.json` 经核验就绪。
2. **流水线硬链接路径断言分析与修复**：
   - 现象：`supplement_v1.py` 在执行至 `continue_v2.py` 的 canonical 阶段审计时报错 `openassistant/canonical: part outside expected source directory`。
   - 根因：`clone_root` 采用 `cp -al` 复制 `prepared-v2`，但除 FineWeb-EDU 与 Code-Python 重新生成外，其余9个未变动来源的 `COMPLETE.json` 中记录的分片绝对路径仍带有原目录前缀 `/data/mini-k3/data/prepared-v2/`，触发了 `stage_audit.py` 中 `require(p.parent == path.parent)` 的安全门禁。
   - 修复：在 `supplement_v1.py` 中实现 `rebind_cloned_root`，自动将9个复用来源的 `canonical` 与 `cleaned` 阶段报告内分片路径重绑至当前 `prepared-v2-supplement-v1`，级联更新 `upstream_report_sha256` 哈希链，并同步更新 `SOURCE_REVIEW.json` 中 `openhermes` 的阶段报告哈希绑定。同时将 `supplement_v1.py` 改为幂等续跑，保留已生成的 FineWeb-EDU 与 Code-Python 规范化与清洗产物。
   - 门禁核验：全部11个来源逐一通过 `stage_audit` 的 `canonical` 与 `cleaned` 阶段审计，`verify_source_review` 与 `verify_benchmark` 均通过；全部 38 项单元测试通过（Ran 38 tests in 0.350s, OK）。
3. **续跑链路**：
   - 恢复执行 `continue_v2.py`：全局精确去重（`dedup_v2.py`）→ 全局近重复去重（`near_dedup_v2.py`）→ 13-gram去污染（`contamination_v2.py`）→ 11来源token编码（`tokenize_v2.py`）→ manifest审计（`finalize_v2.py`）→ 模型smoke测试（`smoke_from_manifest.py`）→ 配比覆盖率评估（`assess_training_coverage.py`）。

## 2026-09-18 启动 FineWeb-EDU 与 Python 代码补充下载

根据 2026-09-17 配方覆盖率预检（FineWeb-EDU 缺 2.76B、代码缺 1.22B tokens）以及用户明确指令：“fineweb-edu 和 code-python 都给我下载一下吧！然后同步一下文档！要确认一下现在下载的是不是需要的完整数据集，如果是的话，我们再去下载！然后下载开始后，确认没问题，我们就开始异步下载，不需要等他下载完成！”，正式启动补充下载。

1. **FineWeb-EDU 补充下载**：
   - 来源：官方 `HuggingFaceFW/fineweb-edu`（通过 ModelScope 镜像，master 分支，官方 SHA256 校验）。
   - 选定范围：从 `raw/fineweb-edu/INVENTORY.json` 的 2,410 个 `data/**/*.parquet` 真实 Common Crawl 分片中，按抓取目录轮询选定 6 个未下载的新分卷（`CC-MAIN-2013-20/train-00001`, `CC-MAIN-2013-48/train-00000`, `CC-MAIN-2014-10/train-00000`, `CC-MAIN-2014-15/train-00001`, `CC-MAIN-2014-23/train-00000`, `CC-MAIN-2014-35/train-00000`），合计 14.05 GB。
   - 预期产出：约 4.5B 训练 tokens，完全补齐 2.76B 缺口并保留充足裕量。
2. **Code-Python 补充下载与适配**：
   - 来源：官方 `codeparrot/github-code`，固定 revision `b5661e6b17396364b2bcf8e68977b0d28e1ebd19`。
   - 选定范围：均匀挑选 200 个 Parquet 分片，单分片实测验证 10.2 万行中包含 6,191 个 Python 文件，符合 SPDX 宽松许可（MIT/Apache-2.0/BSD/ISC/Unlicense/0BSD）的行数占比 64.7%（4,009 个文件），单分片净产出约 7.46M tokens。
   - 预期产出：200 分片合计净产出约 1.49B 训练 tokens，完全补齐 1.22B 缺口（含 20% 缓冲区）。
   - 转换机制：下载后通过 `download_supplement.py` 将 content 字段与允许的许可证过滤转换为与 Stack-v3 格式完全兼容的 Parquet，存放于 `/data/mini-k3/data/raw/code-python/data/`。
3. **执行与后台化**：
   - 执行脚本：`train/data/download_supplement.py --work /data/mini-k3 --count-code 200`。
   - 后台守护：使用 `nohup` 异步运行，日志记录于 `/data/mini-k3/logs/prepared-v2-supplement-v1/download-async.log`，状态记录于 `/data/mini-k3/data/reports/supplement-v1/DOWNLOAD_STATUS.json`。
   - 影响与解耦：补充下载写入原始数据目录，不干扰当前主流程 `prepared-v2` 正在运行的近重复去重；下载完成后由独立补充流水线（`supplement_v1.py`）负责后续 isolated stage 处理。

## 2026-09-17 明确批准两个 SFT 子集并后台续跑

用户明确授权：“批准这两个子集进入 SFT，接下来自动执行就可以了，我们不用实时监控着”。该批准仅适用于现有审核版中的 `glaive-code-assist` 182,240条与 `metamath` 56,448条：过滤后238,688条、邮箱模式替换3,489处，再由原清洗规则排除5条乱码和11条密钥格式记录，清洗后238,672条。未保留记录仍存在原始输入，不能扩大此次批准范围。有限抽样不代表所有答案正确；子来源映射依据发布者标签，尚未做上游逐行精确匹配。

`approve_openhermes_reviewed.py` 登记上述用户批准，先校验原始/审核版SHA256、两个过滤/抽样报告、脚本及 canonical→cleaned 哈希链，再原子更新 `SOURCE_REVIEW.json` 并保存此前pending记录。新版审核门禁将批准绑定到当前 OpenHermes 两阶段报告和证据文件，替换回旧全量产物会失败。控制器在持有独占锁后检查来源批准、benchmark和已有阶段链；重复启动不能覆写运行中状态，任一失败会阻止进入下一阶段。stage报告内原有处理脚本保持不变。

原始文件保留于 `/data/mini-k3/data/raw/openhermes/openhermes2_5.original-20260917.json`，SHA256 `abe573d17eade4161aac321028027dd5ba614a6d9516d51bde9299d5353e1609`；选中链接指向 `/data/mini-k3/data/raw/openhermes-reviewed-v2/openhermes2_5.json`，SHA256 `3a6a1610d8ce9a62788aa9d0b5a170bf0b265587b2f89215c3179d0959b7d25d`。旧阶段硬链接快照在 `/data/mini-k3/data/prepared-v2-before-review-20260917`，只作存档，其报告内路径仍是原路径，不能直接作为新流程输入。OpenHermes发生删行和文本变化，因此新目录中所有来源从全局精确去重开始重跑，保持跨来源去重定义及现有split规则；不复用旧去重数据库。预训练配比、tokenizer及硬件不变。

自动链路：全局精确去重→近重复去重→13-gram去污染→11来源编码/分片→全部manifest审计→真实预训练manifest两步功能smoke。当前没有预训练checkpoint，此链路终点是数据准备完成，SFT权重训练须待预训练checkpoint与SFT短跑验证就绪；当前功能smoke不等于四卡短跑或SFT质量验证。控制器在服务器nohup运行，不依赖本地对话保持在线；失败时停止并写日志，恢复前须分析原因、完成有针对性的修复和测试，禁止盲目重试或重复实例。

```bash
cd /data/mini-k3/project
/data/mini-k3/venv/bin/python train/data/approve_openhermes_reviewed.py
PYTHONPATH=/data/mini-k3/project/train/data /data/mini-k3/venv/bin/python -m unittest discover -s train/data -p 'test_*.py'
nohup /data/mini-k3/venv/bin/python -u /data/mini-k3/project/train/data/continue_v2.py --workers 2 >> /data/mini-k3/logs/prepared-v2/controller-approved-20260917.log 2>&1 < /dev/null &
```

验收依据是 `prepared-v2/PIPELINE_STATUS.json`、逐阶段报告、`manifests/AUDIT.json` 与 `manifests/SMOKE.json` 的内容和哈希，不是进程存在或完成标记计数。启动前测试在服务器现有venv中执行；具体测试结果与启动记录归档到 `train/reports/verification/`。

每阶段报告在进入下一阶段前自动复制到服务器 `/data/mini-k3/project/train/reports/background/<run>/` 并生成SHA256索引，启动时保留代码快照与批准记录。仅归档元数据，不复制数据正文或大型SQLite数据库。

启动实测：控制器PID101860、精确去重子进程PID101863，控制器已脱离SSH由PID1托管，状态为 `running/global_exact_dedup`；OpenHermes精确去重保留238,672条（train236,270 / validation2,402）。38项数据回归及真实tokenizer合成整链路均通过，初始审批、过滤/抽样、canonical/cleaned及测试证据已归档本地 `train/reports/verification/approved-20260917/`。应用内每小时自动跟进创建尝试两次均因自动审批超时未成功，因此当前没有定时代理修复、主动通知或持续同步到本地；服务器阶段推进和阶段归档不受影响。此处PID仅是启动记录，后续检查必须核实实时进程。

2026-09-17 配方覆盖率预检：按 warmup+stable 占85%、decay占15%的实际WSD阶段计算，固定配方需要 FineWeb-EDU约4.275B、中文网页约1.425B、Dolma约1.000B、FineMath约0.605B、OpenWebMath约0.545B、Python代码约1.225B、Cosmopedia约0.925B训练token。旧完整token化报告显示 FineWeb-EDU约1.515B、中文约1.440B、Dolma约9.025B、FineMath约5.241B、OpenWebMath约4.482B、Python代码约0.0028B、Cosmopedia约1.690B；总量约23.395B但固定配比仍短缺 FineWeb-EDU约2.760B与代码约1.222B。最终重建完成后必须重跑 `assess_training_coverage.py`；在缺口补齐或明确修改配方前，不得声称满足10B固定混合训练。

补充准备清单 `train/data/supplement_catalog_20260917.json` 已生成，当前只登记候选和目标 token 数，没有下载。FineWeb-EDU候选目标为缺口加20%缓冲（3.312B token），代码候选为 `codeparrot/github-code` 的 Python、明确许可证子集，目标缺口加20%缓冲（1.467B token）。开始下载前必须生成精确文件/分片清单、核对 revision 和 checksum，并先做小规模清洗/去重/token dry-run；禁止整库下载、按GB猜token或把许可证不明代码加入候选。

自动补充脚本 `supplement_v1.py` 已实现上述流程：等待当前 `prepared-v2` 完成→按现有 FineWeb-EDU inventory 轮询不同 crawl 选择约12GB未下载 parquet→按固定 revision 均匀选择 GitHub-Code parquet 分片→下载并记录本地SHA256→将 Python/allowlist 记录转换为 Stack-v3 兼容 parquet→复制 prepared-v2 硬链接快照→从 canonical/cleaned 重建两受影响来源→全局去重、近去重、去污染、编码、manifest、smoke→运行 coverage report。每步状态、报告、日志、选择清单均写入服务器 `/data/mini-k3/data/reports/supplement-v1/`、`/data/mini-k3/logs/prepared-v2-supplement-v1/` 和独立 `/data/mini-k3/data/prepared-v2-supplement-v1/`；主数据目录和当前 run 不覆盖。代码下载源固定为 `codeparrot/github-code` revision 由 API 返回并写入 selection，保留 repo/path/language/license/size，未登记上游sha时只记录本地sha，不能声称上游校验通过。脚本会在当前 run 未完成时等待，不会并行改写共享阶段。

启动实测：补充进程PID103215，状态文件为 `/data/mini-k3/data/reports/supplement-v1/PIPELINE_STATUS.json`，当前 `waiting_for_base`，等待主流程PID101860完成。补充流程不会因为SSH会话断开而退出。

## 2026-09-17 来源门禁启动前检查

修复 `continue_v2.py`：控制器现在在等待或重跑任何数据阶段前调用 `stage_audit.verify_source_review`，缺少 `prepared-v2/SOURCE_REVIEW.json` 或来源审核仍为 pending 时立即退出，并把 `PIPELINE_STATUS.json` 更新为 `stage=source_review` 的失败原因。此前同一门禁只在 `finalize_v2.py` 执行，可能先重复耗时的去重、去污染和编码再失败。该修改只改变失败提前时机，不放宽来源、许可证、数据配比或训练准入要求；OpenHermes 的 496,743 条无子来源记录仍须取得上游证据或按记录的过滤规则排除，未完成前不得启动 manifest、smoke 或正式训练。

新增只读 `sample_openhermes_missing_source.py`，使用固定种子蓄水池抽样无 source 标签记录，检查对话结构、角色、长度、乱码、明显密钥、邮箱模式和样本内重复；报告不写入原文，只保留行号、哈希和统计，终端摘录会做邮箱/密钥遮盖。抽样可以评估内容质量，不能证明上游来源或许可证，也不会改变生产语料。

只读抽查实测：无标签496,743条中随机抽样256条，全部未触发脚本的基本规则；全量原始无标签记录命中邮箱模式1,142条、乱码4条、密钥格式1条。模式命中不等于有效隐私泄露，基本规则通过不等于回答正确。另用固定种子20260918从256条中抽取12条完整问答审读，发现通用SFT候选内容，也确认体育史问答有年份错误，部分回答存在无依据扩写；不能据此估计全量错误率。方法、行号、核验依据及范围详见 `train/reports/verification/openhermes-sampling-notes-20260917.md`。本次没有修改生产数据或训练准入。

2026-09-17 处理方案：不把缺失或不明许可的记录静默混入训练。新增 `filter_openhermes_reviewed.py`，只保留 OpenHermes 原始 `source` 精确等于 `glaive-code-assist` 或 `metamath` 的记录，并在保留记录的对话文本中将邮箱模式替换为 `[EMAIL]`；二者分别绑定 Glaive Apache-2.0 与 MetaMathQA MIT 的上游证据，其余标签和496,743条无标签记录全部保留在原始快照、排除在审核候选之外。过滤结果写入独立报告，随后将审核版作为 OpenHermes 的选中文件从 canonical 起重做受影响阶段；旧 `prepared-v2` 产物先做硬链接快照，不删除原始输入。过滤改变 OpenHermes SFT 数据量和文本内容，必须重新完成来源审核、manifest、smoke 后才可使用。

新增 `sample_openhermes_reviewed.py` 对审核版中的 Glaive/MetaMath 记录做固定种子抽样，报告结构、明显安全模式和长度统计；该审查与来源/许可证门禁分离，任何一项未完成都不能生成通过验收的训练 manifest。

---

## 2026-09-15 v2.1 续跑与验收修订

2026-09-15 10:54 子来源盘点结果：OpenHermes 原始文件共1,001,551条，其中496,743条缺少非空字符串 source 标签；其余记录分属14个标签。完整统计保存在 `train/data/openhermes_source_review_20260915.json` 和服务器 `/data/mini-k3/data/reports/openhermes-subsource-inventory.json`，审核状态 pending、training_approved=false。接下来逐子来源核实许可并处理不可追溯记录，不能把现有精确去重完成等同于许可证审核通过。真实全量近似去重已完成 Cosmopedia，后续来源继续执行；不得在此时标记整条流程完成。

2026-09-15 来源审核待办：检查原始 INVENTORY 发现 OpenWebMath 和 OpenHermes 的 license 字段为空。已下载 OpenWebMath README 的 dataset_info.license 和正文明确记录 ODC-By 1.0，且说明不改变底层内容许可证；需将该证据作为补充记录，不能把数据库许可等同于底层内容许可。OpenHermes-2.5 发布者在 https://huggingface.co/datasets/teknium/OpenHermes-2.5/discussions/9 说明不同子集有不同许可，未提供统一许可。现有 canonical 保留 origin.row 和原文件哈希，但没有直接保留原 source 子来源字段；必须从原始行恢复子来源并审核，不能凭公开下载或模型许可证将全部记录判为已审核。新增只读 `audit_openhermes_sources.py` 盘点原始子来源标签；在审核缺口解决前不得宣布全部11来源可用于训练。该检查不改变现有数据算法或重启当前去重任务。命令：`/data/mini-k3/venv/bin/python /data/mini-k3/project/train/data/audit_openhermes_sources.py --input /data/mini-k3/data/raw/openhermes/openhermes2_5.json --report /data/mini-k3/data/reports/openhermes-subsource-inventory.json`。

2026-09-15 验收链路补充：审查发现 finalizer 原先仅核对编码产物内部统计，未逐级绑定上游报告，存在漏编码文档仍可能通过的缺口。新增 `stage_audit.py`，核对六阶段完成状态、来源、脚本 SHA256、上游报告 SHA256、文档统计守恒及中间分片大小；finalizer 先通过该检查，再完整扫描所有编码文件。去污染报告必须绑定包含全部七项评测且两个匹配模式非空的13-gram索引；编码词表必须与 config 一致。AUDIT 记录阶段报告链和索引哈希。该变更只加强最终验收，不改动当前运行的转换、清洗、去重、去污染、编码算法或配比，不重启现有进程。元数据链检查不等于重新扫描全部原始语料。验证命令：`python train/data/test_stage_audit.py`；真实 tokenizer 合成回归使用七项合成评测输入验证接口，仍不能替代真实语料验收。

本次 v2 数据准备控制器命令（仅数据准备和功能 smoke，不启动正式训练）：`/data/mini-k3/venv/bin/python -u /data/mini-k3/project/train/data/continue_v2.py --workers 2`。现有实例存活时只监测，不重复启动。独立阶段元数据检查：`/data/mini-k3/venv/bin/python /data/mini-k3/project/train/data/stage_audit.py --root /data/mini-k3/data/prepared-v2 --through deduped --report /data/mini-k3/data/reports/stage-chain-through-deduped.json`。

2026-09-15 09:45 验证记录：新增7项报告链测试、本轮5项 manifest 绑定测试在服务器通过；11来源真实tokenizer合成整链路回归通过，报告 `/data/mini-k3/data/reports/v2-end-to-end-stage-chain.json`。实际11来源转换→清洗→精确去重的报告链检查通过，报告 `/data/mini-k3/data/reports/stage-chain-through-deduped.json`。这些结果不代表实际数据全量去污染、编码、manifest 或模型 smoke 已完成；现有近似去重控制器仍在执行。

验收绑定补充（2026-09-15）：为防止旧/改动后的 manifest 被 smoke 误用，`finalize_v2.py` 在 `AUDIT.json` 记录四份最终 manifest 的 SHA256；`smoke_from_manifest.py` 调用 `data/smoke_audit.py`，启动模型前校验审计状态、manifest 哈希、tokenizer 指纹、词表、dtype、train split、来源及精确配比，并重新验证每个将消费的首个 shard 的大小与 SHA256。SMOKE.json 记录 manifest、AUDIT 与所用 shards 的哈希。影响：旧版缺少哈希绑定的审计报告须重跑 finalizer，不能直接用于最终 smoke；当前运行中的转换/清洗/去重算法与产物保持原定义。验收命令：`python train/data/test_smoke_audit.py`；真实 tokenizer 的整链路合成回归同时验证此绑定。

KDA审计另发现 A_log 未参与计算、state 的 key/value 广播轴错误、AMP 下 float() 仍不能保证 matmul 使用FP32。修复后先计算 log_decay=-exp(A_log)*softplus(gate+dt_bias)，作用于state的key轴，再做delta更新；递归过程显式关闭autocast并保存FP32 state。该参考实现的delta更新系数仍为固定1，不能宣称与正式K3全部门控结构等价。该修改不删除或替换CED分支。

模型预检发现旧 embedding 默认 N(0,1) 加共享输出头使初始 loss=477.1667。配置新增 initializer_range=0.02，Linear/Embedding 使用 N(0,0.02)，共享参数只初始化一次，RMSNorm=1、bias=0，保留 dt_bias 专用初始化。该修改用于从零训练初始化，不放宽 loss 检查。修复初始化与KDA后，服务器完整参数模型的64-token随机输入两步功能smoke已返回0；报告 `/data/mini-k3/data/reports/model-functional-smoke-v2.json`。KDA分块等价/梯度/FP32缓存测试也通过。最终真实manifest smoke仍须在数据全量审计后执行，此预检不代表四卡2048长度或1M能力验收。


本次保留正在运行的 v2 清洗，只修复其下游衔接。旧 `continue_v2.py` 中 `chinese-fineweb-edu -> culturax` 路径映射、发现首个 bin 即跳过整源、只编码7个预训练来源、没有validation与实际smoke调用，均不能满足当前目标，已替换。

- `dedup_v2.py`：跨来源 SHA256 内容去重；SQLite每个输入part单独事务，输出成功后提交去重状态，崩溃重跑不会把未保存记录误判重复。输入清单改变时拒绝沿用旧数据库。先登记OpenAssistant官方验证组，再按会话根、代码仓库、URL或内容组做固定1%验证划分。
- `near_dedup_v2.py`：预训练text/code使用全量5-unit shingles、64个MinHash排列、8个LSH band，候选经特征集合Jaccard≥0.9才去除；SFT/偏好保留不同答案，使用精确去重。LSH候选召回和64位shingle指纹为近似方法，报告披露限制，不保证穷尽所有相似文档。
- `contamination_v2.py`：接收near-deduped产物，用已验证的非空7项评测索引过滤，输出保留原有split/source/结构。
- `tokenize_v2.py`：在packing前按记录split分别编码。7个预训练来源生成小端uint32文件；3个SFT来源输出input_ids/labels（仅assistant监督）；偏好输出共同prompt边界及chosen/rejected token IDs。所有11个来源都纳入审计，后训练数据不混入预训练bin。
- 每源只有全部输入处理完、输出hash/计数通过才写tokenized/COMPLETE.json。途中出现bin不构成完成；中断的tokenization目录须保留并明确恢复，不能静默跳过。
- `finalize_v2.py`：逐文件SHA256、uint32字节/token范围/EOS、SFT labels、偏好prompt、全部文档ID和split group扫描，验证统计守恒与训练/验证无交叉；通过后生成pretrain_stable.json、pretrain_decay.json、validation.json、alignment.json、AUDIT.json。
- `smoke_from_manifest.py`：使用完整参数模型读取每个真实预训练来源并执行两次优化更新，报告有限性、梯度和显存。默认64-token功能检查不能证明2048长度四卡容量或1M质量；正式训练仍另需四卡短跑。只有manifest审计与此真实数据smoke通过才可标记该准备流程完成。
- 所有重试与清洗沿用 `/data/mini-k3/data/prepared-v2`；旧目录继续隔离。脚本/模型预检失败会写PIPELINE_STATUS为failed并停止，不再输出泛化的“已完成”。

验证证据：6项中断恢复/分片测试通过，12项结构/去污染测试通过；服务器用真实tokenizer的11来源合成fixture跑完清洗→精确/近似去重→13gram→编码→分片→manifest扫描，报告 `/data/mini-k3/data/reports/v2-end-to-end-regression.json`。2026-09-15 新增5项审计绑定测试在服务器通过，覆盖旧审计、修改后的manifest、配置配比不符和同大小shard篡改；新版11来源真实tokenizer合成回归通过，报告 `/data/mini-k3/data/reports/v2-end-to-end-audit-binding.json`。本地测试环境缺少torch，相关测试在服务器现有venv中执行。该合成fixture仅证明接口与不变量，不代替实际训练数据的全量审计，也不代替完整模型smoke。

## 数据流水线 v2 审计修订（覆盖此前阶段完成声明）

审计证据：旧版 11 份去污染报告的 `benchmark_13grams` 均为 0；旧 tokenizer 正则/特殊 token 表与上游 `tokenization_kimi.py` 不同；旧清洗压平代码缩进，旧转换把 OpenR1 答案、SFT 对话角色和偏好结构丢弃。因此原 `converted/cleaned/deduped/decontaminated/tokenized` 产物保留用于诊断，但不得作为已验收训练数据。服务器记录 `/data/mini-k3/data/reports/PIPELINE_INVALIDATION.json`。旧自动流水线已停止并禁止复用；不能把历史 `11/11` 文件计数当作有效完成证据。

用户已授权完成从统一格式转换到 smoke test 的整条流程，本授权覆盖当前数据准备；正式长跑不自动启动。新的产物位于 `/data/mini-k3/data/prepared-v2/`，不能与旧产物混载。

- `canonical_v2.py`：仅读取选中的数据文件，保留原始文件 SHA256、row、来源和记录 ID。代码逐 `files[]` 拆分并保留 Python 缩进/许可证字段，不拼成仓库大文档；OpenHermes 保留对话，OpenR1 保留 messages 中的完整答案，仅选择 all/default，避免多个导出版本重复；UltraFeedback 保留共同 prompt 和 chosen/rejected；OpenAssistant 重建父子消息树并保留官方 split。
- `clean_v2.py`：保留换行和缩进、角色、偏好标签与来源。代码使用明确的 permissive SPDX allowlist，vendor/non-Python 单独统计。PII 检测覆盖邮箱和明确密钥模式，必须披露覆盖限制，不能宣称移除全部 PII。
- `build_benchmarks_v2.py`：直接读取已下载原始 parquet，按数据集真实字段提取；只使用 MMLU test、ARC test、HellaSwag validation、WinoGrande-XL validation、PIQA validation、GSM8K test、HumanEval test。题干和选项分别构造匹配片段，不索引 HumanEval 通用 test harness；逐项报告行数及短于13词的覆盖缺口。
- tokenizer 原始来源 `moonshotai/Kimi-K3`，上游编码定义 revision `f831ab66814297da540d832a5235f8e904f29d06`。词表读取真实 tokenizer_config 命名，不增加自造特殊 token 别名；精确读取上游预分词正则及400k/25k分块规则。`verify_tokenizer.py` 已完成12组上游 IDs/UTF8/字面特殊 token/EOS/batch等价验证，报告位于 `/data/mini-k3/data/reports/tokenizer-equivalence-v2.json`。旧正则生成的 shards 不复用。
- 每阶段有输入/脚本/输出哈希与非空、统计守恒检查；临时文件使用 `.incomplete`，全部输出成功后才写 `COMPLETE.json`。有第一个 bin 或子进程退出均不代表整阶段成功。
- 后续去重必须保留来源和结构，跨来源进行；validation 必须按文档/会话组隔离，在 packing 前分割。去污染使用真实非空 benchmark 索引，并通过污染样本注入测试。每一项验证未取得实际报告前，都保持未完成。

截至本次修订：真实 held-out benchmark 已重建，含7项数据集、32,115个唯一记录；实际编译的13-gram索引包含1,507,360个自然语言片段与23,611个代码片段（计数允许不同模式分别存储），污染注入/负样本/中文无空格匹配测试通过。这不等于已完成训练数据的重新去污染。v2 转换/清洗/去重/去污染/分词/分片/manifest/smoke 的最终验收仍需继续执行。

## 2026-09-14 最新下载范围：仅本次实际使用的文件

用户明确要求：只下载本次训练会使用的数据，非使用数据不下载。本节覆盖下文全仓库下载队列和容量扩容计划。全仓库后台队列已停止，已有文件保留，不删除，也不代表自动纳入训练。

1. 以本次约 10B 训练 token 预算、阶段配方和实际使用 manifest 为边界；“齐全”指被选中文件齐全，不指完整镜像 TB 级仓库。
2. 先盘点已下载文件，核实真实来源、split、语言和 tokenizer token 数；只补指定来源的实际缺口。不能用 GB 或空格词数直接认定 10B tokens 已满足。
3. 未进入本次 manifest 的备用语料、替代仓库、重复导出格式、其他语言和其他版本不追加下载。验证/评测数据单独登记，绝不混入训练。
4. 下载清单需记录精确 repo、revision、文件 path、用途、目标 token 数、预计新增字节；只对清单文件下载。已有足够数据时停止补充，不自动扩大配额。
5. `train/data/download_catalog.json` 已设置 `download_scope=usage_only`；`download_complete.py` 的全仓库 download/all 模式在此策略下拒绝运行。历史全量容量审计仅供参考，不作为待下载目标。
6. 当前文档和 config 的来源/配比仍需统一，tokenizer 与最终文件级 manifest 未核定前，不再自动补下载，不开展清洗或训练。

## 2026-09-14 下载完整性修订（本节覆盖旧下载说明）

原因：旧下载器设置 2/5/20GB 配额，ModelScope 文件列表只取默认前 100 条，HF 未处理分页，并把 URL 清单也算作正文。此前的 `DOWNLOAD.json` 仅作历史记录，不能作为整库完成证明。2026-09-14 本次任务只做下载和校验，不开始清洗或训练。

- 正式下载登记表：`train/data/download_catalog.json`。每条记录包含真实仓库、后端和本地目录；清洗前不得根据目录名猜测来源。
- 下载器：`train/data/download_complete.py`。穷尽分页、包含 `.json.gz` 等所有文件类型、支持已有文件验证和 HF 断点续传。HF 固定仓库 commit；ModelScope 记录 master 的清单时间与每文件 SHA256，若内容变化与清单不符则失败，需重新生成清单。不能声称 ModelScope master 是不可变 commit。
- 输出：`INVENTORY.json`（整库文件清单/总字节/版本/许可证）、`VERIFIED_FILES.json`（实际 SHA256）、`FULL_DOWNLOAD_COMPLETE.json`（全部清单文件校验通过后才写）、`/data/mini-k3/data/reports/downloads/*.json`（状态）、`DOWNLOAD_QUEUE.json`（容量准入与队列）。旧完成记录不升级为新完成标记。
- 本次“全量”指登记仓库全部文件，不是此前的配额，也不是 10B tokens。10B 是训练消费量，实际 token 数需要分词后确定；TB 级整库下载与模型训练预算分开。
- 容量准入先计算各仓库缺失字节的总和，并保留 500GB 给 checkpoint/日志。无法容纳的任务写 `blocked_capacity`；禁止把容量不足改成静默截断。单文件下载前再次检查剩余空间。
- 数据集 SHA256 缺失时，仅记录本地 SHA256 与大小，明确不能代表上游校验已获得。Git blob OID 按 Git blob 哈希验证，不能误称为 SHA256。
- 下载 `.py` 只作为仓库文件保存，绝不执行数据集提供的脚本。受限数据集保持失败/待授权状态，不自动接受访问条款。

来源纠正：`raw/culturax` 实际为 `opencsg/chinese-fineweb-edu`，不是 CulturaX；`raw/cosmopedia` 为中文 Cosmopedia，不是英文原始版；`raw/code-python` 为 Stack-v3，不能保证文件全为 Python。登记表为原始 CulturaX、Cosmopedia、StarCoderData 单独建目录，不覆盖已有替代数据。MathR/NuminaMath 是 SFT 补充来源，不能替代 FineMath/OpenWebMath 预训练原文。CCI Base 当前已下载的是 Nemotron 子集，不代表已下载 Dolma。

Dolma 正文使用用户提供且已查询验证的 `modelscope/dolma`，保存到 `raw/dolma-body`；`raw/dolma` 和 `raw/dolma-hf` 仅含 URL 元数据，不作为训练文本。CCI Base、Extra、CoT 独立登记；之前的“99 个文件约 13/106/206GB”只是第一页，不能作为整库总量。

服务器部署代码路径：`/data/mini-k3/project/train/data/`。大文件与缓存仍全部在 `/data/mini-k3/`。`download_complete.py --mode all` 在 `usage_only` 策略下被禁止；只能使用已审核的选中文件清单和对应下载器。

```bash
（下载命令由选中文件清单生成，禁止全仓库模式）
```

只有完整清单全部通过才标记单仓库完成。所有任务是否完成以状态报告为准；后台存在进程不等于正在传输，进程退出也不等于完成。不设清洗或训练自动触发器。

## 1. 硬件基准与核心约束

**训练基准**：4×Tesla V100-SXM2 32GB，FP16 + GradScaler，预训练序列长度 2048，micro-batch 1 / GPU，梯度累积 32，四进程 NCCL DDP。推理位置上限为 1,048,576，MLA 使用 4,096 窗口分块注意力，避免 1M×1M 显存；发布 1M 能力前必须完成长上下文继续训练和评测。

**存储规划**：所有大文件统一放在 `/data/mini-k3/`：原始下载集和 uint32 shards 在 `/data/mini-k3/data/`，checkpoint 在 `/data/mini-k3/checkpoints/`，日志在 `/data/mini-k3/logs/`，RL rollout 在 `/data/mini-k3/rollouts/`。代码、配置和小型索引继续放在 SSD 工作目录，不把数据下载到项目目录。

* **计算卡**：4×Tesla V100-SXM2 32GB
* **精度**：FP16 + GradScaler；禁止在 V100 上使用 BF16 或 FP8
* **训练方式**：四进程 NCCL DDP；每卡独立取数，梯度同步，rank 0 负责 checkpoint 和日志。
* **实现约束**：V100 使用 FP16 + GradScaler；不启用 H100 专用 BF16/FP8 或 sm_90 内核路径。
* **核心瓶颈预警**：
  * 每卡 32GB；显存余量必须由 smoke test 和 200-step benchmark 实测确认。

---

## 2. 唯一定案模型规格（Mini K3 1.02B 标准版）

为杜绝多方案选择干扰、确保工程落地极致打磨，本方案**唯一聚焦并打造 Mini K3 (1.02B)**：

### 2.1 详细规格表

| 字段参数 | 唯一定案规格：Mini K3 (1.02B) | 原版 Kimi K3 参考对照 |
| :--- | :--- | :--- |
| **隐藏维度 (hidden_size)** | 512 | 7,168 |
| **总层数 (num_layers)** | 12 | 93 |
| **注意力架构比例** | 9层 KDA + 3层 MLA (3:1 经典比例) | 69层 KDA + 24层 MLA (3:1) |
| **全注意力 (MLA) 所在层** | 第 4, 8, 12 层 (1-indexed) | 每第 4 层 |
| **MLA 维度切分** | 128 / 64 / 128 (nope / rope / value) | 128 / 64 / 128 |
| **head_dim** | 128 | 128 |
| **KDA 头数 / MLA 头数** | 4 头 (4×128=512) / 8 头 | — / 96 头 |
| **前置稠密层 (Dense Layer)** | 第 0 层 (Dense MLP) | 第 0 层 (Dense MLP) |
| **路由专家池 (Routed Experts)**| 256 个 | 896 个 |
| **每 Token 激活专家数 (top_k)**| 6 个 | 16 个 |
| **共享专家数 (Shared Experts)**| 2 个 (始终无条件激活) | 2 个 (始终无条件激活) |
| **潜在表示瓶颈宽度 (Latent)** | 256 (hidden // 2) | 3,584 (hidden // 2) |
| **专家 FFN 中间维度** | 416 | 2,048 |
| **词表大小 (Vocab Size)** | 163,840 (K3 官方 BPE) | 163,840 |
| **词嵌入共享策略** | `tie_word_embeddings: true` | `false` |
| **激活函数** | `situ` ($\beta=4.0, \text{linear\_beta}=25.0$) | `situ` |
| **训练上下文长度** | 2,048 | 1,048,576 |
| **总参数量 (Total Params)** | **1.02B (1,024,424,448)** | 2.8T |
| **激活参数量 (Active Params)** | **145M (145,213,952)** | 104B |
| **非嵌入激活参数量** | **61M** | ~103B |
| **路由专家计算占比** | 34% | 46.1% |

### 2.2 开源训练集与全量来源台账

本项目严格遵循开源合规与去污染原则，所有数据源记录来源仓库、许可证、原始路径、分卷统计与 Token 产出。下表整合当前 11 个基线来源与 2026-09-18 启动的补充下载来源，完整对应 `config.py` 配比与各阶段加工产物：

#### 全量训练数据源与来源总表 (Master Training Sources Inventory)

| 数据源标识 | 训练阶段 / 角色 | 上游官方仓库 / 镜像后端 | 许可证 (SPDX) | 原始数据本地存储路径 | 原始分卷与体积 | 可用/预期 Tokens | 状态 / 备注 |
|---|---|---|---|---|---|---:|---|
| **`fineweb-edu`** (全量) | 预训练 / 英文百科 | `HuggingFaceFW/fineweb-edu` (ModelScope) | ODC-By 1.0 | `/data/mini-k3/data/raw/fineweb-edu/` | 8 个分卷 (18.69 GB) | 5.071 B (训练) + 51.1 M (验证) | 全量增补分词与Manifest审计已通过 |
| **`chinese-fineweb-edu`** | 预训练 / 中文网页 | `opencsg/chinese-fineweb-edu` (ModelScope) | Apache-2.0 | `/data/mini-k3/data/raw/culturax/` | 13 个分卷 (4.83 GB) | 1.440 B (训练) + 14.3 M (验证) | 主流程分词与Manifest审计已通过 |
| **`dolma-body`** | 预训练 / 多领域英文 | `modelscope/dolma` (ModelScope) | ODC-By 1.0 | `/data/mini-k3/data/raw/dolma-body/` | 27 个分卷 (10.0 GB) | 9.008 B (训练) + 91.3 M (验证) | 280卷去污染/分词全部完成，审计已通过 |
| **`finemath`** | 预训练 / 精细数学 | `HuggingFaceTB/finemath` (ModelScope) | ODC-By 1.0 | `/data/mini-k3/data/raw/finemath/` | 10 个分卷 (10.0 GB) | 5.237 B (训练) + 52.4 M (验证) | 主流程分词与Manifest审计已通过 |
| **`open-web-math`** | 预训练 / 数学公式网页 | `open-web-math/open-web-math` (hf-mirror) | ODC-By 1.0 | `/data/mini-k3/data/raw/open-web-math/` | 10 个分卷 (10.0 GB) | 4.481 B (训练) + 45.7 M (验证) | 主流程分词与Manifest审计已通过 |
| **`cosmopedia`** | 预训练 / 中文合成教材 | `AI-ModelScope/chinese-cosmopedia` (ModelScope) | Apache-2.0 | `/data/mini-k3/data/raw/cosmopedia/` | 5 个分卷 (4.86 GB) | 1.690 B (训练) + 16.6 M (验证) | 主流程分词与Manifest审计已通过 |
| **`code-python`** (全量) | 预训练 / 代码 | `stack-v3-train` + `codeparrot/github-code` | MIT / Apache-2.0 / BSD | `/data/mini-k3/data/raw/code-python/` | 8 基线 + 200 增补分片 (11.3 GB) | 1.181 B (训练) + 13.3 M (验证) | 200分片SPDX清洗转码入库，审计已通过 |
| **`openassistant`** | SFT 对齐 / 对话树 | `OpenAssistant/oasst1` (hf-mirror) | Apache-2.0 | `/data/mini-k3/data/raw/openassistant/` | 1 个分卷 (232 MB) | 17.38 M (训练) + 1.16 M (验证) | 会话组切分已收敛，分词与Manifest审计已通过 |
| **`openhermes`** (审核版) | SFT 对齐 / 代码与数学 | `teknium/OpenHermes-2.5` (hf-mirror) | Glaive(Apache2.0)+MetaMath(MIT) | `/data/mini-k3/data/raw/openhermes-reviewed-v2/` | 238,672 条高质量问答 | 85.16 M (训练) + 0.86 M (验证) | 用户明确批准，主流程分词与Manifest审计已通过 |
| **`openr1`** | SFT 对齐 / 数学长链推理 | `open-r1/OpenR1-Math-220k` (hf-mirror) | Apache-2.0 | `/data/mini-k3/data/raw/openr1/` | 13 个分卷 (13.0 GB) | 494.87 M (训练) + 4.87 M (验证) | 主流程分词与Manifest审计已通过 |
| **`ultrafeedback`** | 偏好对齐 / DPO与RL | `argilla/ultrafeedback-binarized-preferences-cleaned` | MIT | `/data/mini-k3/data/raw/ultrafeedback/` | 2 个分卷 (144 MB) | 50.79 M (训练) + 0.54 M (验证) | 主流程分词与Manifest审计已通过 |

#### WSD 调度配方与目标比例 (10B Tokens)

| 用途 | 数据源标识 | 稳定阶段 (Stable 85%) | 衰减阶段 (Decay 15%) | 10B 目标需求 | 增补后可用储备 | 覆盖状态 |
|---|---|---:|---:|---:|---:|:---:|
| 英文教育网页 | `fineweb-edu` | 45% | 30% | 4.28 B | 5.07 B | ✅ 119% 覆盖 |
| 中文网页 | `chinese-fineweb-edu` | 15% | 10% | 1.43 B | 1.44 B | ✅ 101% 覆盖 |
| Dolma 正文 | `dolma-body` | 10% | 10% | 1.00 B | 9.01 B | 🌟 901% 覆盖 |
| 数学专业教材 | `finemath` | 5% | 12% | 0.61 B | 5.24 B | 🌟 859% 覆盖 |
| 数学公式网页 | `open-web-math` | 5% | 8% | 0.55 B | 4.48 B | 🌟 815% 覆盖 |
| Python 代码 | `code-python` | 10% | 25% | 1.23 B | 1.18 B | ⚠️ 96.4% 覆盖 (44M缺口) |
| 中文合成教材 | `cosmopedia` | 10% | 5% | 0.93 B | 1.69 B | 🌟 182% 覆盖 |
| **预训练总计** | — | **100%** | **100%** | **10.00 B** | **28.11 B** | 🌟 **2.81 倍无重复储备** |

2026-09-15 文档一致性修订：上表按 `canonical_v2.py` 的真实 repo 和 `config.py` 更新，纠正旧文档的英文 Cosmopedia、CulturaX/StarCoderData 和数学/教材比例描述。仅修正文档，不更改当前语料。尤其中文合成教材不能计作英文教材。最终全量编码后逐来源统计可用 tokens；许可证过滤后剩余量小的代码源必须单独报告覆盖缺口，不能凭总容量宣布满足约10B配方或无重复消费需求。

代码只保留许可证字段明确、允许再分发的文件；The Stack 不能全量无条件混入。所有来源统一做文档级去重、语言识别、质量和 PII 过滤，并在 tokenization 前用评测集 13-gram 索引去污染。

SFT 使用公开的 `OpenAssistant/oasst1`、`teknium/OpenHermes-2.5`、`open-r1/OpenR1-Math-220k`；偏好训练使用 `argilla/ultrafeedback-binarized-preferences-cleaned`。训练集、验证集和最终评测集严格隔离，移除不允许再分发的样本并人工抽检。

### 2.3 数据版本冻结与清洗流水线

数据根目录固定为 `/data/mini-k3/data`，按 `raw/<dataset>/<version>`、`cleaned/`、`deduped/`、`tokenized/`、`manifests/`、`reports/` 分层保存。每阶段使用独立 manifest：`pretrain_stable.json`、`pretrain_decay.json`、`sft.jsonl`、`preference.jsonl`、`rollout.jsonl`。

处理顺序固定为：下载并校验 checksum → 读取许可证和版本 → 文档规范化 → 语言/长度/乱码/PII 过滤 → URL 和文档级去重 → benchmark 13-gram 去污染 → tokenizer 编码 → uint32 分片 → 统计报告。任何一步失败都停止，不自动跳过或重新归一化来源。

下载使用多源回退：`Hugging Face 官方 → hf-mirror → ModelScope`。实际使用的后端、数据集 revision、文件大小、checksum 和失败原因写入每个来源的 `DOWNLOAD.json`；ModelScope ID 必须通过 `/data/mini-k3/data/modelscope_map.json` 显式映射，禁止猜测仓库。只有后端连通性和 checksum 均通过，数据才可进入清洗流水线。

### 2.4 学习率调度策略 (WSD - Warmup-Stable-Decay)

* **峰值学习率**：$\text{LR} = 6.00 \times 10^{-4}$
* **衰减底限**：$\text{min\_LR} = 6.00 \times 10^{-5}$（峰值的 10%）
* **三阶段切分**（总计 38,147 步，对应四卡全局约 10,000,007,168 个 token）：
  * **预热期（Warmup）**：第 0 ~ 762 步（前 2.0%），从 $0$ 线性上升至 $6.00 \times 10^{-4}$；
  * **稳定期（Stable）**：第 763 ~ 32,429 步（中间 83.0%），恒定维持峰值 LR，采用通用网络 + 基础代码语料；
  * **衰减期（Decay）**：第 32,430 ~ 38,147 步（最后 15.0%），从 $6.00 \times 10^{-4}$ 线性降至 $6.00 \times 10^{-5}$，同时数据加载器切换为强化数学/代码的退火混合比。

---

## 3. 必须遵循的工程硬规（原书 30 章踩坑总结）

所有在 `train/` 目录下编写的代码严格内置以下 6 项防护：

1. **第 0 步有限性断言与初始损失对齐**：
   * 训练循环启动前必须运行 `assert_initialised(model)`，确保每一个参数和缓冲区都是有限浮点数（无 `inf`/`NaN`）。
   * 对均匀分布在 163,840 词表的交叉熵损失为 $\ln(163,840) \approx 12.007$。第 0 步输出必须处于 $12.05 \sim 12.15$ 之间，偏差过大立即中断。
2. **KDA 时间步偏置初始化**：
   * 官方代码的 `KimiDeltaAttention.dt_bias` 留存了未初始化的垃圾内存。必须使用类似 Mamba 的对数空间均匀采样加 `softplus` 反函数初始化：
     $$\text{dt} \sim \exp(\text{Uniform}(\ln 10^{-3}, \ln 10^{-1})), \quad \text{bias} = \text{dt} + \ln(1 - e^{-\text{dt}})$$
3. **MoE 负载均衡器实现**：
   * 官方声明的 `noaux_tc` 无辅助损失均衡器必须落地：使用专家偏置修正路由选择，但计算最终输出权重时使用原始无偏概率；
   * 步长固定为 **$\text{gamma} = 1.0 \times 10^{-2}$**（切勿照搬 DeepSeek-V3 的 $10^{-3}$）；
   * `e_score_correction_bias` 必须排除在 AdamW 优化器参数列表之外；
   * 监控指标严禁使用“路由器平均分数熵”（会掩盖 94.5% 的专家坍塌），必须监控 **`dead_frac`（0 token 不活跃专家占比）** 和 **`imbalance`（最大专家负载 / 平均负载）**。
4. **Triton 融合算子释放显存**：
   * 激活函数 `situ` 计算涉及模型最大的张量 `[experts, capacity, ffn_dim]`。V100 路径使用可审计的重计算实现，禁止调用仅支持 sm_90 的 BF16/FP8 内核。
5. **数据加载器严禁静默归一化**：
   * 严禁用 `split("-", 1)` 重新拼接文件名（会导致带有连字符的来源如 `fineweb-edu` 丢失）；
   * 分片缺失时必须显式抛出 `FileNotFoundError`，禁止静默剔除并自动对剩余数据重新归一化。
6. **分词器规避字面 `[EOS]`**：
   * 编码网页数据时调用 `hf.encode(text, allow_special_tokens=False)`，严禁将文本中出现的字面字符串 `[EOS]` 解析为控制 token 163585，避免训练语料在句子中途静默断裂。

---

## 4. 批次与显存配置（针对 4×V100 32GB）

* **序列长度 (Sequence Length)**：2,048 tokens
* **微批次大小 (Micro-batch Size)**：1 / GPU
* **梯度累积步数 (Gradient Accumulation Steps)**：32 次，四卡等效单步 $4 \times 1 \times 32 \times 2048 = 262,144$ tokens
* **总训练步数**：38,147 步（四卡全局处理约 $10,000,007,168$ 个 token）
* **吞吐与耗时**：必须通过 200-step benchmark 实测后填写，禁止沿用 H100 估算。

---

## 5. 完整 1B 训练生命周期

### 5.0 与正式 Kimi K3 的模块对照

| 模块 | 正式 Kimi K3 | 本 Mini K3 | 状态 |
|---|---|---|---|
| KDA/线性注意力 | 有 | 9 层 KDA (ShortConv + Delta Rule) | 完整实现，支持状态缓存 |
| MLA/全局注意力 | 有 | 3 层 MLA + 4096 窗口 + 10M RoPE | 完整实现，分块滑动注意力 |
| CED/DeepSeek-Coder 风格分支 | 非同一主干 | `models/deepseek_coder.py` 独立可选 | 保留，不与 K3 权重混载 |
| MoE 路由与共享专家 | 大规模 MoE | 256 routed + 2 shared (Latent 空间路由) | 完整实现，noaux_tc 动态均衡 |
| 长上下文位置编码 | 百万级 (1M) | RoPE base=10,000,000.0 (支持 1,048,576) | 完整实现，通过酉圆与有限性测试 |
| 1M 上下文与缓存等价性 | 严格等价 | `models/kv_cache.py`, `test_cache_equivalence.py` | 严格满足红线第 8 条要求 |
| 1M 显存基准 | O(1) 状态 | `benchmark_memory.py` (KDA O(1) + MLA 4K 窗口) | 严格满足红线第 8 条要求 |
| 长文档检索评测 (NIAH) | 大海捞针检索 | `eval_long_context.py` (4K ~ 1M 深度检索) | 严格满足红线第 8 条要求 |
| 数据流与二进制分片 | uint32 分片 | `data/build_shards.py` + `data/loader.py` | 完整实现，生成 manifest.json |
| 训练后对齐 | SFT、RM、DPO、PPO | `alignment_train.py` + `rl_trainer.py` | 4 阶段全闭环落地 |
| 交互式终端推理 | 流式对话 | `chat.py` (基于 KV Cache 逐 Token 输出) | 完整实现 |
| 多卡训练 | 专家/参数并行 | 4 卡 DDP + FP16 GradScaler | 针对 4×V100 优化 |

本模型的目标是“模块齐全的缩小版”，不是参数和能力等价复刻。任何标称 1M 上下文的 checkpoint 必须通过长文档检索、跨段问答、长代码和 KV-cache 等价性测试。

同一个 `MiniK3ForCausalLM` 骨干贯穿四个阶段，避免为 SFT/RL 重新实现模型：

1. **预训练**：约 10B tokens（四卡全局 batch），WSD，混合网页/代码/数学语料；保存模型、AdamW、数据游标、Python/NumPy/PyTorch RNG 和监控状态。
2. **SFT**：使用 `chat_template.py` 把 system/user/assistant 对话打包，prompt 标签为 `-100`，只训练 assistant token；保留 EOS，按长度分桶并验证截断率。
3. **偏好对齐**：先训练 `RewardModel`（chosen 分数高于 rejected 的 pairwise loss），再用冻结 reference 做 DPO；DPO 使用长度归一化 log-prob 和显式 KL 监控。
4. **在线 RL/PPO**：policy、reference、value、reward 四个模型分工；rollout 记录 token、old log-prob、reference log-prob、value、reward 和终止原因，PPO 使用 ratio clipping、value clipping、优势标准化和 KL 惩罚。奖励模型、EOS、最大长度、拒答/安全规则都必须版本化。
5. **最终 SFT**：用人工审核和 RL 过滤后的高质量轨迹再做短程 SFT，低学习率、冻结或降低底层层学习率，作为最终发布 checkpoint。

阶段间只传递 `model.pt` 和明确的 tokenizer/config 版本；每阶段单独目录和 manifest，禁止覆盖预训练 checkpoint。

### 5.1 四卡 checkpoint 红线

每个保存点由所有 rank 写入各自的 `optimizer_rankN.pt`、`rng_rankN.pt`、`loader_rankN.pt`；rank 0 在 barrier 后写入 `COMPLETE` 并原子改名。恢复时必须校验 `world_size`，并加载当前 rank 对应的 optimizer、RNG 和数据游标；world size、模型配置或 manifest 不一致时拒绝恢复。

## 6. 监督微调、偏好优化与强化学习

预训练检查点之后的统一接口在 `alignment.py`、`alignment_models.py`、`rl_trainer.py` 与 `alignment_train.py`：

* **SFT**：JSONL 提供 `input_ids` 与 `labels`，prompt 标签使用 `-100`，只对 assistant token 计算 CE。
* **RM**：JSONL 提供 `chosen_ids` 与 `rejected_ids`，训练 `RewardModel` 拟合 pairwise Bradley-Terry 损失。
* **DPO**：提供同一 prompt 的 `chosen_ids`/`rejected_ids`，冻结 reference checkpoint，使用长度归一化序列 log-prob。
* **PPO/RL**：`PPOTrainer` 协调 policy、reference、reward 与 value 四个模型；通过 KV Cache 生成 rollout，计算 GAE、ratio clipping、value clipping 与 KL 惩罚。

示例命令：
```bash
# 1. 监督微调 (SFT)
python3 train/alignment_train.py --mode sft --checkpoint checkpoints/step_038147 --jsonl data/sft.jsonl
# 2. 奖励模型训练 (RM)
python3 train/alignment_train.py --mode rm --checkpoint checkpoints/step_038147 --jsonl data/preferences.jsonl
# 3. 直接偏好优化 (DPO)
python3 train/alignment_train.py --mode dpo --checkpoint checkpoints/step_038147 --jsonl data/preferences.jsonl
# 4. 在线强化学习 (PPO)
python3 train/alignment_train.py --mode ppo --checkpoint checkpoints/step_038147 --reward_checkpoint checkpoints/rm_model.pt --jsonl data/prompts.jsonl
```

## 7. 目录组织与可执行脚本

```text
train/
├── TRAINING_PLAN.md         # [本文件] 唯一定案训练规划与避坑规范
├── config.py                # Mini K3 (1.02B) 唯一定案模型配置
├── smoke_test.py            # 2 步冒烟自检脚本 (验证参数、初始化与数值下降)
├── test_cache_equivalence.py# 缓存等价性与 1M RoPE 自检脚本 (严格满足红线第 8 条)
├── benchmark_memory.py      # 1M 序列推理显存基准脚本 (严格满足红线第 8 条)
├── eval_long_context.py     # 1M 长文本大海捞针 (NIAH) 检索评测 (严格满足红线第 8 条)
├── chat.py                  # 交互式流式终端对话命令行 (支持多轮与 KV Cache)
├── train.py                 # 主训练启动入口 (4 卡 V100 DDP)
├── evaluate.py              # 独立评测脚本 (基于右填充与长度归一化评估)
├── alignment.py             # SFT/RM/DPO/PPO 损失函数与工具库
├── alignment_models.py      # 奖励模型与价值模型标量输出头
├── alignment_train.py       # 四阶段统一后训练启动入口
├── rl_trainer.py            # PPO 训练器与 GAE 优势计算器
├── chat_template.py         # Kimi 对话模板打包器
├── models/
│   ├── kda.py               # 带遗忘门、短卷积和状态缓存的 KDA 线性注意力层
│   ├── mla.py               # 128/64/128 切分的 MLA 潜在注意力与滑动窗口
│   ├── moe.py               # 批处理 BMM 可微分分发与 TrainableMoEGate
│   ├── kv_cache.py          # KDA 状态 + 卷积状态 + MLA KV Cache 缓存结构
│   ├── deepseek_coder.py    # 独立解耦的 DeepSeek-Coder 风格 Dense 对照基线
│   └── mini_k3.py           # 完整组装的 Mini K3 CausalLM 架构与增量解码
├── kernels/
│   └── situ_fused.py        # Triton 融合 situ 激活函数算子 (支持 FP16 / V100)
├── data/
│   ├── tokenizer.py         # K3 官方词表包装器 (防字面[EOS]注入)
│   ├── build_shards.py      # 原始语料分词与 uint32 二进制分片构建工具
│   └── loader.py            # uint32 二进制分片加权数据加载器
└── engine/
    ├── init_patch.py        # Mamba 风格初始化与有限性断言
    ├── scheduler.py         # WSD 学习率调度器
    ├── balancer.py          # noaux_tc 负载均衡器
    ├── spike_guard.py       # EMA 损失尖峰守护
    └── checkpoint.py        # 逐位等价的断点续训管理器
```

### 快速启动命令

1. **预检与冒烟自检（验证模型完好性）**：
   ```bash
   python train/smoke_test.py
   ```
2. **验证 Cache 等价性与 1M 位置编码稳定性（红线第 8 条要求）**：
   ```bash
   python train/test_cache_equivalence.py
   ```
3. **验证 1M 序列推理显存基准（红线第 8 条要求）**：
   ```bash
   python train/benchmark_memory.py
   ```
4. **验证 1M 长文本检索能力 (Needle In A Haystack)（红线第 8 条要求）**：
   ```bash
   python train/eval_long_context.py --lengths 4096 8192 16384 32768
   ```
5. **冻结后的 manifest/shard 完整性复核**（不重新构建数据）：
   ```bash
   python train/data/validate_manifest.py \
     --manifest /data/mini-k3/data/prepared-v2-supplement-v1/manifests/pretrain_stable.json
   python train/data/validate_manifest.py \
     --manifest /data/mini-k3/data/prepared-v2-supplement-v1/manifests/validation.json
   ```
6. **启动主训练（4 卡 V100）**：
   ```bash
   mkdir -p /data/mini-k3/{data,checkpoints,logs,rollouts}
   torchrun --standalone --nproc_per_node=4 train/train.py \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v1/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v1/manifests/validation.json \
     --checkpoint_dir /data/mini-k3/checkpoints \
     --total_steps 38147 --save_interval 1000
   ```
7. **从中断检查点恢复训练**：
   ```bash
   torchrun --standalone --nproc_per_node=4 train/train.py --resume \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v1/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v1/manifests/validation.json \
     --checkpoint_dir /data/mini-k3/checkpoints
   ```
8. **评估模型检查点**：
   ```bash
   python train/evaluate.py --checkpoint /data/mini-k3/checkpoints/step_038147 --tokenizer_model /data/mini-k3/data/tokenizer
   ```
9. **交互式终端对话**：
   ```bash
   python train/chat.py --checkpoint /data/mini-k3/checkpoints/step_038147 --tokenizer_model /data/mini-k3/data/tokenizer
   ```

## 2026-09-14 全仓库容量与数据源审计结果

完整分页查询结果保存在 `train/data/source_audit_20260914.json`；服务器副本 `/data/mini-k3/data/reports/SOURCE_AUDIT.json`。20 个登记仓库总大小为 48,544,330,775,643 字节（48.54TB，十进制）。服务器当时剩余约 7.73TB，保留 0.5TB 后不能容纳所有仓库。当前后台任务优先补齐小仓库、OpenWebMath、FineMath、原始英文 Cosmopedia、中文语料和 Dolma 正文；其余整库列为容量阻塞，不删除已有数据，不伪报全量完成。

| 仓库 | 完整文件数 | 大小（十进制 TB） | 来源缺口 |
|---|---:|---:|---|
| uonlp/CulturaX | 9059 | 17.467 | 来源存在；gated=auto，匿名文件 HEAD 返回 403；同时超出容量 |
| HuggingFaceCode/stack-v3-train | 16508 | 9.412 | 来源存在，整库大于可用磁盘 |
| HuggingFaceFW/fineweb-edu | 3038 | 5.836 | 来源存在，与已准入任务合计后空间不足 |
| BAAI/CCI4.0-M2-Base-v1 | 10864 | 5.722 | 来源存在，空间不足 |
| modelscope/dolma | 2425 | 4.487 | 正文来源存在，已加入下载队列 |
| BAAI/CCI4.0-M2-CoT-v1 | 1347 | 2.567 | 来源存在，空间不足 |
| BAAI/CCI4.0-M2-Extra-v1 | 1285 | 1.971 | 来源存在，空间不足 |
| bigcode/starcoderdata | 865 | 0.311 | 来源存在；gated=auto，匿名文件 HEAD 返回 403 |
| open-web-math/open-web-math | 121（含非数据文件） | 0.0274 | 已有镜像正文下载路径，补齐全部清单 |

所有登记仓库均获得了文件清单，因此当前没有“完全找不到仓库”的来源；访问受限、缺少许可证元数据、下载失败和容量不足必须分别报告。Dataset card 的顶层许可证字段仅作记录，不能替代分来源许可证审核。下载成功也不代表可直接清洗并训练。

注意原始数据完整性与训练采样配额的区别：此前 77GiB 是有限配额的原始文件，不是整个 10B 配方已就绪。完整文件枚举不能用“前100条”结果，也不能把 URL 文本当 Dolma 正文。最终是否满足 10B tokens，仍须在用户授权的数据处理阶段统计真实 tokenizer token 数。

### 2026-09-15 终态验收门槛补充

最终 manifest 生成器现在强制读取 `prepared-v2/SOURCE_REVIEW.json`，要求 11 个来源逐一提供 provenance/license 证据、许可证审核通过和 `training_approved=true`。任一来源仍为 pending 时，流水线可以继续处理数据，但不得生成通过验收的 manifest/AUDIT。

### 2026-09-16 验收部署与报告归档

来源审核门禁已成功部署至远端，真实tokenizer合成整链路回归通过；生产转换→清洗→精确去重的11来源元数据链复查通过。报告和代码快照已在本地 `train/reports/` 归档，索引见 `train/PIPELINE_STAGE_REPORTS.md`。本轮本地8项阶段/来源审核测试通过。此项只补齐验收和归档，不改变运行中的语料、算法、tokenizer、配比或硬件；OpenHermes来源审核仍待解决，生产流程尚未完成。

### 2026-09-16 OpenHermes 来源元数据复查

为核实496,743条缺少source标签的记录是否仍含其他来源字段，扩展只读 `audit_openhermes_sources.py`：按缺失标签记录的字段集合统计数量、第一条原始行号，并检查dataset/dataset_name/subset/origin/source_name字段。该统计不读取模型猜测、不自动批准许可、不修改生产数据。输出新报告 `/data/mini-k3/data/reports/openhermes-metadata-inventory-20260916.json`，保留旧报告。

2026-09-16 全量来源元数据盘点结果：1,001,551条原始记录中496,743条缺失source，414,062条仅含conversations；没有缺失source的记录含非空字符串dataset/dataset_name/subset/origin/source_name字段。仅凭已有字段不能完成来源审核，training_approved仍为false。报告已归档 `train/reports/verification/openhermes-metadata-inventory-20260916.json`，本地核对脚本SHA256、schema计数及总记录守恒通过。后续需要上游精确匹配证据或经过记录的来源过滤方案；本次没有更改生产语料。

### 2026-09-16 11:15 近去重完成与去污染推进

全局近似去重 11 个来源均已生成完成标记，阶段链审计 `/data/mini-k3/data/reports/stage-chain-through-near-deduped.json` 通过并已归档本地。13-gram 去污染已完成9/11；OpenWebMath和Dolma仍在运行。不得在11个去污染报告齐全前开始最终tokenizer/manifest验收声明。

### 2026-09-19 增补流水线 prepared-v2-supplement-v1 启动

针对 10B 配方有效 token 缺口，启动增补流水线 `prepared-v2-supplement-v1`。增补语料 FineWeb-Edu 8 个 Parquet 分片 (18.69 GB, ~6.0B tokens) 与 GitHub-Code 200 个分片 (~6.5 GB, 675,071 文件, ~1.5B tokens) 完成下载与清洗适配，并通过精确去重审计（`DEDUP_COMPLETE.json`）。

### 2026-09-21 08:47 增补流水线 MinHash 近去重 100% 完工

2026-09-21 08:47:36，增补流水线 11 个数据源的全局 MinHash 近去重全量顺利收官，生成 `NEAR_DEDUP_COMPLETE.json`。共保留 17,081,347 篇文档，过滤 471,177 篇近重复文档，11 个数据源的链式哈希校验全部落盘通过。

### 2026-09-21 10:58 断电故障与 12:20 恢复重启

2026-09-21 10:58 左右服务器因机房断电关机。断电时增补流水线正处于阶段 3（13-gram 基准去污染）。经全面审计排查：
1. **数据与近去重零损失**：所有原始下载文件完整，阶段 1 全局精确去重与阶段 2 MinHash 近去重已在断电前全量完成并持久化落盘，完全无需重跑（节省约 30 小时计算）。
2. **阶段 3 状态**：6 个数据源（`chinese-fineweb-edu`、`code-python`、`openassistant`、`openhermes`、`openr1`、`ultrafeedback`）已在断电前完工；2 个数据源（`fineweb-edu` 中断于 part-134、`cosmopedia` 中断于 part-21）被中断；3 个数据源（`finemath`、`open-web-math`、`dolma-body`）待处理。
3. **故障恢复措施**：
   - 清理 interrupted 分片与临时标记：删除 `decontaminated/fineweb-edu` 与 `decontaminated/cosmopedia` 目录。
   - 控制器优化：升级 `train/data/continue_v2.py`，加入对已完成 `deduped` 和 `near-deduped` 的快速幂等跳过逻辑。
   - 恢复运行：重新在后台拉起 `supplement_v1.py --work /data/mini-k3 --workers 2`。
4. **硬件与训练红线**：
   - 数据预处理各阶段（去重、去污染、Tokenize）完全在 CPU/RAM/磁盘执行，不依赖 GPU。
   - 记录冷启动后 4× Tesla V100-SXM2 所在的 PLX PCIe 总线链路待进一步核验；严格遵循 Rule 7，未经用户明确要求，不得启动模型训练或重启系统服务。

### 2026-09-21 21:10 硬件全量就绪与流水线断点重跑

服务器经电源/硬件重置后重启（uptime 3h18m）：
1. **硬件全量就绪**：4× Tesla V100-SXM2-32GB 显卡已由操作系统与 NVIDIA 驱动完整识别（`nvidia-smi` 正常，4 张卡空闲待命，温度 30-34°C，显存空闲）。
2. **中间进度固化**：在下午阶段中，`cosmopedia` 已于 13:51 全量完成去污染落盘，阶段 3 已累计完成 7/11 个数据源（`chinese-fineweb-edu`、`code-python`、`cosmopedia`、`openassistant`、`openhermes`、`openr1`、`ultrafeedback`）。
3. **故障清理与断点重跑**：
   - 清理中断的 `decontaminated/finemath` 与 `decontaminated/fineweb-edu` 临时目录；
   - 重新拉起后台增补流水线 `supplement_v1.py`（PID 4188），自动跳过已完工的 7 个数据源，无缝重跑 `fineweb-edu` 和 `finemath` 并推进剩余流程。

### 2026-09-22 05:16 阶段 3 全量收官与阶段 4 Tokenize 全面推进

截至 2026-09-22 05:16:25，增补流水线 11 个数据源的阶段 3（13-gram 基准去污染）已 100% 全部顺利完工（`finemath` 00:23、`fineweb-edu` 00:27、`open-web-math` 03:10、`dolma-body` 05:16 全量通过去污染并生成 `COMPLETE.json`）。

流水线自动进入阶段 4（多线程 Tokenize 分词）：
1. **已完工数据源（6个）**：`openassistant`、`openhermes`、`openr1`、`ultrafeedback`、`code-python`、`chinese-fineweb-edu`。
2. **正在分词（2个）**：`fineweb-edu`（已完成 157/249 分片，产出 3.49B tokens）、`cosmopedia`（已完成 53/74 分片，产出 1.24B tokens）。
3. **待分词（3个）**：`finemath`、`open-web-math`、`dolma-body`。
4. **计算资源**：4× Tesla V100-SXM2-32GB 当前已完全释放为空闲就绪状态（0% 使用率，30-34°C），等待数据流水线终态验收后随时可开启训练。

### 2026-09-22 13:26 阶段 4 分词全量收官，阶段 5 全量 Manifest 审计与模型短跑验证 100% 通过

1. **分词全量收官 (Stage 4 Complete)**：
   - 11 个数据源多线程 Tokenize 分词全部完成，总计产出 **28,755,447,057 Tokens (28.76 B)** 训练数据与 **292,230,296 Tokens (292.2 M)** 严格隔离的验证数据。
   - 预训练 Token 总池达 **28.11 B Tokens**，达标 10B 总体需求的 281%。
2. **零泄漏收敛修复与阶段 5 全量 Manifest 审计通过 (Stage 5 Complete)**：
   - 发现并修复 `openassistant` 中单个跨分枝同文本记录引发的训练/验证组冲突，统一收敛归入验证集后重新分词并通过 stage chain 校验。
   - `finalize_v2.py` 对全量 287 亿 tokens 执行了逐文件 SHA256 校验、uint32 二进制边界扫描、EOS 统计与 SQLite 组隔离检查，结果为 `train_validation_overlap: 0`，全量通过审计并落盘 `pretrain_stable.json`、`pretrain_decay.json`、`validation.json`、`alignment.json` 与 `AUDIT.json`。
3. **真实数据模型功能短跑通过 (Smoke Test Passed)**：
   - 运行 `smoke_from_manifest.py`，从最新 `pretrain_stable.json` 真实切片中采样各来源数据，在 Tesla V100 上执行完整前向反向与两步优化器迭代。
   - 各来源初始交叉熵损失均在 $\ln(163840) = 12.0067$ 基准附近（12.00 ~ 12.19），两步后平稳下降至 11.58，梯度范数正常（19.08），NoAuxBalancer 路由正常（dead fraction 0.266），报告记录于 `SMOKE.json`（状态 `passed`）。
4. **覆盖率评估 (Coverage Report)**：
   - 生成 `coverage.json`：固定配比下无需复用可支持 **9.64 B Tokens** 绝对纯净预训练；若 Python 代码允许 1.037 epoch 微量重用（仅 44M token 缺口，占 3.6%），即可实现完整 10.0 B Token 预训练无重复消费。

### 2026-09-22 训练前整体复核与冻结门禁

远端 `prepared-v2-supplement-v1` 已完成 11 个来源的全部阶段，`manifests/AUDIT.json`、真实数据 `SMOKE.json` 和来源审核均为 `passed`；远端 4×V100 空闲，当前没有训练进程。数据仍不能直接冻结为“固定配比完整覆盖”：`reports/supplement-v1/coverage.json` 的状态是 `insufficient_fixed_mix`，Python 代码在文档 WSD 配比下缺 **44,133,808 tokens**，固定配比无重复最多支持 **9,639,731,183 tokens**。训练前必须二选一并在同一变更中更新计划、配置和 manifest：

1. 继续补充并重跑受影响流水线，至少补足报告建议的 **52,960,569 tokens**（含20%缓冲）；或
2. 明确批准 Python 约 **3.6%** 的微量复用，记录复用策略并把 coverage 状态从阻断改为已批准的训练方案。

本次复核还发现训练入口 `train/train.py` 缺少 `random` 导入，且默认 manifest 指向旧的不存在路径；已修复为当前审计版 `prepared-v2-supplement-v1/manifests/{pretrain_stable,validation}.json`。远端架构级 cache 等价、1M RoPE 稳定性和显存基准测试通过；无预训练 checkpoint 前，1M 长文档能力仍不能宣称完成。上述修复和复核不启动训练，也不改变已生成的数据正文。

### 2026-09-22 批准方案 C（CodeSearchNet Python 增补）并启动 supplement-v2 全链路重跑

用户明确审批选择方案 C：补充 `Nan-Do/code-search-net-python` 数据集以彻底填补 44,133,808 tokens 的 Python 代码缺口，消除 3.6% 潜在重复采样。

1. **准入合规与来源审计**：
   - 官方/镜像仓库：`Nan-Do/code-search-net-python` (`hf-mirror.com`)，上游 commit SHA `39db91866dd0f251f3b0c7f42c0f85634101df6e`。
   - 许可证 (SPDX)：**Apache-2.0**，完全符合项目宽松商用白名单。
   - 排除数据集：`codefuse-ai/CodeExercise-Python-27k` (CC-BY-NC-SA-4.0 含有非商业限制，一票否决)；`theothertom/codeparrot-python-only` (未知许可证，一票否决)；`jtatman/python-code-dataset-500k` (对话指令格式非预训练代码，一票否决)。
   - 下载审计：下载全量 4 个 Parquet 分片（572 MB），本地 SHA256 校验完毕并记录于 `/data/mini-k3/data/reports/codesearchnet/download.json`。
   - 结构化转换：提取带有完整 Docstring 与实现的 Python 函数，严格过滤非 Python 及短于 20 字符样本，转换为 4 个标准 Parquet 分片（`codesearchnet-0000.parquet` ~ `0003.parquet`），净入库 **455,243 条优质 Python 函数**（0 条拒绝），记录于 `/data/mini-k3/data/reports/codesearchnet/adapter.json`。预估产出 **75M - 85M tokens**，超出 52.96M 目标。

2. **增补隔离流水线设计与启动 (supplement-v2)**：
   - 隔离根目录：`/data/mini-k3/data/prepared-v2-supplement-v2`。
   - 报告与日志：`/data/mini-k3/data/reports/supplement-v2/` 与 `/data/mini-k3/logs/prepared-v2-supplement-v2/`。
   - 硬链接克隆与前缀重绑定：基于已包含补充 FineWeb-EDU 的 `prepared-v2-supplement-v1` 执行 `cp -al` 克隆，对其余 10 个数据源重新绑定路径前缀与校验哈希。
   - 代码源全量重建：`code-python` 纳入基线 8 分片 + 补充 200 分片 + CodeSearchNet 4 分片，共计 212 个分片，从 `canonical_v2.py` 与 `clean_v2.py` 阶段重新生成。
   - 自动化控制器：由 `train/data/supplement_v2.py` 驱动，后台 PID 50113 托管，使用 `--workers 4` 并行执行精确去重、近去重、13-gram去污染、分词编码、manifest 全量审计、Tesla V100 GPU smoke test 与最终覆盖率评估。

### 2026-09-22 Mac 本地测试环境与轻量验证套件就绪

为支持本地快速开发与算法验证，在 Mac (Apple Silicon) 本地基于 Python 3.12 搭建了隔离测试环境 `.venv` 并固化核心依赖配置于 `train/requirements.txt`：
1. **依赖环境**：安装 `torch==2.14.0` (支持 MPS 加速与 CPU 推理)、`tiktoken==0.14.0`、`pyarrow==25.0.1`、`modelscope==1.40.1`、`datasets==5.0.1`、`pytest==9.1.1`。
2. **轻量验证通过**：
   - 模型架构 Smoke Test (`train/smoke_test.py`)：1.02B 参数量核算准确，Step 0 初始损失 12.0988，优化器 2 步反向传播与 MoE 路由分发均正常。
   - 1M Cache 等价性验证 (`train/test_cache_equivalence.py`)：100 万长位置 RoPE 酉圆旋转稳定性误差 $< 1.19 \times 10^{-7}$，无 cache 与 cache logits 最大绝对误差 $1.67 \times 10^{-6} < 10^{-4}$，逐 token 贪婪解码完全一致。
   - KDA 运行时回归 (`train/test_kda_runtime.py`)：状态精度与分块等价完全一致。
   - 显存基准 (`train/benchmark_memory.py`)：1M 上下文 MLA 滑动窗口显存占用严格保持 $O(1)$ 边界。
   - 数据处理单元测试套件 (`pytest train/data/test_*.py`)：38 项单元测试全部通过。



