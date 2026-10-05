# 4×Tesla V100-SXM2 (32GB) 预训练与对齐 Mini K3 规划总则

> 本文档覆盖当前 4 卡 V100 机器的预训练、SFT、DPO 与 PPO 对齐流程。所有训练相关代码均归档于 `train/` 目录中。

## 2026-10-04 当前状态：门禁通过，200 步四卡短跑完成

* 当前 `train/` 已同步到 `v100:/data/mini-k3/project/train/`，关键文件哈希与本地一致；远端旧的根目录测试副本已删除，训练只使用 `train/` 下的代码。
* 真实数据 smoke 通过，`SMOKE.json` 记录的模型代码哈希为 `4daf36695d48cf88224f04fe932e50b1f2ae23015ad75ea7123cb0f70ab018b7`；单卡峰值 allocated/reserved 显存约 14.09/14.72 GiB，失败与运行 sidecar 均已清理。
* readiness 通过：训练 manifest 为 `d2cce19c492c5204bfd369578581ecf5a60f772087da45f002d516f8a2e1b7ea`，验证 manifest 为 `541136c5a7f06cd6606513cecddc2bf20af8ace6cca5cb5ad92df503ded029c7`，coverage 为 `sufficient_fixed_mix`。
* 按文档参数完成 4 卡 FP16/GradScaler 200 步短跑（远端 shell 的 `PATH` 没有 `torchrun`，实际使用等价的 `/data/mini-k3/venv/bin/torchrun`）。第 0 步 lm 为 12.1114，最终第 199 步 lm 为 8.3152，验证 lm 为 8.4983；总 token 52,428,800；全程无 OOM、非有限值、梯度跳过或连续溢出。每卡峰值约 19.50 GiB allocated / 20.80 GiB reserved，稳定吞吐约 3,304 token/s。
* 完整检查点位于 `/data/mini-k3/checkpoints/pretrain-short-2048/step_000199`，`best/` 同步存在且带 `COMPLETE` 标记；GPU 已释放。10B 长跑尚未启动，仍需用户确认短跑吞吐后再执行。

## 2026-10-03 第一次短跑两次 OOM：根因与整改（训练当前没有在跑）

**现状**：短跑没有跑成，四张卡空闲，没有任何检查点。下面列出两次失败、根因、本次改动和下一步。本次只改了本地 `train/` 的代码和文档，没有 ssh 改服务器、没有启动训练、没有下载数据。

**两次失败**（服务器日志 `/data/mini-k3/logs/pretrain-short-2048.log`，以及 `.20261003191012.bak`）：

1. 07:02 EDT 第一次：第 0 步主干损失 12.1101（在 11.90–12.25 内），记录损失 15.7229（= lm + 0.3×MTP）。第 1 步反向申请 1.25 GiB 时 OOM。此时 PyTorch 已分配约 24.9 GiB，缓存约 5.0 GiB。
2. 07:10 EDT 第二次：加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。日志里有 `expandable_segments not supported on this platform`，所以这个变量没起作用。07:57 在第 10–19 步之间 OOM，位置在 situ 的反向。申请 372 MiB 时已分配 30.43 GiB。速度约 1,660 token/s（每个微步约 4.9 s），照这个速度 10B 要约 70 天。第 10 步 loss 15.35，Imb 1.64–2.21，Dead 0%。

**根因**（四个只读评审复核过）：

| 问题 | 原因 | 本次修改 | 影响 |
|---|---|---|---|
| 直接 OOM | 256 个专家都补齐到最忙专家的 token 数。单篇 2048 文档可让一个专家拿到约 915 次路由，372 MiB = 256×915×416×4 B。step 级 Imb 看不到微批次内的不均衡 | `RoutedExperts`：专家权重堆成 `gate_up [256, 832, 256]`、`down [256, 256, 416]`；每个专家只补齐到 64 的倍数，按块 bmm。不丢 token。补齐行数上限为 256×63，与路由不均无关 | 检查点键名变为 `layers.N.mlp.experts.gate_up/down`（之前没有可用检查点）。参数量不变：1,150,739,900 / 159,400,252 |
| 静态显存约 19 GB | FP32 参数 4.6 GB + 梯度 4.6 GB + DDP 桶副本 4.6 GB + Muon 动量 4.2 GB + AdamW 0.8 GB | DDP `gradient_as_bucket_view=True` | 每卡省一份 FP32 梯度，约 4.6 GB |
| logits | 训练时整段 `[2048, 163840]` logits 和 MTP 的一份同时驻留，约 5–7 GB | `compute_logits=False`：按 512 token 分块，在重计算下做 lm_head+CE，值与整段相同 | 不再驻留全词表 logits。训练、验证、step-0 探针、smoke、evaluate、SFT 都走这条路径 |
| MLA/CSA2 | gather 中间量约 2 GB；局部窗 unfold 实体化；全局分支丢掉压缩 key，直接把单头索引分数当注意力 logit；缓存与无缓存在 4096 之后不一致（违反红线第 8 条） | 重写 CSA2：每个 query 在同一个 softmax 里看最近 128 个原始 token 和索引器选出的 top-k 压缩条目，条目 logit 是逐头的 q·k/√d。索引器在 FP32、输入 detach，用 DSA 风格 KL 训练，权重 `csa_indexer_loss_weight=0.1`。缓存保存原始尾部 max(128,4) 个 latent 和全部压缩条目（只在第 4、8 层），不再有 4096 窗口。条目多时自动改用 MLA 权重吸收，结果完全相同 | 语义变化：缓存为 O(L/4)。1M 时 FP16 约 256 MiB，不是 O(1)。缓存与无缓存在任意长度都应一致，已用 120 token、15 倍局部窗、三种分块方式测试 |
| KDA | 每个 64-chunk 一次 `torch.linalg.solve`（LU 加主机同步），每微步 576 次；5D 临时张量每层约 2 GB | UT 变换：所有 chunk 的块内计算一次并行完成；单位下三角用 `solve_triangular`；块间只剩两次小矩阵乘；直接传 log 衰减；chunk 32（`kda_chunk_size`，结果与 chunk 大小无关） | 没有主机同步，5D 临时张量约 134 MB/层。与逐 token 递推在 float64 下对齐，包括梯度 |
| 其他同步 | Engram 每次前向约 144 次 `int(gpu_tensor)`；MoE bincount；逐参数 isfinite 循环；约 6k 个矩阵逐个做 Muon | Engram 哈希向量化（与旧哈希逐位相同）；用 scatter_add 计数；一次梯度范数判有限；专家按批做 Newton–Schulz | 每层 MoE 只剩 1 次同步（补齐总数） |
| 路由精度 | 路由器在 autocast 下跑 FP16 | 路由 logits 关闭 autocast，用 FP32 | 选择更稳定 |
| AttentionResidual | 堆叠在 checkpoint 外保存，约 1 GB | 放进每层的 checkpoint | — |
| situ 文档 | `situ_fused.py` 里有从未调用的 Triton 内核和“省 40GB”的说法 | 删除死代码和错误说法 | 行为不变 |

**超参变化（规则 3）**：

* **Muon 更新尺度**：每个正交化后的矩阵乘以 `0.2·sqrt(max(m,n))`（Moonlight 约定，`muon_update_scale=0.2`），更新的逐元素 RMS 约为 0.2，与 AdamW 共用 `6e-4`。原来的 RMS 是 `1/sqrt(max(m,n))`，约 0.03–0.06，所以 Muon 实际步长比原来大约 4–7 倍。峰值学习率不变。短跑要重点看前 200 步的 loss 和 grad norm。
* 新增配置：`moe_block_size=64`、`loss_chunk_size=512`、`kda_chunk_size=32`、`csa_indexer_loss_weight=0.1`、`muon_update_scale=0.2`。`attention_window` 不再使用，`--attention-window` 参数会被直接拒绝。
* 总损失 = lm + 0.3×MTP + 0.1×索引器 KL。日志分别打印 `loss / lm / mtp / aux`。第 0 步门禁、验证、best 检查点和 spike 检测都只看 **lm**。

**训练引擎**：

* 验证集是固定切片：每次评测前把验证 loader 倒回构造时的游标。`--validation_batches 8`（每卡）。评测发生在每 `--validation_interval`、每个存档步和最后一步。
* FP16 溢出跳步：只调用 `scaler.update()` 降低缩放，不计入 spike。连续 20 次溢出才中止（相当于缩放降了 2^20）。损失尖峰（lm 超过 EMA 的 1.5 倍）连续 5 次中止。
* 检查点：optimizer/scaler 只由 rank 0 写一份，所有卡都读这一份。每卡单独保存 RNG 和 loader。文件和目录都做 fsync。`best_metric` 持久化，`best/` 用硬链接。最新检查点损坏时，回退到上一个 COMPLETE 并大声警告。
* 运行签名新增：GA、micro batch、种子、WSD 比例、`schedule_total_steps`、两套配比、上述新配置、整份 config 的哈希。所以 config.py 一改，`--resume` 就会拒绝。
* 衰减期内恢复：loader 状态保存当前目标配比，从衰减期中途恢复是逐位一致的（CPU 测试：中断后恢复与不中断的最终权重完全相同）。
* `--schedule_total_steps`：短跑用 10B 的学习率曲线（预热 762 步）。不加这个参数时，200 步短跑只预热 4 步，并在第 170 步切到衰减配比。
* loader 遇到正权重来源读完时报错，不再静默重新归一化。readiness 不再回退到写死的 coverage 文件。
* **新门禁**：`SMOKE.json` 必须记录 `model_code_sha256`（`train/model_fingerprint.py`：config.py、models/、kernels/、engine/muon.py、engine/balancer.py 的哈希），而且必须与当前代码一致，否则 readiness 和 train.py 都拒绝启动。本次改了模型代码，服务器上现有的 SMOKE.json 已失效，必须重跑 smoke。smoke 每个优化步对每个来源各取一批，只有通过时才原子替换 SMOKE.json，失败写到 `SMOKE.failed.json`。
* NCCL 超时 60 分钟；改用 `torch.amp.GradScaler('cuda')`；日志新增每步时间、tok/s、各卡最大 allocated/reserved 显存（第 0、1 步和每个日志间隔都打印）。

**本地验证**（Mac CPU，torch 2.14）：`pytest train --ignore=train/reports` 134 通过、1 跳过（缺真实分词器）。另外直接运行 `test_architecture.py`、`test_v41_modules.py`、`test_kda_runtime.py`、`test_cache_equivalence.py`、`test_model_runtime.py`、`benchmark_memory.py`，全部通过。`test_model_runtime.py` 覆盖：块分发 MoE 与朴素逐专家循环一致；分块 CE 与整段 CE 一致；批量 Muon 与逐矩阵一致；CSA2 与逐 query 朴素实现一致；开关 checkpoint 时 loss 和梯度一致，且除视觉塔外每个参数都有梯度。另用真实维度（仅把路由专家减到 8 个以省内存）在 CPU 上跑 L=2048 前向+反向：lm 12.109（\(\ln 163840=12.007\)）、mtp 12.057、aux 0.019，全部有限，进程峰值 RSS 5.7 GB；这只证明真实形状能跑通，不代表 V100 显存。**V100 上的显存峰值和吞吐还没测**，只能由下面的 GPU 步骤回答。

**下一步，按顺序执行，每一步都要用户明确同意**：

1. 把本地 `train/` 同步到 `v100:/data/mini-k3/project/train/`，逐文件核对 `sha256sum`。删除服务器上多出来的旧文件：`train/test_readiness.py`、`train/test_run_options.py`、`train/validate_manifest.py`（真正的文件在 `train/data/` 下）。
2. 单卡重跑真实数据 smoke，生成带 `model_code_sha256` 的 SMOKE.json：
   ```bash
   CUDA_VISIBLE_DEVICES=0 /data/mini-k3/venv/bin/python train/smoke_from_manifest.py \
     --manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --report /data/mini-k3/data/prepared-v2-supplement-v2/manifests/SMOKE.json
   ```
3. 运行 readiness：`train/data/check_training_readiness.py`，命令见 2026-09-30 一节。
4. 200 步四卡短跑。不再带 `PYTORCH_CUDA_ALLOC_CONF`：
   ```bash
   /data/mini-k3/venv/bin/torchrun --standalone --nproc_per_node=4 train/train.py \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json \
     --checkpoint_dir /data/mini-k3/checkpoints/pretrain-short-2048 \
     --total_steps 200 --schedule_total_steps 38147 --save_interval 200 --log_interval 10
   ```
   合格标准：
   * 不 OOM；第 0 步 lm 在 11.90–12.25；lm 下降。
   * 第 0/1 步和每个日志间隔的 `max alloc / reserved` 都记下来，峰值要比 32 GiB 低出余量。
   * 记下 tok/s，按它重新估算 10B 的耗时。
   * Dead 0% 左右，Imb 有界；没有连续溢出中止。
   
   最后一步是第 199 步，存档为 `step_000199`。
5. 短跑合格并且用户确认吞吐可以接受之后，才开 10B。10B 的最后一个存档是 `step_038146`（步号从 0 开始）。


## 2026-09-27 收口记录与下一步

> **已被 2026-10-03 一节取代。** 本节写于短跑之前（当时“训练还没启动”）。之后短跑已尝试两次，都 OOM。其中“CSA 按 32 个 query 分块取回”“benchmark 检查窗口”等实现说明已被重写。下一步和命令以文首为准。下面保留原文作为记录，但删除了旧的短跑命令：它缺少 `--schedule_total_steps`，而且带着无效的 `PYTORCH_CUDA_ALLOC_CONF`。

代码和数据都停在「可以开短跑」这一步。训练还没启动。

已经收口的部分：

- 默认模型是 13 层 Mini K3，1,150,739,900 总参数、159,400,252 激活参数。KDA、门控无位置编码 MLA、Attention Residuals、四路 mHC、CSA2、Engram、缩小的 MoonViT、FP4 推理缓存、逐头 Muon 和深度 1 的 MTP 都在这条主干里。
- 本地全套实测已通过：`test_architecture.py`、`test_v41_modules.py`、完整模型 `smoke_test.py`（初始主干损失 12.0917）、`test_cache_equivalence.py`、`test_kda_runtime.py`、`benchmark_memory.py`，以及 60 项数据流水线测试。
- 已修的生成路径：缓存解码保留前文给 Engram；缓存位置按实际前向长度累加；CSA 分组按绝对位置对齐。CSA 先投影共享条目，再按 32 个 query 分块取回。
- SFT 使用模型自己的损失，包含 0.3 倍 MTP。`eval_long_context.py` 接受 `--tokenizer_model`。
- 文本数据不用再加。`prepared-v2-supplement-v2` 的覆盖率是 `sufficient_fixed_mix`：固定配比 100.000 亿 token，不重复采样最多 101.033 亿。最紧的是中文网页，余量约 1473 万 token。Python 缺口已经补上。对齐仍用 OpenAssistant、审核过的 OpenHermes、OpenR1、UltraFeedback。这次核对服务器没有连上，数字来自 `train/reports/verification/supplement-v2-20260926/coverage.json`。

下一步，按这个顺序做，不要插别的阶段：

1. 四张 V100 空出来之后，先把当前 `train/` 同步到 `v100:/data/mini-k3/project/train/`，确认服务器跑的就是这一版。
2. 在 `/data/mini-k3/project` 跑 200 步短跑。目录不要拿去恢复 10B。（命令见文首 2026-10-03 一节）

合格标准：四卡不 OOM，第 0 步主干损失在 11.90–12.25，随后损失下降，日志配比接近稳定期配比。同时记下每步时间和 `dead_frac`。这一跑要回答的就是显存峰值和速度。2026-10-03 的两次尝试都 OOM，见文首记录。

3. 短跑合格后，再跑 38,147 步，检查点目录用 `/data/mini-k3/checkpoints`，不要从短跑目录恢复。
4. 10B 完成之后才做长上下文。第一段 4096、学习率 \(6\times10^{-5}\)，单独目录。8192 和 16384 只在前一段损失和显存都稳时才开。
5. SFT、DPO、PPO、GRPO 用决定保留的那一档检查点。视觉塔等有图像数据再单开，不放进这次文本预训练。

## 2026-09-30 训练前复核：服务器版本与终态门禁

本次只做只读核对和代码修复，没有启动训练、下载数据或重启服务。

1. v100 的 `/data/mini-k3/project/train` 原来仍是 12 层、1,028,056,392 参数的旧副本；当前定案代码是 13 层、1,150,739,900 参数。正式短跑前必须重新同步当前 `train/`，并核对 `sha256sum`，不得使用旧副本。
2. v100 的 `prepared-v2-supplement-v2/PIPELINE_STATUS.json` 停在 2026-09-24 的 `full_manifest_audit/failed`。这是修复 OpenAssistant 分组后遗留的旧状态；同一数据根的 2026-09-26 `manifests/AUDIT.json`、`SMOKE.json` 和 `reports/supplement-v2/coverage.json` 已分别为 `passed`、`passed`、`sufficient_fixed_mix`。启动前须通过当前代码重新收敛为 `PIPELINE_STATUS.status=complete`，不能只看 `AUDIT.json`。
3. `train/readiness.py` 和 `train/data/check_training_readiness.py` 现在在模型建卡前检查：同一审计目录、schema-v2、词表、稳定期配比、全部 shard 存在且字节数一致、`AUDIT.json` 的 manifest 哈希绑定、真实数据 smoke 绑定、pipeline complete 和固定配比 coverage。训练入口不再只检查 manifest 文件是否存在。
4. 训练入口固定默认种子 42，并恢复 step-0 探针消耗的 Python RNG；对齐的 Reward/Value head 改为调用完整 Mini K3 前向，不能再把四路 mHC 层当成单输入层调用。

同步完成后，短跑前置检查命令为：

```bash
/data/mini-k3/venv/bin/python train/data/check_training_readiness.py \
  --manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
  --validation-manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json
```

该检查通过后才允许执行下面已有的 200 步命令；通过不等于 200 步四卡显存和吞吐已经验证。

## 2026-09-27 架构对齐：小而完整的 K3，下一步仍是 2048 短跑

> **部分已被 2026-10-03 取代：**
> * 第 2 条“2048 训练走整段因果注意力，超过 4096 按窗口截断”不再成立。现在所有长度都走 CSA2：局部 128 加压缩条目，没有 4096 窗口。
> * 第 6 条中“全局最多取 512 条”仍然成立，但条目 logit 改为逐头 q·k，索引器只负责选条目。
> * 第 9 条的 FP4 缓存现在存的是压缩条目（opt-in）。
> * 文末 CSA 实现说明和“权重量级状态约 7.30 GB”作废。7.30 GB 只算了 FP16 权重和优化器状态；实际训练用 FP32 参数存储，静态显存约 14 GB 起，外加激活，见文首。

原因：对照 Kimi K3 技术报告和 DeepSeek-V4 / V4.1 之后，主干里有几处和论文不一致，12 层上也缺了 K3 用来在深度上取信息的 Attention Residuals。这些都是参数和算子级的改动，不改数据、不改 2048 的训练命令。训练还没开始，没有旧检查点需要迁移。

这次写进默认模型的部分：

1. KDA 的衰减改为有下界的 scaled sigmoid：\(g=\mathrm{g_{min}}\mathrm{sigmoid}(e^{A_h} z)\)，\(\mathrm{g_{min}}=-5\)，\(A_h\) 初始为 0。输出先做头 RMSNorm，再乘全秩 sigmoid 门。短卷积、L2 归一化和可学习 \(\beta\) 保留。分块递推与逐步递推在 11 token 上对齐。
2. MLA 改为门控、无位置编码。位置只由 KDA 的衰减携带，不再做 RoPE，也不为加长上下文去改 RoPE base。缓存只存 KV latent，推理时重建 K/V。2048 训练仍走整段因果注意力；超过 4096 的推理继续按窗口截断。
3. 13 层：9 层 KDA 加 4 层门控 MLA，MLA 在第 4、8、12、13 层。最后两层连续是 MLA，对应 K3 末尾多出来的那一层全局注意力。Attention Residuals 用完整形式，查询初始为 0。
4. 路由专家仍在 256 维潜空间里计算。RMSNorm 在专家混合之后、升维之前。两个共享专家全宽相加。负载用分位数均衡，256 个箱子覆盖 \([-2,2]\)，本步前向不用本步算出的偏置。
5. 四路 single-pass mHC。残差流是 4 条宽度 512 的流，层内计算仍是 512，不是把隐藏维乘 4。Sinkhorn 20 次。初始时残差接近恒等，写入和读出都在第 0 条流。
6. 前 8 层是因果编码器，后 5 层是解码器。编码器的 MLA 为 Full。第 12 层 Reindex，第 13 层 Reuse。CSA2 把每 4 个 token 压成一条，局部窗口 128，全局最多取 512 条。2048 长度上，窗口之外的前文走压缩条目。
7. Engram 放在第 2 层和第 8 层，阶数 2/3/4，每阶 8 个头，桶大小取不小于 10007 的互异质数。输出投影初始为 0。文本批次会更新它。
8. MoonViT-V2 按文本模型的比例缩小：4 层、宽度 512、patch 14、2×2 pixel shuffle、空间注意力加时间注意力。没有图像的批次不调用它，因此纯文本步不更新视觉参数。
9. 矩阵权重用逐头 Muon，嵌入、偏置和归一化用 AdamW。推理缓存把 MLA latent 打成 E2M1 FP4；训练激活仍是 FP16。深度 1 的 MTP 权重仍是 0.3。
10. `train.py --model ced` 仍是单独的稠密对照，不是这条主干里的因果编码器。

CSA2 先把共享的压缩条目投影成 value，再按 query 分块取回，避免在 2048 长度上先把 `[序列, top_k, rank]` 投成完整头维度。SFT 使用模型自己的损失，也就是 assistant 的下一个 token 加上 0.3 倍 MTP；某一行在第 2 个 token 之后没有监督目标时，这一步只训练下一个 token。长文检索脚本接受 `--tokenizer_model`。

参数量改为 1,150,739,900，激活 159,400,252。种子 42、批大小 2、长度 64 的主干损失是 12.14，仍在 11.90–12.25。权重量级状态约 7.30 GB：FP16 权重 2.30 GB，Muon 动量 4.20 GB，AdamW 的动量和方差 0.80 GB。下一步仍是 200 步、序列 2048、目录 `/data/mini-k3/checkpoints/pretrain-short-2048`。这次改了结构，短跑从零开始。没有图像批次时，视觉塔留在检查点里，但不参与文本损失。

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

KDA审计另发现 A_log 未参与计算、state 的 key/value 广播轴错误、AMP 下 float() 仍不能保证 matmul 使用FP32。修复后先计算 log_decay=-exp(A_log)*softplus(gate+dt_bias)，作用于state的key轴，再做delta更新；递归过程显式关闭autocast并保存FP32 state。delta 写入使用可学习的 per-head beta，输出在进投影前乘 per-channel 门。该修改不删除或替换 CED 分支。

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

**训练基准**：4×Tesla V100-SXM2 32GB，FP16 + GradScaler，预训练序列长度 2048，micro-batch 1 / GPU，梯度累积 32，四进程 NCCL DDP。推理位置上限为 1,048,576。MLA 走 CSA2：局部 128 个原始 token 加最多 512 条压缩条目，不做 1M×1M 注意力。缓存随长度按 L/4 条增长，1M 时 FP16 约 256 MiB。发布 1M 能力前必须完成长上下文继续训练和评测（红线第 8 条）。

**存储规划**：所有大文件统一放在 `/data/mini-k3/`：原始下载集和 uint32 shards 在 `/data/mini-k3/data/`，checkpoint 在 `/data/mini-k3/checkpoints/`，日志在 `/data/mini-k3/logs/`，RL rollout 在 `/data/mini-k3/rollouts/`。代码、配置和小型索引继续放在 SSD 工作目录，不把数据下载到项目目录。

* **计算卡**：4×Tesla V100-SXM2 32GB
* **精度**：FP16 + GradScaler；禁止在 V100 上使用 BF16 或 FP8
* **训练方式**：四进程 NCCL DDP；每卡独立取数，梯度同步，rank 0 负责 checkpoint 和日志。
* **实现约束**：V100 使用 FP16 + GradScaler；不启用 H100 专用 BF16/FP8 或 sm_90 内核路径。
* **核心瓶颈预警**：
  * 每卡 32GB；显存余量必须由 smoke test 和 200-step benchmark 实测确认。

---

## 2. 唯一定案模型规格（Mini K3，当前结构实算 1,150,739,900）

规格以当前代码 `MiniK3ForCausalLM(MiniK3Config()).count_parameters()` 为准。2026-09-27 实算包含 KDA、门控无位置编码 MLA、Attention Residuals、四路 mHC、CSA2、Engram、缩小的 MoonViT、FP4 推理缓存和逐头 Muon。不再沿用同日较早的 1,028,762,056。

### 2.1 详细规格表

| 字段参数 | 当前代码实算：Mini K3 | 原版 Kimi K3 参考对照 |
| :--- | :--- | :--- |
| **隐藏维度 (hidden_size)** | 512 | 7,168 |
| **总层数 (num_layers)** | 13 | 93 |
| **注意力架构比例** | 9层 KDA + 4层 MLA，末尾两层都是 MLA | 69层 KDA + 24层 MLA，末尾连续两层 MLA |
| **全注意力 (MLA) 所在层** | 第 4, 8, 12, 13 层 (1-indexed) | 每第 4 层，再加末尾一层 |
| **编码器 / 解码器** | 前 8 层 / 后 5 层 | V4.1 为 20 / 20 |
| **MLA** | 无位置编码，内容维 128，value 128，输出有全秩门 | 无位置编码的门控 MLA |
| **head_dim** | 128 | 128 |
| **KDA 头数 / MLA 头数** | 4 头 (4×128=512) / 8 头 | — / 96 头 |
| **前置稠密层 (Dense Layer)** | 第 0 层 (Dense MLP) | 第 0 层 (Dense MLP) |
| **路由专家池 (Routed Experts)**| 256 个 | 896 个 |
| **每 Token 激活专家数 (top_k)**| 6 个 | 16 个 |
| **共享专家数 (Shared Experts)**| 2 个 (始终无条件激活) | 2 个 (始终无条件激活) |
| **潜在表示瓶颈宽度 (Latent)** | 256 (hidden // 2) | 3,584 (hidden // 2) |
| **专家 FFN 中间维度** | 416 | 3,072 |
| **词表大小 (Vocab Size)** | 163,840 (K3 官方 BPE) | 163,840 |
| **词嵌入共享策略** | `tie_word_embeddings: true` | `false` |
| **激活函数** | `situ` ($\beta=4.0, \text{linear\_beta}=25.0$) | `situ` |
| **训练上下文长度** | 2,048 | 1,048,576 |
| **总参数量 (Total Params)** | **1,150,739,900** | 2.8T |
| **激活参数量 (Active Params)** | **159,400,252** | 104B |
| **非嵌入激活参数量** | **75,514,172** | ~103B |
| **路由专家占激活参数** | 14.4%（23,003,136 / 159,400,252） | 46.1% |
| **路由专家占非嵌入激活参数** | 30.5%（23,003,136 / 75,514,172） | — |

参数按模块拆开（同一实算）：

| 模块 | 参数量 | 占总量 |
|---|---:|---:|
| 256 个路由专家 × 12 层 | 981,467,136 | 85.29% |
| 绑定词嵌入 | 83,886,080 | 7.29% |
| MoonViT-V2（缩小） | 17,345,536 | 1.51% |
| Engram 哈希表 | 15,531,648 | 1.35% |
| 2 个共享专家 | 15,335,424 | 1.33% |
| 4 层门控 MLA 与 CSA2 | 11,213,824 | 0.97% |
| 9 层 KDA 主体 | 10,105,380 | 0.88% |
| MoE 路由与潜空间投影 | 4,724,736 | 0.41% |
| MTP 头（默认开启） | 3,672,064 | 0.32% |
| 第 0 层稠密 MLP | 3,145,728 | 0.27% |
| KDA 写入门、输出门、头归一化 | 2,383,524 | 0.21% |
| Engram 融合层 | 1,580,544 | 0.14% |
| 四路 mHC | 320,116 | 0.03% |
| Attention Residuals | 14,336 | — |
| 层归一化 | 13,824 | — |
| **合计** | **1,150,739,900** | **100%** |

激活参数 = 总参数 − 路由专家 − Engram 表 − 视觉塔 + 路由专家 × 6 / 256 + 每个 token 实际查到的 Engram 行。文本步不算视觉塔。当前 `train.py` 保持 FP32 参数存储并在 V100 上用 FP16 autocast/GradScaler；因此 1,150,739,900 个参数仅权重约 4.60 GB，梯度、Muon/AdamW 状态、激活和 KDA 临时张量必须由 2048、四卡短跑实测，不能沿用旧的 7.30 GB 下限。64-token 单卡真实 smoke 峰值为 15,136,763,904 bytes，不外推正式长度。预训练记录的 loss = 下一个 token + 0.3 倍 MTP + 0.1 倍 CSA 索引器 KL，日志分开打印 lm/mtp/aux。第 0 步是否合格、验证和 best 检查点都只看下一个 token 的 loss（lm）。种子 42 的 64-token 主干损失是 12.14。矩阵权重走逐头 Muon：堆叠的路由专家逐个专家正交化，更新乘 0.2·sqrt(max(m,n))。嵌入、Engram 表、卷积核和一维参数走 AdamW。KDA 按 32 token 一块做 UT 变换，求解单位下三角，没有主机同步。激活重计算包住注意力和 MoE；专家负载和分位数直方图都在重计算之外只累加一次。推理 FP4 条目缓存是 opt-in，默认先用未量化 cache 做等价性门禁。`generate` 目前只用目标模型逐 token cache 解码，MTP 仅作为训练损失，不宣称 speculative decoding。

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
| **`code-python`** (全量) | 预训练 / 代码 | `stack-v3-train` + `codeparrot/github-code` + CodeSearchNet | MIT / Apache-2.0 / BSD | `/data/mini-k3/data/raw/code-python/` | 212 个 parquet | 1.288 B (训练) | supplement-v2 审计通过，不再用 1.181B |
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
| Python 代码 | `code-python` | 10% | 25% | 1.225 B | 1.288 B | 105% 覆盖 |
| 中文合成教材 | `cosmopedia` | 10% | 5% | 0.93 B | 1.69 B | 🌟 182% 覆盖 |
| **预训练总计** | — | **100%** | **100%** | **10.00 B** | **28.215 B** | 固定配比无重复上限 10.103 B |

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
  * **预热期（Warmup）**：第 0 ~ 761 步，共 762 = int(38147×0.02) 步。第 s 步 LR = 6e-4×(s+1)/762，第 761 步达到 $6.00 \times 10^{-4}$；
  * **稳定期（Stable）**：第 762 ~ 32,423 步，恒定维持峰值 LR，使用稳定期配比；
  * **衰减期（Decay）**：第 32,424 = int(38147×0.85) ~ 38,146 步（最后一步的步号是 38,146），从 $6.00 \times 10^{-4}$ 线性降到第 38,146 步的 $6.009 \times 10^{-5}$。数据加载器在第 32,424 步切换为强化数学/代码的衰减配比。
  * 短跑用 `--total_steps 200 --schedule_total_steps 38147` 沿用这条曲线：第 199 步 LR ≈ 1.575e-4，不进入衰减期。

---

## 3. 必须遵循的工程硬规（原书 30 章踩坑总结）

所有在 `train/` 目录下编写的代码严格内置以下 6 项防护：

1. **第 0 步有限性断言与初始损失对齐**：
   * 训练循环启动前必须运行 `assert_initialised(model)`，确保每一个参数和缓冲区都是有限浮点数（无 `inf`/`NaN`）。
   * 对均匀分布在 163,840 词表的交叉熵损失为 $\ln(163,840) \approx 12.007$。第 0 步输出必须处于 $11.90 \sim 12.25$ 之间，偏差过大立即中断。真实数据 smoke 测到过 12.00 和 12.19，收窄到 12.05–12.15 会把合格初始化判失败。
2. **KDA 衰减与时间步偏置**：
   * 衰减是 \(g_{\min}\mathrm{sigmoid}(e^{A_h} z)\)，\(g_{\min}=-5\)，\(A_h\) 初始为 0。不要改回负 softplus 再夹断。
   * `dt_bias` 仍用对数空间均匀采样加 softplus 反函数初始化，作为衰减 logit 的偏置：
     $$\text{dt} \sim \exp(\text{Uniform}(\ln 10^{-3}, \ln 10^{-1})), \quad \text{bias} = \text{dt} + \ln(1 - e^{-\text{dt}})$$
3. **MoE 分位数均衡**：
   * 选择专家时加上偏置，混合权重只用未加偏置的分数，并在选中的 k 个专家上归一化。
   * 偏置取「分数减第 k+1 名截止值」的分位数，再减去均值。256 个箱子覆盖 \([-2, 2]\)，四卡和 32 次微批次的计数加总后再读。本步前向不用本步算出的偏置。
   * `e_score_correction_bias` 必须排除在 AdamW 之外。
   * 监控 **`dead_frac`** 和 **`imbalance`**，不用路由器分数熵。
4. **MoE 显存不得随路由不均增长**：
   * 路由专家按 64 token 的块分发，每个专家只补齐到 64 的倍数，补齐行数上限为 专家数×63。禁止把所有专家补齐到最忙专家的 token 数（2026-10-03 OOM 的直接原因）。
   * `situ` 用重计算实现，只保存输入；V100 上禁止调用仅支持 sm_90 的 BF16/FP8 内核。仓库里没有 Triton 内核。
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
* **吞吐与耗时**：必须通过 200-step benchmark 实测后填写，禁止沿用 H100 估算。2026-10-03 旧代码实测约 1,660 token/s（10B 约 70 天），新代码待测。
* **显存（每卡，估算）**：FP32 参数 4.6 GB + FP32 梯度 4.6 GB（DDP 桶视图，不再多一份）+ Muon 动量约 4.2 GB + AdamW 约 0.8 GB ≈ 14.2 GB 静态。激活靠逐层重计算、分块 CE、按块 MoE、CSA2 分块控制，峰值以短跑日志里的 `max alloc / reserved` 为准。
* **logits**：训练、验证、smoke 一律 `compute_logits=False`，按 512 token 分块算 CE；推理预填充用 `logits_to_keep=1`。

---

## 5. 完整 1B 训练生命周期

### 5.0 与正式 Kimi K3 的模块对照

| 模块 | 正式 Kimi K3 | 本 Mini K3 | 状态 |
|---|---|---|---|
| KDA/线性注意力 | 有下界 scaled sigmoid、头 RMSNorm、输出门 | 9 层，同一套公式 | 分块递推已与逐步递推对照 |
| Attention Residuals | 按块压缩以省流水线通信 | 13 层用完整形式，查询初始为 0 | 参数量 14,336 |
| mHC | 4 路，Sinkhorn 20 次 | 4 路 single-pass，层内宽度仍是 512 | 初始接近恒等映射 |
| CSA2 | Full / Reindex / Reuse | 第 4、8 层 Full，第 12 层 Reindex，第 13 层 Reuse | 分组 4，局部 128，全局 top 512 |
| Engram | 大表条件记忆 | 第 2、8 层，阶 2/3/4，8 头，质数桶 | 输出投影初始为 0 |
| MoonViT-V2 | 27 层、约 401M | 4 层、宽度 512、patch 14、2×2 shuffle | 无图像的批次不调用 |
| FP4 KV | 推理缓存 E2M1 | 软件打包，训练激活仍是 FP16 | V100 没有 FP4 张量核 |
| 优化器 | Per-Head Muon | 矩阵用逐头 Muon，其余用 AdamW | 对齐阶段同样使用这个优化器 |
| MTP | V4 仍保留深度 1 的 MTP | 损失权重 0.3，默认开启 | 第 0 步门禁用的是下一个 token 的 loss |
| MLA/全局注意力 | 门控、无位置编码 | 4 层 CSA2：局部 128 加索引器 top-512 压缩条目（逐头 q·k），无 4096 窗口 | 缓存为原始尾部加压缩条目，O(L/4) |
| 稠密对照 | 不是 V4.1 的 Causal Encoder-Decoder | `models/deepseek_coder.py`，`train.py --model ced` | 默认关闭，不写入 Mini K3 |
| MoE | Stable LatentMoE，分位数均衡 | 256 routed + 2 shared，上投影前 RMSNorm | 偏置不进 AdamW |
| 长上下文 | KDA 携带位置，MLA 不加 RoPE | 位置上限仍记 1,048,576 | 这不是训练长度 |
| 1M 上下文与缓存等价性 | 严格等价 | `models/kv_cache.py`, `test_cache_equivalence.py` | 小模型上 Full/Reindex/Reuse 三种模式、15 倍局部窗、三种分块方式一致（差异 <1e-6）。这只是实现自检，不能代替长上下文继续训练后的检索成绩 |
| 1M 显存基准 | KDA O(1) + CSA 条目 O(L/4) | `benchmark_memory.py` | 理论表（1M 时约 259 MB）加小模型实测条目数。不是 1M 预填充的显存证明 |
| 长文档检索评测 (NIAH) | 大海捞针检索 | `eval_long_context.py` | 继续训练之后再评。默认长度：2048/4096/8192/16384 中不超过训练长度的那些；更长要加 `--allow-untrained-length`，结果标 UNTRAINED |
| 数据流与二进制分片 | uint32 分片 | `data/build_shards.py` + `data/loader.py` | 完整实现，生成 manifest.json |
| 训练后对齐 | SFT、RM、DPO、PPO、GRPO | `alignment_train.py --mode grpo` | GRPO 不用价值网络；奖励检查 `<think>` 和答案 |
| 交互式终端推理 | 流式对话 | `chat.py` (基于 KV Cache 逐 Token 输出) | 完整实现 |
| 多卡训练 | 专家/参数并行 | 4 卡 DDP + FP16 GradScaler | 针对 4×V100 优化 |

本模型的目标是“模块齐全的缩小版”，不是参数和能力等价复刻。任何标称 1M 上下文的 checkpoint 必须通过长文档检索、跨段问答、长代码和 KV-cache 等价性测试。

同一个 `MiniK3ForCausalLM` 骨干贯穿四个阶段，避免为 SFT/RL 重新实现模型：

1. **预训练**：约 10B tokens（四卡全局 batch），WSD，混合网页/代码/数学语料；保存模型、AdamW、数据游标、Python/NumPy/PyTorch RNG 和监控状态。
2. **SFT**：使用 `chat_template.py` 把 system/user/assistant 对话打包，prompt 标签为 `-100`，只训练 assistant token，并保留 0.3 倍 MTP；保留 EOS，按长度分桶并验证截断率。
3. **偏好对齐**：先训练 `RewardModel`（chosen 分数高于 rejected 的 pairwise loss），再用冻结 reference 做 DPO；DPO 默认对响应 log-prob 求和，`--dpo_length_norm` 改为按 token 平均，并记录隐式奖励、margin 和 k3 KL。
4. **在线 RL/PPO**：policy、reference、value、reward 四个模型分工；rollout 记录 token、old log-prob、reference log-prob、value、reward 和终止原因，PPO 使用 ratio clipping、value clipping、优势标准化和 KL 惩罚。奖励模型、EOS、最大长度、拒答/安全规则都必须版本化。
5. **最终 SFT**：用人工审核和 RL 过滤后的高质量轨迹再做短程 SFT，低学习率、冻结或降低底层层学习率，作为最终发布 checkpoint。

阶段间只传递 `model.pt` 和明确的 tokenizer/config 版本；每阶段单独目录和 manifest，禁止覆盖预训练 checkpoint。

### 5.1 四卡 checkpoint 红线

每个保存点 `step_{step:06d}/`：rank 0 写一份 `model.pt`、`optimizer.pt`、`scaler.pt`、`meta.pt`（DDP 下各卡的优化器状态相同）；每个 rank 写自己的 `rng_rankN.pt`、`loader_rankN.pt`。文件和目录都 fsync，rank 0 在 barrier 后写 `COMPLETE` 并原子改名。恢复时校验 `world_size` 和运行签名（含整份 config 哈希、schedule_total_steps、两套配比），任一不一致就拒绝。最新 COMPLETE 损坏时回退到上一个并警告。`best/` 按验证 lm loss 选择，用硬链接发布。

## 6. 监督微调、偏好优化与强化学习

> 2026-10-03 修订（原因：评审发现 GRPO 不截 EOS、拿不到金标答案、KL 是序列均值几乎不起作用、PPO 只有 1 个 epoch 且 ratio 恒为 1、各模式共用输出目录、对话模板与 `tokenize_v2` 不一致、PPO 四个 FP32 模型放不进一张 V100；影响：后训练命令、默认值和输出目录全部更新）。

预训练检查点之后的统一接口在 `alignment.py`、`alignment_models.py`、`rl_trainer.py` 与 `alignment_train.py`：

* **输出目录**：默认 `/data/mini-k3/checkpoints/alignment-<mode>`，每个模式分开。目录里已有 `model.pt`、`reward_model.pt`、`value_model.pt` 或 `alignment_meta.json` 时拒绝写入，除非 `--overwrite`。预训练 checkpoint 目录永远拒绝写入。每次保存都写 `alignment_meta.json`，记录模式、基座、序列长度、数据集证据、参数和产物 SHA256。
* **checkpoint 解析**：`--checkpoint` 可以是以下任一种：
  * 运行根目录：取最新带 `COMPLETE` 的 `step_*`，没有就报错；
  * step 目录；
  * 对齐输出目录；
  * `model.pt` 文件。
  
  序列长度依次取 `meta.pt` 的 `run_signature.sequence_length`、`alignment_meta.json`；`--sequence_length` 可以覆盖；都没有时用 2048。
* **SFT**：损失由模型的分块路径计算（`compute_logits=False`），包含 assistant 的下一个 token、0.3 倍 MTP 和训练态的索引器 KL。`--steps` 按 micro-batch 计数，每 `--grad_accum_steps` 个做一次更新。最后不满一组时按实际个数平均。`--log_interval` 按优化器更新计数。
* **RM**：`RewardModel` 从因果 LM 初始化（`source=causal`），输出 `reward_model.pt`。margin 和准确率取一组 micro-batch 的平均。打分头（1×512）标记 `adam_only`，走 AdamW。
* **DPO**：reference 冻结，CUDA 上为 FP16。默认对响应 log-prob 求和；`--dpo_length_norm` 改为按 token 平均。日志记录 chosen/rejected 隐式奖励、margin、准确率和 chosen 上的 k3 KL。`--dpo_beta` 默认 0.1。
* **PPO**：
  * rollout 在第一个 EOS 截断：保留 EOS，丢弃之后的 token。
  * rollout 时一次性记录逐 token 的 old log-prob、reference log-prob 和 value。
  * 奖励模型分数放在最后一个响应 token 上，做逐 token GAE（`--gamma 1.0 --lam 0.95`），优势在响应 token 上标准化。
  * 损失 = 逐 token clipped surrogate + `kl_beta`×逐 token k3 KL（默认 0.02）+ clipped value loss。KL 是损失项，不并入奖励。
  * `--ppo_epochs` 默认 2。
  * `--reward_checkpoint` 必须是含 `reward_model.pt` 的目录或该文件本身，没有静默回退。value 默认从 `--checkpoint` 初始化，也可以用 `--value_checkpoint` 指定 `value_model.pt`。
* **GRPO**：
  * 只读审计过的 SFT 文件（OpenR1）。提示是第一个监督标签之前的 token，含 `<|assistant|>\n` 头；金标答案绝不进入提示。
  * tokenized 行里没有答案字段，答案按 `id` 从 `decontaminated/openr1/part-*.jsonl` 的 `answer` 字段读取。`decontaminated/openr1/COMPLETE.json` 的哈希必须等于 alignment manifest 中该源的 `upstream_sha256`，每个分片都核对大小和 SHA256。
  * 没有答案的源（OpenAssistant、OpenHermes）直接报错。
  * 奖励 = `<think>` 格式 0.5 + 非空答案 0.5 + 与金标完全一致 1.0（也接受最后一个 `\boxed{}` 里的内容）。
  * KL 为逐 token k3（默认 0.04）；`--grpo_epochs` 默认 1；rollout 时记录 old log-prob。
* **共用 RL 参数**：`--max_new_tokens 256`（OpenR1 推理链很长，正式跑大概率要调大）、`--temperature 1.0`、`--top_p 1.0`、`--clip_eps 0.2`、`--prompt_batch 4`、`--group_size 4`、`--micro_batch_size 1`、`--eos_token_id 163585`、`--pad_token_id 163839`。
* **显存方案（4×V100 32GB，FP16）**：
  * 可训练模型保留 FP32 主权重，前向用 `autocast(float16)`，反向用 `GradScaler`。
  * 冻结的 reference/reward 转 FP16（`--frozen_dtype fp16`，每个约 2.3 GB）。
  * PPO 建议 `--device cuda:0 --value_device cuda:1 --ref_device cuda:2 --reward_device cuda:3`。
  * log-prob 从 hidden 分块计算，不常驻全词表 logits。
  * CUDA 路径还没在 V100 上实测。
* **MoE 均衡**：每次优化器更新后，对 policy/value/RM 调用 `NoAuxBalancer.step()`；`--no_balancer` 可关闭。
* **已知限制**：
  * RL/DPO 的 log-prob 前向不带 labels，所以 CSA 索引器 KL 在 DPO/PPO/GRPO 中不训练（SFT 中仍训练）。
  * padding token 会计入 MoE 负载统计。
  * `encode_preference`（UltraFeedback）把 prompt 轮整串编码，响应不带结尾换行，与 SFT 模板不同；推理按 SFT 模板。这是已审计数据代码的行为，不改。

示例命令。`<ctx>` 用决定保留的那一档检查点：长上下文目录，或 `/data/mini-k3/checkpoints/step_038146`。

```bash
python3 train/alignment_train.py --mode sft --checkpoint <ctx> \
  --jsonl /data/mini-k3/data/prepared-v2-supplement-v2/tokenized/openhermes/train.jsonl
python3 train/alignment_train.py --mode rm --checkpoint /data/mini-k3/checkpoints/alignment-sft \
  --jsonl /data/mini-k3/data/prepared-v2-supplement-v2/tokenized/ultrafeedback/train.jsonl
python3 train/alignment_train.py --mode dpo --checkpoint /data/mini-k3/checkpoints/alignment-sft \
  --jsonl /data/mini-k3/data/prepared-v2-supplement-v2/tokenized/ultrafeedback/train.jsonl --ref_device cuda:1
python3 train/alignment_train.py --mode ppo --checkpoint /data/mini-k3/checkpoints/alignment-sft \
  --reward_checkpoint /data/mini-k3/checkpoints/alignment-rm \
  --jsonl /data/mini-k3/data/prepared-v2-supplement-v2/tokenized/ultrafeedback/train.jsonl \
  --device cuda:0 --value_device cuda:1 --ref_device cuda:2 --reward_device cuda:3
python3 train/alignment_train.py --mode grpo --checkpoint /data/mini-k3/checkpoints/alignment-sft \
  --jsonl /data/mini-k3/data/prepared-v2-supplement-v2/tokenized/openr1/train.jsonl \
  --tokenizer_model /data/mini-k3/data/tokenizer --ref_device cuda:1
```

**对话模板**：`chat_template.pack_chat` 与 `tokenize_v2.encode_messages` 逐 token 一致。头部与正文分开编码，assistant 正文为 `content+"\n"+EOS`。截断时从最旧的整轮开始丢，开头的 system 放得下就保留，最新的 user 轮绝不丢。

**chat.py / eval_long_context.py**：
* `chat.py` 用 KV cache 逐 token 流式输出，按累计 id 增量解码，不会拆开多字节 UTF-8。
* `eval_long_context.py` 做分块缓存预填充（`--prefill_chunk 4096`），参数默认 `--samples 5`、`--seed 1234`、`--max_new_tokens 8`，CUDA 上默认 FP16 autocast。

## 7. 目录组织与可执行脚本

```text
train/
├── TRAINING_PLAN.md          # [本文件] 唯一定案训练规划
├── PIPELINE_STAGE_REPORTS.md # 数据流水线阶段报告
├── config.py                 # Mini K3 唯一定案配置（1,150,739,900 / 159,400,252）
├── model_fingerprint.py      # 模型代码哈希，SMOKE.json 与 readiness 绑定
├── readiness.py              # 训练前数据/审计/smoke/覆盖率门禁
├── run_options.py            # 序列长度、学习率、schedule 长度等运行参数解析
├── train.py                  # 预训练入口（4 卡 V100 DDP，FP16 GradScaler）
├── smoke_test.py             # 随机 token 架构自检（不授权训练）
├── smoke_from_manifest.py    # 真实数据 smoke，写 SMOKE.json（含 model_code_sha256）
├── evaluate.py               # 验证集 lm loss 与题目准确率
├── eval_answers.py           # 选择题答案解析
├── eval_long_context.py      # 大海捞针长文检索（红线第 8 条）
├── benchmark_memory.py       # CSA2 缓存显存理论表与小模型实测（红线第 8 条）
├── test_cache_equivalence.py # 无 cache / cache logits 等价（红线第 8 条）
├── test_model_runtime.py     # MoE 分发、分块 CE、Muon、CSA2 参考实现、重计算梯度
├── test_kda_runtime.py       # KDA 分块与逐步递推、梯度
├── test_architecture.py / test_v41_modules.py
├── test_training_engine.py / test_train_loop_cpu.py  # 检查点、spike、恢复逐位一致
├── chat.py / chat_template.py
├── alignment.py / alignment_models.py / alignment_train.py / alignment_fit.py / alignment_readiness.py
├── rl_trainer.py / rule_reward.py
├── models/
│   ├── mini_k3.py            # 组装、分块 CE、logits_to_keep、缓存解码
│   ├── kda.py                # KDA，UT 变换分块递推
│   ├── mla.py                # 门控 NoPE MLA + CSA2（局部带 + 索引器 top-k 条目）
│   ├── moe.py                # 路由器（FP32）、堆叠专家按块分发、共享专家
│   ├── kv_cache.py           # KDA 状态、卷积状态、CSA2 尾部与条目存储
│   ├── mhc.py / attn_res.py / engram.py / mtp.py / moonvit.py / fp4.py / attention_mask.py
│   └── deepseek_coder.py     # CED 稠密对照（不进 Mini K3）
├── kernels/
│   └── situ_fused.py         # situ 重计算实现（无 Triton）
├── engine/
│   ├── muon.py               # 逐头/逐专家 Muon + AdamW
│   ├── scheduler.py          # WSD
│   ├── balancer.py           # 分位数均衡
│   ├── spike_guard.py        # lm EMA 尖峰与溢出跳步
│   ├── checkpoint.py         # 原子检查点、回退、best
│   └── init_patch.py
└── data/                     # 数据流水线（*_v2.py、stage_audit.py 等受哈希绑定，禁止修改）、loader.py、测试
```

### 快速启动命令

以下 GPU 命令都需要用户明确同意后才能执行（红线第 7 条）。

1. **架构自检（随机 token，不验证数据，也不授权训练）**：
   ```bash
   python train/smoke_test.py
   ```
   生产数据另跑真实 manifest smoke。它绑定当前 `AUDIT.json` 和模型代码哈希，只在通过时原子替换 SMOKE.json：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train/smoke_from_manifest.py \
     --manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --report /data/mini-k3/data/prepared-v2-supplement-v2/manifests/SMOKE.json
   ```
   改动 `config.py`、`models/`、`kernels/` 或 `engine/muon.py` 后，必须重跑这一步。
2. **无缓存与缓存的 logits 一致**：`python train/test_cache_equivalence.py`
3. **缓存显存表**：`python train/benchmark_memory.py`
4. **长文检索（只评到训过的长度）**：
   ```bash
   python train/eval_long_context.py \
     --checkpoint /data/mini-k3/checkpoints/long-context-4096 \
     --tokenizer_model /data/mini-k3/data/tokenizer --lengths 2048 4096
   ```
   `--checkpoint` 可以给运行根目录，会自动取最新 COMPLETE 的 step。超过训练长度需要加 `--allow-untrained-length`，结果标 UNTRAINED。
5. **manifest/shard 字节复核**（不授权训练）：
   ```bash
   python train/data/validate_manifest.py \
     --manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json
   python train/data/validate_manifest.py \
     --manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json
   ```
6. **200 步短跑，通过后再开 10B**。短跑目录不作为 10B 的恢复点。四张卡被占用时不要启动。
   ```bash
   torchrun --standalone --nproc_per_node=4 train/train.py \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json \
     --checkpoint_dir /data/mini-k3/checkpoints/pretrain-short-2048 \
     --total_steps 200 --schedule_total_steps 38147 --save_interval 200 --log_interval 10
   ```
   合格标准见文首。合格之后跑 10B：
   ```bash
   torchrun --standalone --nproc_per_node=4 train/train.py \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json \
     --checkpoint_dir /data/mini-k3/checkpoints \
     --total_steps 38147 --save_interval 1000
   ```
   最后一个存档是 `/data/mini-k3/checkpoints/step_038146`。
7. **长上下文第一段，只在 10B 结束之后**。不要放进短跑，也不要插进 10B 中间。CSA2 没有 4096 窗口，选 4096 只是第一次把长度翻倍。学习率用 \(6\times10^{-5}\)，检查点目录分开。
   ```bash
   torchrun --standalone --nproc_per_node=4 train/train.py \
     --sequence-length 4096 --peak-lr 6e-5 \
     --init-checkpoint /data/mini-k3/checkpoints/step_038146 \
     --checkpoint_dir /data/mini-k3/checkpoints/long-context-4096 \
     --total_steps 500 --save_interval 100 \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json
   ```
   这一段自身的 WSD：预热 10 步，从第 425 步起衰减，并切到衰减配比。最后一个存档是 `long-context-4096/step_000499`。
   
   只有 4096 段的 lm loss 没有明显高于 2048 结束时，而且显存放得下，才开 8192，再开 16384。每段换新目录，用上一段最后的 step 目录作 `--init-checkpoint`。超过 16384 还要加 `--allow-long-sequence`。1,048,576 不排进训练。每段结束只评到刚训过的长度（见第 4 条）。SFT、DPO、PPO 用决定保留的那一档检查点。
8. **CED 不在这条主线上**。主模型 2048 预训练和长上下文的决定都定了之后才单独开，权重不能载入 Mini K3。
   ```bash
   torchrun --standalone --nproc_per_node=4 train/train.py --model ced \
     --sequence-length 2048 --total_steps 200 \
     --checkpoint_dir /data/mini-k3/checkpoints/ced-2048 \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json
   ```
9. **从中断检查点恢复**。运行签名（含 config 哈希）不一致时会拒绝：
   ```bash
   torchrun --standalone --nproc_per_node=4 train/train.py --resume \
     --data_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json \
     --validation_manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json \
     --checkpoint_dir /data/mini-k3/checkpoints
   ```
10. **评估检查点**（正式结果是验证集 lm loss，不是一句样例）：
    ```bash
    python train/evaluate.py --checkpoint /data/mini-k3/checkpoints/step_038146 \
      --manifest /data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json \
      --batches 32 --sequence-length 2048
    ```
    有题目文件时，另加 `--problems questions.jsonl --tokenizer_model /data/mini-k3/data/tokenizer`。`--sanity` 只检查打分代码。
11. **交互式对话**：
    ```bash
    python train/chat.py --checkpoint /data/mini-k3/checkpoints/step_038146 --tokenizer_model /data/mini-k3/data/tokenizer
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
   - 历史 12 层原型的模型架构 Smoke Test：1.02B 参数量核算准确，Step 0 初始损失 12.0988，优化器 2 步反向传播与 MoE 路由分发均正常；该原型不再作为当前 13 层模型的 readiness 证据。
   - 1M Cache 等价性验证 (`train/test_cache_equivalence.py`)：100 万长位置 RoPE 酉圆旋转稳定性误差 $< 1.19 \times 10^{-7}$，无 cache 与 cache logits 最大绝对误差 $1.67 \times 10^{-6} < 10^{-4}$，逐 token 贪婪解码完全一致。
   - KDA 运行时回归 (`train/test_kda_runtime.py`)：状态精度与分块等价完全一致。
   - 显存基准 (`train/benchmark_memory.py`)：1M 上下文 MLA 滑动窗口显存占用严格保持 $O(1)$ 边界。
   - 数据处理单元测试套件 (`pytest train/data/test_*.py`)：38 项单元测试全部通过。

### 2026-09-26 supplement-v2 全量重跑收官、10B 配比达标与终态审计状态

截至 2026-09-24 22:15，隔离增补流水线 `supplement-v2`（运行根目录 `/data/mini-k3/data/prepared-v2-supplement-v2`，控制器 PID 51070）已顺利跑完所有大算力处理与 Tokenizer 编码阶段：

1. **各阶段执行进度与收官时间戳**：
   - `canonical` & `cleaned`：`code-python` 成功整合 CodeSearchNet 455,243 条 Python 函数，清洗后保留 1,438,545 条样本（其余 10 来源重绑通过）。11/11 归档通过。
   - `global_exact_dedup`：全局精确去重于 09-22 22:13 完成，全库保留 18,007,744 篇文档。11/11 归档通过。
   - `global_near_dedup`：全局 MinHash-LSH 近重复去重于 09-24 12:08 全部完成。`code-python` 滤除近重复 45,328 篇（保留 1,228,755 篇）；`fineweb-edu` 滤除 235,709 篇（保留 4,931,516 篇）。11/11 归档通过。
   - `decontaminated`：13-gram 评测集基准去污染于 09-24 18:55 全部完成，11 来源报告统一绑定基准索引。11/11 归档通过。
   - `tokenization`：多线程 Tokenize 编码于 09-24 22:15 全部完成，生成完整 uint32 shards 与 document ledgers。11/11 归档通过。

2. **10B 预训练配比核验（7/7 来源全部达标，总池 28.21B Tokens）**：
   - **`code-python`**：实测可用 train tokens 达 **1,288,200,254**（对比 10B WSD 目标 1,225,000,878，**净盈余 +63,199,376 tokens**，彻底填平此前 44.1M 缺口）；
   - `fineweb-edu`：5,070,590,896 tokens（盈余 +795.6M）；
   - `chinese-fineweb-edu`：1,439,726,521 tokens（盈余 +14.7M）；
   - `cosmopedia`：1,689,844,133 tokens（盈余 +764.8M）；
   - `dolma-body`：9,008,293,636 tokens（盈余 +8,008.3M）；
   - `finemath`：5,237,079,424 tokens（盈余 +4,632.1M）；
   - `open-web-math`：4,480,775,152 tokens（盈余 +3,935.8M）；
   - **预训练 Token 合计**：**28,214,510,016 (28.21 B)**，在严格遵循固定混合比例前提下，纯净无重复采样完全覆盖 10B 预训练目标。

3. **终态 Manifest 审计与收敛修复定位**：
   - 09-24 22:15 流水线推进至 `finalize_v2.py` 时触发 `ValueError: Train/validation group overlap`。
   - 经对全库全部 11 个数据源、共计 **16,097,968 个 `split_group`** 进行哈希全量扫描：除 `openassistant` 存在 1 处跨 split 重叠外，其余 10 个数据源（包含全部预训练语料）均为 **0 冲突**。
   - 根因：`openassistant` 某对话树（`bfe63f8ebe9065b57ad3b71b1ad7e22cea8a12d59e99a72e9979024af633f914`）中，行 34140 因内容哈希命中官方验证集被标记为 `validation`，而同组的其余 3 条分支（行 34141~34143）因 `group_key` 未级联更新至 `holdout` 被误分入 `train`。
   - 解决方案：修复 `dedup_v2.py` 中 `holdout` 组判定，将同对话树全量划入 `validation`，重新编码 `openassistant`（仅 5.3 万条，耗时约 30 秒）后重跑 `finalize_v2.py` 即可生成正式 manifests 与 `AUDIT.json`。

4. **服务器算力与阶段报告归档**：
   - 硬件就绪：4× Tesla V100-SXM2-32GB 当前全部处于空闲就绪状态（0% 占用，显存 4MiB），磁盘剩余 4.2 TB。
   - 阶段报告归档：远端阶段归档保存在 `/data/mini-k3/project/train/reports/background/1790072703778497846-51070/`，小型元数据报告已完整同步到本地 `train/reports/background/1790072703778497846-51070/`。

### 2026-09-26 OpenAssistant 分组收敛后，supplement-v2 终审通过

09-24 的终审没有生成新 manifest：`prepared-v2-supplement-v2/manifests/` 当时是 v1 的硬链接，代码 token 仍记为 1,180,867,070。2026-09-26 只修这一处，没有重跑全局去重，也没有启动训练。

1. `dedup_v2.py` 增加 `promote_content_holdout_groups`：官方验证文本出现在另一段对话里时，整段对话进入 holdout，不再只标中那一条。回归测试 `test_validation_text_promotes_the_whole_conversation` 通过。已有 v2 去重库不重跑；这条规则保证以后的新库不会再拆开同一 `split_group`。
2. `repair_openassistant_split.py` 把会话 `bfe63f8ebe9065b57ad3b71b1ad7e22cea8a12d59e99a72e9979024af633f914` 里 3 条训练集分支改入验证集，分别写回 deduped、near-deduped、decontaminated，并刷新脚本哈希链。其余 10 个来源的正文没有改。
3. OpenAssistant 重新编码：训练 50,266 条 / 17,380,419 token，验证 3,233 条 / 1,160,229 token。旧编码目录保留为 `tokenized/openassistant-before-split-repair-20260926`。
4. `finalize_v2.py` 11/11 通过。`AUDIT.json` 状态 `passed`，`train_validation_overlap` 为 0，预训练训练 token **28,214,510,016**。`pretrain_stable.json` 的 567 个分片全部指向 `prepared-v2-supplement-v2`，其中 code-python 为 **1,288,200,254** token、26 个分片。该 manifest 与 v1 不再是同一 inode。
5. 覆盖率 `reports/supplement-v2/coverage.json` 状态 `sufficient_fixed_mix`，缺口为空。固定配比不重复最多到 **10,103,344,007** token，绑住上限的是中文网页（1.440B 对需求 1.425B）。10B 目标落在这个上限之内。
6. 真实数据 smoke（64 token、完整参数、两步）状态 `passed`。各来源初始 loss 12.00–12.19，第二步 11.58。这仍不是四卡 2048 短跑。
7. 训练入口和本节启动命令改为 `prepared-v2-supplement-v2` 的 `pretrain_stable.json` 与 `validation.json`。

### 2026-09-26 数据与脚本复核

复核没有重跑去重，也没有启动训练。

1. 服务器上对 v2 的 11 个来源重新执行 `verify_stage_chain`（到 tokenized）、`verify_source_review` 和 benchmark 索引校验，全部通过。OpenAssistant 去污染正文 53,499 条的 `content_sha256` 与正文重算一致，训练 50,266 / 验证 3,233，没有残留的跨 split 分组；分词索引的划分与正文一致。被改写的那 4 条都在验证集。
2. 精确去重和近去重的 SQLite 仍记录修复前的 OpenAssistant 分片哈希，近去重的输入身份也不再等于当前 `deduped/COMPLETE.json`。训练不读这两个库。再次执行 `dedup_v2.py` 或 `near_dedup_v2.py` 会在改写产物之前因哈希不一致而停止。去污染脚本在已有 `COMPLETE.json` 时直接拒绝重跑。
3. 加载器原来按分片下标对 4 取模。中文网页和 Python 的分片大多是 5,000 万 token，两张卡因此只有 3.50 亿和 3.00 亿，低于本卡在 10B 配比下要读的 3.5625 亿和 3.0625 亿，读完后会从分片开头再读。全局覆盖率看不出这个缺口。`loader.py` 改为每个分片按 token 切成 4 段互不重叠的区间，各卡读各的区间。按现有分片大小，四卡各自都能达到配比所需，不再靠整片取模。
4. 训练代码补上三处已经写了开关、但行为不对的实现：MLA 滑动窗口按每个 query 保留最近 4096 个 token（2048 训练长度下仍是整段因果注意力）；MoE 的两个共享专家分开计算再相加，不再合成一个两倍宽的 MLP；`activation_checkpointing` 会包住注意力前向。CED 的 RoPE 与 \(N(0, 0.02)\) 初始化是在独立文件里先修好的。
   2026-09-27 更正：`train.py --model ced` 仍是单独的稠密对照。主干里的因果编码器是前 8 层，不是这个开关。同日的结构对齐见文首。参数量为 1,150,739,900。正式 10B 从零训练。
5. 四卡训练循环原先会让各卡各写一份 `model.pt`、用本卡 loss 单独决定是否跳步，并且验证只在存档时抽 1 条、还不做卡间平均。现在只由 rank 0 写模型权重，跳步用四卡平均 loss，非有限梯度会降低 GradScaler，验证每 500 步取 4 个 batch 的全局平均。第 0 步检查读完会把数据游标放回去，不丢掉第一批 token。检查点读取显式关闭 `weights_only`，否则 PyTorch 2.6 之后读不回优化器和随机数，中断后无法续跑。对齐脚本按 `sequence_length` 保留序列尾部，奖励和价值不再把 token 0 当成填充。`evaluate.py` 对 `--problems` JSONL 计算准确率；不带该文件时只允许显式的 `--sanity`，不能把一句样例当成评测结果。2026-09-27 确定介入顺序：200 步短跑和 38,147 步 10B 都保持 2048、Mini K3、峰值学习率 \(6\times10^{-4}\)。长上下文从 10B 检查点之后才开始，第一段是 4096、学习率 \(6\times10^{-5}\)；8192 和 16384 只在前一段 loss 和显存都稳时才开。CED 放在主线和长上下文决定之后，单独占卡。超过 16384 需要 `--allow-long-sequence`。不设学习率就加长序列会被拒绝。

> 2026-10-03 更正（已被文首“第一次短跑两次 OOM：根因与整改”取代）：第 4 条的 MLA 滑动窗口已删除，`--attention-window` 现在直接报错，注意力是 CSA2；第 5 条的“每 500 步取 4 个 batch”改为固定验证切片 `--validation_batches 8`，按 `lm_loss` 选 best；优化器/GradScaler 只由 rank 0 保存一份。以文首为准。

### 2026-10-01 训练前代码与数据门禁复核（未启动正式训练）

> 2026-10-03 更正：下文第 3 条的单卡 smoke 与第 6 条的 cache 测试针对旧模型代码。模型代码已改（MoE、CSA2、KDA、Muon），`SMOKE.json` 现在绑定 `model_code_sha256`，旧 smoke 失效，必须重跑 `smoke_from_manifest.py`。第 5 条的运行签名已加入配置 sha256 等字段，旧检查点不能续跑。第 7 条提到的 `validate_manifest.py` 本地已删除，服务器上的残留文件需要清理。

本次修复对应当前 v2 数据和 13 层、1,150,739,900 总参数配置；所有改动先同步到 v100，再做只读或小张量验证：

1. 数据门禁 `train/data/check_training_readiness.py` 现在同时校验 schema-v2、`<u4`、tokenizer fingerprint、每个 shard 的字节数/总 token、`AUDIT.json` 的 manifest 哈希与零重叠、`SMOKE.json` 的审计绑定、固定配比覆盖报告和目标 token 数。旧控制器的失败状态由 `reconcile_pipeline_status.py` 在保留旧状态哈希的前提下原子重写为 `reconciled_audited_finalization`；没有改 shard、没有下载。
2. v100 readiness 结果为 `ready`：训练 manifest SHA256 `d2cce19c...1b7ea`，验证 manifest SHA256 `541136c5...29c7`，审计 SHA256 `9b99af75...2632`，固定配比覆盖 `sufficient_fixed_mix`，10,000,007,168 token 目标无缺口；训练池 28,214,510,016 token。
3. 修复了真实 smoke 的 GradScaler 顺序（先 `unscale_` 再裁剪）、mHC 初始读出分布无梯度、KDA chunk future `exp` 溢出、Engram 的当前 token 哈希遗漏、数据加载器耗尽后静默复用、默认 FP4 cache 与默认 torch.compile 的未验证路径。v100 单卡真实七源两步 full-model smoke 通过：13 层参数签名与配置一致，loss/梯度/优化器有限，峰值显存 15,136,763,904 bytes（序列 64；不外推 2048 或 1M）。
4. 对齐入口增加 `alignment.json` 的来源、文件大小、SHA256、tokenizer fingerprint 和模式检查；偏好样本裁剪保留同一个 prompt 边界；RM checkpoint 可以正确加载 `backbone.* + score.*` 格式。alignment manifest 的 openassistant SFT 文件已在 v100 通过绑定校验。
5. 检查点现在保存/恢复 GradScaler、训练/验证游标和运行签名（模型、序列、学习率、world size、manifest 哈希、参数签名），恢复前先拒绝不一致实验。数据不足或 shard 读完会硬失败，不再生成合成 token。
6. cache 等价性小配置测试在 v100 通过（最大 logits 差 `4.62e-6`，贪心 token 一致）；这只是实现回归。`eval_long_context.py` 现在必须提供已训练 checkpoint，`test_cache_equivalence.py` 和 `benchmark_memory.py` 的输出也明确不把小模型/理论窗口当成 1M 能力。按本文件第 8 条，尚无训练 checkpoint，因此 1M 长文档能力、长上下文 loss 和正式显存基准仍未完成。
7. `smoke_test.py` 已明确标为随机 token 的架构自检，只检查初始化、反向传播和优化器，不再输出“可以训练”；真实语料门禁只能由 `smoke_from_manifest.py` 与 `readiness.py` 共同给出。`validate_manifest.py` 也降级为仅 shard 字节检查，不能替代完整 readiness。
8. 长序列入口现在强制 `--init-checkpoint`，并在续训时显式提供较低的 `--peak-lr`；没有 2048 检查点时，4096/8192/16384 以及更长序列会直接拒绝。200 步短跑只要求审计覆盖量不低于本次预算，不再错误要求 coverage 的 10B 目标等于短跑 token 数。
9. `evaluate.py --manifest` 先验证 validation manifest 的 AUDIT 哈希、零重叠、tokenizer fingerprint 和每个 shard 的字节元数据；命令仍是只读评估，验证集不再接受任意可读 JSON。

正式 200 步短跑、10B 预训练及其长上下文延续均仍需用户明确启动；本次复核没有执行这些任务。
