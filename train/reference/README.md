# 预训练一个迷你 Kimi K3

一个围绕小规模语言模型预训练的技术阅读项目，涵盖模型设计、数据处理、训练流程、性能优化和评测，共 30 章。点击下方章节即可直接阅读，图片、公式和注释均随正文展示。

[使用与部署说明](docs/guide/README.md)

## 从这里开始

- [01. 我们实际训练了什么](docs/01-what-we-actually-trained/README.md)
- [02. 怎样的训练才达到实验室标准](docs/02-what-makes-a-run-lab-grade/README.md)
- [03. 如何阅读本书](docs/03-how-to-read-this-book/README.md)

## 模型

- [04. 逐行解读 K3 的配置](docs/04-reading-k3s-config/README.md)
- [05. 诚实地缩小模型规模](docs/05-scaling-it-down-without-lying/README.md)
- [06. 注意力层组合：九层 KDA 与三层 MLA](docs/06-the-attention-stack/README.md)
- [07. 真正执行路由的混合专家模型](docs/07-the-moe-that-actually-routes/README.md)
- [08. 统计参数，并校验统计方法](docs/08-counting-the-parameters/README.md)

## 数据

- [09. 构建训练语料](docs/09-assembling-the-corpus/README.md)
- [10. 去污染：让评测数字有据可依的报告](docs/10-decontamination/README.md)
- [11. 当过滤器匹配到的只是噪声](docs/11-when-a-filter-fires-on-nothing/README.md)
- [12. 我们没有重新训练的分词器](docs/12-the-tokenizer/README.md)
- [13. 数据混合，以及险些让训练只使用单一来源的缺陷](docs/13-the-mix-and-the-manifest-bug/README.md)

## 让模型训练起来

- [14. 发布的代码无法直接训练](docs/14-the-released-code-cannot-train/README.md)
- [15. 训练循环](docs/15-the-training-loop/README.md)
- [16. 专家坍塌，以及未能发现它的指标](docs/16-expert-collapse/README.md)
- [17. 失去 GPU 后仍能恢复的检查点](docs/17-checkpointing/README.md)
- [18. 监控看不见的训练过程](docs/18-watching-a-run-you-cannot-see/README.md)

## 让训练更快

- [19. 什么是 MFU，以及它为何决定计算资源的兑换率](docs/19-what-mfu-is/README.md)
- [20. 专家循环：工作量增加八倍，耗时却不变](docs/20-the-expert-loop/README.md)
- [21. 六个没有奏效的想法](docs/21-six-ideas-that-did-not-work/README.md)
- [22. 性能剖析：只有 12% 的时间用于算术运算](docs/22-the-profile/README.md)
- [23. 两个奏效的内核，以及一个失败的内核](docs/23-two-kernels-that-worked/README.md)
- [24. FP8，以及 MFU 反而更低的大 GPU](docs/24-fp8-and-the-bigger-gpu/README.md)
- [25. 静默回退](docs/25-the-silent-fallback/README.md)

## 不止一张 GPU

- [26. 跨 GPU 分片模型](docs/26-sharding-the-model/README.md)
- [27. 三个不会导致崩溃的分布式缺陷](docs/27-three-bugs-that-do-not-crash/README.md)

## 训练有效吗

- [28. 从头到尾审视损失曲线](docs/28-the-loss-curve/README.md)
- [29. 沿训练 token 进度评测，而非只看终点](docs/29-benchmarks-across-tokens/README.md)
- [30. 252 美元能买到什么](docs/30-what-252-dollars-buys/README.md)
