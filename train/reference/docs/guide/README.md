# 使用与部署说明

[← 返回章节目录](../../README.md)

本项目提供 30 章技术内容、配套插图、公式与代码示例，支持 Markdown 分章阅读和 HTML 网站浏览。

内容以一次小规模预训练工作日志为主线，讨论模型缩放、语料去污染、训练稳定性、性能优化、分布式训练和评测。仓库按**纯静态阅读资料**组织，可直接部署。

## 阅读入口

| 内容 | 文件 |
| --- | --- |
| GitHub 直接阅读 | [Markdown 章节目录](../../README.md) |
| 英文目录与分章阅读 | [网站首页](../../index.html) |
| 英文完整内容 | [英文全文 HTML](../../full_book.html) |

在 GitHub 中，通过根目录 README 进入各章的 README.md，即可直接阅读正文。图片可点击查看原始文件，表格采用原生 Markdown 语法，注释使用脚注链接。格式约定参见 [GitHub Markdown 说明](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)。

数学表达式使用 LaTeX：行内公式写在 `$...$` 中，独立推导放在 `$$...$$` 公式块中。GitHub 可直接渲染；使用其他 Markdown 阅读器时，请启用数学公式支持。参见 [GitHub 数学公式说明](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。

仓库中的英文 HTML 用于后续静态网站部署，中文章节通过 GitHub 的 Markdown 页面阅读。

## 内容结构

全书共 **30 章、179 张正文插图、39 张表格和 130 条原有注释**。

| 章节 | 主题 |
| --- | --- |
| 1–3 | 训练目标、实验规范与阅读方法 |
| 4–8 | K3 配置、模型缩放、KDA／MLA、混合专家与参数统计 |
| 9–13 | 语料构建、去污染、分词器与数据混合 |
| 14–18 | 训练代码修复、训练循环、专家坍塌、检查点与监控 |
| 19–25 | MFU、专家分发、性能剖析、融合内核、FP8 与静默回退 |
| 26–27 | 模型分片与分布式缺陷 |
| 28–30 | 损失曲线、过程评测与成本总结 |

## 阅读说明

- **图片**：179 张正文插图随章节展示，点击图片可查看原始文件和细节。
- **数学公式**：分式、上下标、科学计数法及独立推导采用数学排版；取近似值的结果使用近似等号。
- **程序代码**：配置项、文件路径和可运行代码保留代码格式。
- **表格与注释**：39 张表格使用原生 Markdown 表格，130 条注释可通过脚注链接往返查看。
- **数据说明**：部分存储量、检查点数量、耗时计算和性能占比存在待确认之处，已在相关章节单独说明。详见[第 17 章](../17-checkpointing/README.md)、[第 20 章](../20-the-expert-loop/README.md)及[第 22 章](../22-the-profile/README.md)。

## 本地产物

`output/` 用于存放本地导出文件，已加入 `.gitignore`，不随仓库上传。仓库中的章节阅读和图片显示均不依赖该目录。

## 本地预览

在项目根目录运行：

```bash
python3 -m http.server 8000
```

然后打开 [http://localhost:8000/](http://localhost:8000/)，可预览英文目录、分章与全文页面。中文 Markdown 请通过 GitHub 或支持 Markdown 的阅读器查看。

页面运行所需文件已包含在仓库中，无需安装 Python 包或 Node.js 依赖。英文页面引用了 Google Fonts；无法访问时会使用后备字体。

## 静态网站部署

部署目录选**仓库根目录**，首页为 `index.html`，构建命令留空。保留 `assets/`、`figures/`、`chapters/` 的相对位置，即可发布英文 HTML 阅读站。中文 Markdown 位于 `docs/`，在 GitHub 仓库页面中直接阅读。

在 GitHub Pages 中，可在仓库 **Settings → Pages** 选择 **Deploy from a branch**，指定存放这些文件的分支（例如 `main`）和 **`/(root)`**。仓库已包含 `.nojekyll`，用于直接发布静态文件。参见 [GitHub Pages 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)。

## 目录说明

```text
.
├── README.md                 # 简介与章节目录
├── docs/                     # 30 章 README 与使用说明
├── .gitignore                # 忽略系统文件和临时产物
├── .nojekyll                 # GitHub Pages 静态发布标记
├── index.html               # 阅读首页
├── full_book.html           # 英文完整内容
├── chapters/                # 30 个英文分章页面
├── assets/                  # 页面实际使用的 CSS 和 JavaScript
└── figures/                 # 正文插图与首页封面
```

维护正文时，请保持图片路径、公式、代码和章节导航完整；导出其他格式时，应使用同一版本的内容。

章节 README 和英文 HTML 通过相对路径使用 `figures/`；请将图片目录与正文一并保留。

## 来源与署名

- 原著：**Pretraining a Mini Kimi K3**
- 作者：**Dr. Raj Dandekar**
- 原站／机构：**Vizuara AI Labs**
- [原书入口](https://books.vizuara.ai/read/pretraining-a-mini-k3/what-we-actually-trained)

书籍内容、插图及原站资源的权利与许可条款以原作者／原站说明为准；请保留来源和署名。
