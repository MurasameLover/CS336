# CS336 — Transformers (个人作业仓库)

这是我的 **Stanford CS336** 课程个人作业仓库，包含大语言模型与 Transformer 架构相关的五次作业实现。

## 作业列表

| # | 主题 | 核心内容 |
|---|------|---------|
| 1 | **Basics** — BPE tokenizer、Transformer 模块、训练循环 | BPE 分词、注意力机制、残差连接、AdamW、学习率调度 |
| 2 | **Systems** — 高效算子、分布式训练、混合精度 | Triton 算子、Flash Attention、FSDP/DDP、`torch.compile`、bf16/fp8 |
| 3 | **Scaling** — 缩放定律、损失预测、基础设施 | 计算最优训练、数据筛选、大规模启动脚本 |
| 4 | **Data** — 数据管道、去重、过滤 | 质量过滤、去重（MinHash、精确匹配）、混合策略 |
| 5 | **Alignment** — RLHF、DPO、偏好优化 | 奖励建模、PPO、DPO、拒绝采样 |

每次作业均为 **独立实现**——按照课程讲义从零编写 tokenizer、模型模块、优化器和训练基础设施。

技术栈：Python、PyTorch、uv（环境管理）。

---

*个人课程作业仓库，仅供学习参考。*
