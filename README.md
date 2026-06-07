# CS336 — Transformers (个人作业仓库)

这是我的 **Stanford CS336** 课程个人作业仓库，包含大语言模型与 Transformer 架构相关的五次作业实现。

---

## 作业 1：Basics — 从零构建 Transformer

从零实现 BPE tokenizer、Transformer 各模块、优化器、训练循环。所有模块均不依赖 PyTorch 内置的高级 API（如 `nn.MultiheadAttention`），完全手写以加深理解。

### 项目结构

```
cs336_basics/
├── tokenizer.py              # BPE tokenizer（从零实现）
├── train_bpe.py              # BPE 训练脚本
├── pretokenization_example.py # 预分词示例

├── linear.py                 # 线性层（手写 Linear）
├── Softmax.py                # Softmax 函数
├── RMSNorm.py                # RMS 层归一化
├── embedding.py              # Token Embedding 层
├── rope.py                   # 旋转位置编码 (RoPE)
├── swiglu.py                 # SwiGLU 前馈网络

├── attention.py              # Scaled Dot-Product Attention
├── MultiHeadSelfAttention.py # 多头自注意力（基础版）
├── MultiHeadSelfAttention_with_RoPE.py # 多头自注意力（带 RoPE）

├── transformer_block.py      # Transformer 块（Pre-LN：MHA + SwiGLU FFN）
├── transformer_lm.py         # 完整 Transformer 语言模型

├── cross_entropy.py          # 交叉熵损失函数
├── AdamW.py                  # AdamW 优化器（手写）
├── gradient_clipping.py      # 梯度裁剪（L2 范数）
├── learning_rate_schedule.py # 学习率调度（Warmup + 余弦退火）

├── data_loading.py           # 数据加载（从 numpy 数组采样批次）
├── checkpoint.py             # 模型检查点保存/加载
├── generation.py             # 文本生成（温度采样 + Top-p 采样）
└── train.py                  # 完整训练循环

tests/
├── adapters.py               # 测试适配器（将手写模块接入测试框架）
├── test_tokenizer.py         # Tokenizer 测试
├── test_train_bpe.py         # BPE 训练测试
├── test_model.py             # 模型测试
├── test_nn_utils.py          # 神经网络工具测试
├── test_optimizer.py         # 优化器测试
├── test_data.py              # 数据处理测试
├── test_serialization.py     # 序列化测试
├── common.py                 # 测试工具函数
└── conftest.py               # pytest 配置
```

### 各模块功能

| 类别 | 文件 | 功能 |
|------|------|------|
| **Tokenization** | `tokenizer.py` | 从零实现的 BPE 分词器，支持训练词表、编码、解码 |
| | `train_bpe.py` | BPE 训练流程（文本预处理 → 合并统计 → 保存词表） |
| **基础算子** | `linear.py` | 手写线性层 `y = xW^T` |
| | `Softmax.py` | 数值稳定的 Softmax 实现（减去最大值防止溢出） |
| | `RMSNorm.py` | RMS 层归一化 |
| **嵌入 & 位置编码** | `embedding.py` | Token Embedding（查表） |
| | `rope.py` | 旋转位置编码（复数旋转矩阵实现） |
| **注意力机制** | `attention.py` | 基础 Scaled Dot-Product Attention |
| | `MultiHeadSelfAttention.py` | 多头自注意力（QKV 投影 → 拆头 → 因果 mask → 合并 → 输出投影） |
| | `MultiHeadSelfAttention_with_RoPE.py` | 带 RoPE 的多头自注意力 |
| **模型架构** | `transformer_block.py` | Pre-LN Transformer 块（`RMSNorm → MHA → 残差 → RMSNorm → SwiGLU → 残差`）|
| | `transformer_lm.py` | 完整语言模型（`Embedding → N×Block → RMSNorm → LM Head`）|
| **推理** | `generation.py` | 文本生成（温度采样、Top-p 核采样、完整生成流水线） |

| **训练** | `cross_entropy.py` | 交叉熵损失（手动实现，含数值稳定性优化） |
| | `AdamW.py` | AdamW 优化器（手写，含权重衰减、动量更新） |
| | `gradient_clipping.py` | 梯度裁剪（按全局 L2 范数缩放） |
| | `learning_rate_schedule.py` | Warmup + 余弦退火学习率调度 |
| | `data_loading.py` | 从内存映射 numpy 数组随机采样训练/验证批次 |
| | `checkpoint.py` | 模型状态字典保存与恢复 |
| | `train.py` | 完整训练流水线（日志、验证、checkpoint、梯度裁剪、LR 调度） |

---

## 作业 3：Scaling — 缩放定律

基于作业 1 的自研 Transformer，在 Stanford 的训练 API 上进行大规模 IsoFLOP 实验，验证语言模型缩放定律。

### 项目结构

```
assignment3-scaling-main/
├── cs336_scaling/          # 核心框架包
│   ├── training/           #   训练基础设施（模型定义、数据加载、优化器、训练循环）
│   ├── api/                #   FastAPI 训练服务（提交/查询/管理实验）
│   ├── client.py           #   训练 API 客户端
│   ├── db/                 #   数据库模型（PostgreSQL ORM 表结构）
│   ├── schemas/            #   请求/响应 Pydantic 模型
│   ├── scheduler/          #   实验调度器（队列管理、任务分发）
│   ├── tokenized_data.py   #   分词数据处理工具
│   └── auth.py / budget.py #   认证与配额管理
│
├── workspace/myself_experiment/  # ★ 个人实验工作区
│   ├── model.py                  #   模型定义
│   ├── train.py / validate.py    #   训练与验证脚本
│   ├── config.py / data.py       #   配置与数据加载
│   │
│   ├── isoflops_runs/            #   IsoFLOP 实验运行结果（按配置分组 C0–C4）
│   │   ├── C0_XXS/ … C4_XL/     #   每组含 results.json（loss 曲线等指标）
│   │   └── ...
│   ├── validation_run/           #   验证运行结果（results.json + validation_results.json）
│   ├── figures/                  #   生成的图表（scaling_laws.pdf, isoflops_u_curves.pdf …）
│   │
│   ├── run_isoflops.py           #   IsoFLOP 实验批量运行脚本
│   ├── analyze.py                #   结果分析与缩放定律拟合
│   ├── extend_experiment.py      #   扩展实验脚本（补充 C3/C4 配置）
│   ├── isoflops_results.json     #   汇总的 IsoFLOP 结果
│   └── analysis_predictions.json #   分析的预测结果
│
├── examples/           # API 使用示例（Jupyter notebook）
├── scripts/            # 辅助脚本（数据下载等）
├── tests/              # 测试（API、调度器）
└── data/               # 分词后的训练数据
```

### 实验设计

- 在 **6 种模型规模**（XXS → XL）上运行 IsoFLOP 实验
- 固定计算预算下，扫描最佳 batch size 与学习率组合
- 拟合 **缩放定律曲线**，预测最优计算分配
- 扩展实验（C3/C4）验证更大大规模配置的泛化能力

---

## 后续作业概览

| # | 主题 | 核心内容 |
|---|------|---------|
| 1 | **Basics** ✅ | BPE tokenizer、Transformer 模块、训练循环 |
| 2 | **Systems** → | 高效算子、分布式训练、混合精度 |
| 3 | **Scaling** ↑ | 缩放定律、IsoFLOP 实验、损失预测 |
| 4 | **Data** | 数据管道、去重、过滤 |
| 5 | **Alignment** | RLHF、DPO、偏好优化 |

---

## 技术栈

- **语言**：Python 3.11+
- **框架**：PyTorch
- **环境管理**：uv
- **测试**：pytest

---

*个人课程作业仓库，仅供学习参考。*
