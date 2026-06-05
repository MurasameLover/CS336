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
| **训练** | `cross_entropy.py` | 交叉熵损失（手动实现，含数值稳定性优化） |
| | `AdamW.py` | AdamW 优化器（手写，含权重衰减、动量更新） |
| | `gradient_clipping.py` | 梯度裁剪（按全局 L2 范数缩放） |
| | `learning_rate_schedule.py` | Warmup + 余弦退火学习率调度 |
| | `data_loading.py` | 从内存映射 numpy 数组随机采样训练/验证批次 |
| | `checkpoint.py` | 模型状态字典保存与恢复 |
| | `train.py` | 完整训练流水线（日志、验证、checkpoint、梯度裁剪、LR 调度） |

---

## 后续作业概览

| # | 主题 | 核心内容 |
|---|------|---------|
| 1 | **Basics** ✅ | BPE tokenizer、Transformer 模块、训练循环 |
| 2 | **Systems** | 高效算子、分布式训练、混合精度 |
| 3 | **Scaling** | 缩放定律、损失预测、基础设施 |
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
