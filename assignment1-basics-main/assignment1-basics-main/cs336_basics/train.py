from .data_loading import get_batch
from .transformer_lm import TransformerLM
from .cross_entropy import cross_entropy_loss
from .gradient_clipping import gradient_clipping
from .AdamW import AdamW
from .learning_rate_schedule import get_lr_cosine_schedule
from .checkpoint import save_checkpoint, load_checkpoint

import torch
import torch.nn as nn
import numpy as np

# 配置超参数
class Config:
    batch_size = 16
    vocab_size = 10000  # 词表大小
    context_length = 256    # 上下文长度
    d_model = 512   # 隐藏层维度
    d_ff = 1344     # 前向传播隐藏层维度.(8/3×512=1365.3, 取近64倍数=1344)
    n_layers = 4    # 层数
    n_heads = 16    # 多头注意力的头数
    theta = 10000.0 # RoPE的缩放参数
    betas = (0.9, 0.95) # AdamW的 betas
    weight_decay = 0.01 # AdamW的权重衰减系数
    lr_max = 1e-4   # 最大学习率
    lr_min = 1e-5   # 最小学习率
    warm_up_steps = 500   # 预热步数
    cosine_stop_steps = 7000    # 第二阶段结束时的步数
    max_l2_norm = 1.0   # 梯度裁剪的阈值

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data/train.txt"    # 训练数据路径
    val_path = "data/val.txt"   # 验证数据路径
    checkpoint_path = "checkpoint.pt"    # 检查点路径
    steps = 10000   # 训练步数
    steps_per_checkpoint = 500      # 每500步保存一次检查点
    steps_per_log = 50      # 每50步打印一次损失    
    steps_per_eval = 500               # 每 500 步验证一次
    eval_iters = 100                   # 验证时取 100 个 batch 平均

def main():
    config = Config()

    """ 初始化 """
    # 初始化模型
    model = TransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.d_ff,
        config.theta
    )
    model.to(config.device)

    # 初始化优化器
    # AdamW中实际上还有一个eps参数，但是我使用了默认值，也可以自己配置
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr_max,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    """ 数据加载 """
    # 这里使用np.memmap 加载大数据
    train_data = np.memmap(config.data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(config.val_path, dtype=np.uint16, mode="r")

    """ Training Loop """
    model.train()
    total_loss = 0.0
    for step in range(config.steps):
        # 获取批次数据
        inputs, targets = get_batch(
            train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )

        # 前向传播 + 计算loss
        logits = model(inputs)  

        """
        当前的形状：
            logits: (batch_size, context_length, vocab_size)
            targets: (batch_size, context_length)
        我实现的cross_entropy_loss函数中:
            def cross_entropy_loss(logits: (N, vocab_size), targets: (N,))
        每个 (batch, seq) 位置都有一个正确的预测目标,要对所有 batch × seq 个位置求交叉熵然后平均
        - logits → (batch*seq, vocab_size)   view(-1, vocab_size)
        - targets → (batch*seq,)             view(-1)
        """
        loss = cross_entropy_loss(
            logits.view(-1, config.vocab_size),
            targets.view(-1)
        )

        # 反向传播
        optimizer.zero_grad()   # 优化器中的梯度需要清零，不然会累积
        loss.backward()
        total_loss += loss.item()
        
        # 梯度裁剪
        gradient_clipping(model.parameters(), max_l2_norm=config.max_l2_norm)

        # 更新学习率
        lr = get_lr_cosine_schedule(
            step,
            config.lr_max,
            config.lr_min,
            config.warm_up_steps,
            config.cosine_stop_steps
        )
        # 学习率更新后，需要在优化器中更新
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 优化器步进
        optimizer.step()

        # 保存checkpoint
        if (step + 1) % config.steps_per_checkpoint == 0:
            save_checkpoint(
                model,
                optimizer,
                step + 1,
                config.checkpoint_path
            )
            print(f"Saved checkpoint at step {step+1}")

        # 打印日志
        if (step + 1) % config.steps_per_log == 0:
            avg_loss = total_loss / config.steps_per_log
            print(f"Step {step+1} | Loss {avg_loss} | LR {lr}")
            total_loss = 0.0

        # 验证
        if (step + 1) % config.steps_per_eval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(config.eval_iters):
                    inputs, targets = get_batch(
                        val_data,
                        batch_size=config.batch_size,
                        context_length=config.context_length,
                        device=config.device,
                    )
                    logits = model(inputs)
                    loss = cross_entropy_loss(
                        logits.view(-1, config.vocab_size),
                        targets.view(-1)
                    )
                    val_loss += loss.item()
            avg_val_loss = val_loss / config.eval_iters
            print(f"Step {step+1} | Val Loss {avg_val_loss:.4f}")
            model.train()