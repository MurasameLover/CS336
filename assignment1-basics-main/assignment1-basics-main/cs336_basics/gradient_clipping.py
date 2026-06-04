"""
在训练过程中，
有时会遇到产生非常大梯度（large gradients）的训练样本，
这可能导致训练变得不稳定,
于是在每次反向传播（backward）之后、优化器更新参数（optimizer step）之前，
限制梯度的范数（norm）大小

假设所有参数的梯度拼接起来形成梯度向量： g
计算其ℓ2范数： ∥ g ∥ 2
梯度的 L2 范数就是所有参数梯度的平方和的平方根：
    total_norm = √( Σ ||g_i||₂² )

情况1： 范数 < M (预设的最大范数)
    g′= g
情况2： 范数 ≥ M
    g′ = g * M / (g的范数 + ϵ)      这里ϵ取1e-6
"""

def gradient_clipping(parameters, max_l2_norm):
    # 计算所有参数的梯度范数
    # 先计算所有参数梯度的平方和，再开平方根
    total_l2 = 0.0
    for p in parameters:
        if p.grad is not None:
            total_l2 += (p.grad.data ** 2).sum().item()
            """
            这里计算l2范数有更简洁的写法，我上面的写法是最基础的“计算所有参数梯度的平方再求和”，最后开平方
            pytorch中p.grad.norm(p)可以计算 p 范数，p=2 即 L2 范数
            先通过p.grad.norm(2)得到一个参数的l2范数，再平方，全部参数求和后再开平方，更简洁。
             p.grad.norm(2) 直接计算 L2 范数 √(Σgᵢ²)，再 ** 2 就是 Σgᵢ²

            具体实现如下：
            total_sq = 0.0
            for p in parameters:
                if p.grad is not None:
                total_sq += p.grad.norm(2).item() ** 2
            total_norm = total_sq ** 0.5
            """

    total_l2_norm = total_l2 ** 0.5
    
    # 如果梯度大于阈值，则进行裁剪
    if total_l2_norm > max_l2_norm:
        scale = max_l2_norm / (total_l2_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)