import numpy as np

"""
warm up + 余弦退火 学习率调度器
第一阶段：Warmup
    t<Tw​时
        学习率从 0 线性增长到lr_max
        lr_t = t / Tw * lr_max
第二阶段：Cosine decay
    t>=Tw && t<Tc 时
        学习率从 lr_max 平滑下降到lr_min
        lr_t = lr_min + 0.5 * (1 + cos(pi * (t - Tw)/(Tc - Tw))) * (lr_max - lr_min)
第三阶段：学习率保持不变
    t>=Tc 时
        lr_t = lr_min
"""

def get_lr_cosine_schedule(
    t,      # 当前迭代次数
    lr_max,
    lr_min,
    warm_up_steps,  # 预热步数
    cosine_stop_steps   # 第二阶段结束时的步数
):
    # 第一阶段
    if t < warm_up_steps:
        return t / warm_up_steps * lr_max

    # 第二阶段
    elif t >= warm_up_steps and t <= cosine_stop_steps:
        return lr_min + 0.5 * (1 + np.cos(np.pi * (t - warm_up_steps) / (cosine_stop_steps - warm_up_steps))) * (lr_max - lr_min)

    # 第三阶段
    else:
        return lr_min
