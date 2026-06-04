import torch

"""
loss计算公式：
L = m + log(sum(exp(o_j - m))) - o_y
m = max(o_j)        o_y是真实标签对应的logits

先用一个例子展示整个流程
batch_size = 2
vocab_size = 4
logits =
[
    [2.0, 1.0, 0.1, 0.5],
    [0.2, 3.0, 1.0, 2.0]
]
shape = (2, 4)

targets =
[
    0,
    3
]
shape = (2,)
表示：
第一个样本正确类别是0
第二个样本正确类别是3

Step1: 减去每一行最大的值，防止exp(x) 溢出
x_max =tensor([
            [2.0],
            [3.0]
        ])
logits - logits.max(dim=-1, keepdim=True).values
logits - x_max = tensor([
            [ 0.0, -1.0, -1.9, -1.5],
            [-2.8,  0.0, -2.0, -1.0]
        ])

Step2: 计算exp
torch.exp(logits - x_max)
tensor([
    [1.0000, 0.3679, 0.1496, 0.2231],
    [0.0608, 1.0000, 0.1353, 0.3679]
])

Step3: 对exp求和
sum_exp =
[
    1.7406,
    1.5640
]
shape:(2,)   这里没有keepdim = True

Step4: 计算log
[
    0.5544,
    0.4474
]

Step5： x_max.squeeze(-1)
本来：
x_max =
[
    [2.0],
    [3.0]
]
转换为：
[
    2.0,
    3.0
]

Step6：相加
log_sum_exp = x_max.squeeze(-1) + log
[
    2.5544,
    3.4474
]

Step7：取真实标签
logits[range(batch_size), targets]
其中range(batch_size) = [0,1]; targets = [0,3]
所以索引的是 logits[0,0] 和 logits[1,3]
即 2.0 和 2.0
得到tensor([2.0, 2.0])

Step8： 计算loss
loss = -logits[range(batch_size), targets] + log_sum_exp
第一行： -2.0 + 2.5544 = 0.5544
第二行： -2.0 + 3.4474 = 1.4474
得到loss = tensor([0.5544, 1.4474])

Step9：求平均
loss.mean()
得到0.6922

"""

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    batch_size = logits.shape[0]
    x_max = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = x_max.squeeze(-1) + torch.log(
        torch.exp(logits - x_max).sum(dim=-1)
    )
    loss = -logits[range(batch_size), targets] + log_sum_exp
    return loss.mean()
