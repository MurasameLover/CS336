import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    Embedding 层将token_ids映射为embedding向量,形状为(vocab_size, d_model).
    (batch_size, sequence_length) -> (batch_size, sequence_length, d_model)
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype))

        """ 初始化权重, 服从N(0,1), 截断到[-3,3] """
        # mean表示均值， std表示标准差， a表示截断下限，b表示截断上限
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x 形状为 (batch_size, sequence_length)
        x 为 token_ids

        self.weight 的形状是 (vocab_size, d_model)，可以想象成一个表格：

                    第0列  第1列  ...  第d_model-1列
        vocab[0]:   0.1    0.2   ...    0.5
        vocab[1]:   0.3   -0.1   ...    0.7
        vocab[2]:   0.0    0.4   ...    -0.2
        ...

        当你用整数张量做索引 weight[token_ids],PyTorch 会把 token_ids 里的每个整数，替换成 weight 中对应行的向量：

        token_ids = [
                        [1, 2, 3],
                        [4, 5, 6]
        ]

        weight[token_ids] = [
            [weight[1], weight[2], weight[3]],    # 第一行: token 1,2,3 的嵌入
            [weight[4], weight[5], weight[6]],    # 第二行: token 4,5,6 的嵌入
        ]

        """
        return self.weight[x]