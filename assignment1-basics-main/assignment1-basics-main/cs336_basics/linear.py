import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None,
        dtype: torch.dtype | None
    ):
        super().__init__()
