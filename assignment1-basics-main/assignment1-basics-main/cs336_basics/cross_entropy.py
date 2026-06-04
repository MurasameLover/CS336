import torch

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
