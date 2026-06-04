import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    

    """
    相关语法：

    addcdiv_(tensor1, tensor2, value) → self + value × tensor1 / tensor2

    add_(x, alpha=a) 是 self + x·α 
    """

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']
                beta1, beta2 = group['betas']

                #  α_t = α * √(1-β₂ᵗ) / (1-β₁ᵗ)
                lr_t = group['lr'] * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

                # Weight decay: θ -= α·λ·θ
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # 更新动量
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                # 动量更新: θ -= α_t · m / (√v + ε)
                p.data.addcdiv_(m, v.sqrt().add_(group['eps']), value=-lr_t)
