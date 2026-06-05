"""
Softmax + 温度采样 + Top-p采样 + 生成文本

Softmax 带温度:
    softmax(v, τ)ᵢ = exp(vᵢ/τ) / Σⱼ exp(vⱼ/τ)
    τ → 0 时趋近 argmax（确定性），τ 大时更随机

在语言模型推理时，用模型输出的 logits 采样下一个 token，
循环往复直到遇到 <|endoftext|> 或达到最大长度。
"""

import torch
from .Softmax import softmax


def softmax_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """带温度的 softmax。"""
    return softmax(logits / temperature, dim=-1)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    从 logits 中采样下一个 token。

    top_p 是 nucleus sampling（核采样）的参数：
    只保留累积概率前 top_p 的 token，其余截断后重新归一化。
    """
    probs = softmax_with_temperature(logits, temperature)

    if top_p is not None:
        # 按概率降序排列
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)

        # 累积概率超过 top_p 的 token 被截断
        # 第一个超过 top_p 的 token 保留（减去 sorted_probs 本身）
        mask = cum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0

        # 重新归一化
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # 从截断后的分布采样
        next_token = sorted_probs.multinomial(1)
        # 映射回原始的 token ID
        next_token = sorted_indices.gather(-1, next_token)
    else:
        next_token = probs.multinomial(1)

    return next_token.squeeze(-1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    """
    给定 prompt，逐 token 生成文本，直到遇到 <|endoftext|>（如有）或达到 max_length。

    流程：
    ① 用 tokenizer 把 prompt 编码成 token ID 序列
    ② 模型前向传播 → logits
    ③ 取最后一位的 logits → softmax → 采样下一个 token
    ④ 追加到序列末尾
    ⑤ 重复 ②~④ 直到停止条件
    """
    model.eval()

    # 编码 prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)

    # 找到 end-of-sequence token 的 ID（如果有）
    eos_token_id = None
    if "<|endoftext|>" in tokenizer.special_tokens:
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    for _ in range(max_length):
        logits = model(input_ids)            # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]       # (vocab_size,) 取最后一个位置

        next_token = sample(next_logits, temperature, top_p)

        # 遇到 <|endoftext|> 则停止生成
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # 把新 token 追加到序列末尾
        input_ids = torch.cat(
            [input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1
        )

    # 解码回文本
    return tokenizer.decode(input_ids[0].tolist())

"""
使用示例：

from cs336_basics.generation import generate

# 加载训练好的模型和 tokenizer
model = TransformerLM(...)
model.load_state_dict(torch.load("checkpoint.pt")["model"])
model.to("cuda")

tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# 生成
text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_length=200,
    temperature=0.8,
    top_p=0.9,
    device="cuda",
)
print(text)
"""

