from .pretokenization_example import find_chunk_boundaries

import regex as re
from typing import Iterator


#按照special_tokens进行切分
def split_by_special_tokens(
    text: str,
    special_tokens: list[str]
) -> Iterator[str]:
    """按 special token 切分文本,special token 本身作为独立片段保留。"""
    if not special_tokens:
        yield text
        return 
    # 长 token 优先匹配（例如 "<|endoftext|><|endoftext|>" 优先于 "<|endoftext|>"）
    sorted_tokens = sorted(special_tokens, key=len, reverse=True) #reverse是降序
    escaped_tokens = [re.escape(token) for token in sorted_tokens] #将special_tokens中的部分字符转义，避免被翻译为正则语法
    pattern = f"({'|'.join(escaped_tokens)})"

    """
    re.split()有两种模式：
        一、不捕获——分隔符被丢弃
        re.split(r"<\|endoftext\|>", "Hello<|endoftext|>world")
        # → ['Hello', 'world']     ← <|endoftext|> 没了！

        二、用捕获组——分隔符保留在结果中
        re.split(r"(<\|endoftext\|>)", "Hello<|endoftext|>world")
        # → ['Hello', '<|endoftext|>', 'world']   ← 完美

    所以pattern = f"({'|'.join(escaped_tokens)})"构建一个带()的捕获组,可以在接下来的re.split()中保留special_tokens
    """
    for part in re.split(pattern, text):
        if part:    #过滤空字符串
            yield part


# merge 函数
def merge_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """将 word 中所有相邻的 (pair[0], pair[1]) 合并为一个 token"""
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(pair[0] + pair[1])  # 合并
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result) 


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

    """初始化词表vocab: (256个字节+special_tokens)"""
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    """
    先根据pretokenization_example.py中的示例,将待处理的文本切分成多个chunk,并行处理加快pretokenization的速度
    再在每个chunk中“按照special_tokens切分 -> 非special_tokens用GPT-2的正则切分 -> 统计频率 -> 合并”
    """

    from collections import Counter
    word_freqs = Counter()  # 词频统计

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            for segment in split_by_special_tokens(chunk, special_tokens):
                if segment in special_tokens:   
                    continue    #special_tokens不参与频率统计
                else:
                    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                    for match in re.finditer(PAT, segment):
                        word_freqs[match.group(0)] += 1
    """
    将词频统计结果转换成如下形式:
    {
        (b'l', b'o', b'w'): 5,
        (b'l', b'o', b'w', b'e', b'r'): 2,
        (b'w', b'i', b'd', b'est'): 3,
        (b'n', b'e', b'w', b'est'): 6,
    }
    """
    word_freqs = {
        tuple(bytes([b]) for b in word.encode('utf-8')): freq 
        for word, freq in word_freqs.items()
    }

    """ 统计相邻pair的频率 """
    pair_freqs = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pair_freqs[pair] += freq

    """ 开始merge循环 """
    merges = []     #作业要求记录merge过程中新添加的pair
    while len(vocab) < vocab_size:
        # 找到频率最高的pair
        best_freq = max(pair_freqs.values())
        pairs = [pair for pair, freq in pair_freqs.items() if freq == best_freq]
        best_pair = max(pairs)

        # 记录merge
        merges.append(best_pair)

        # 新token加入vocab
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # # 在所有word中合并这个pair
        # new_word_freqs = Counter()
        # for word, freq in word_freqs.items():
        #     new_word = merge_pair(word, best_pair)
        #     new_word_freqs[new_word] += freq

        # # 更新词频统计
        # pair_freqs = Counter()
        # for word, freq in new_word_freqs.items():
        #     for i in range(len(word)-1):
        #         pair = (word[i], word[i+1])
        #         pair_freqs[pair] += freq
        # word_freqs = new_word_freqs

        #更新词频统计（增量更新法）
        new_word_freqs = Counter()
        for word, freq in word_freqs.items():
            new_word = merge_pair(word, best_pair)
            # 如果合并后word有变化,则更新pair_freqs和word_freqs
            if new_word != word:
                # 移除旧word对pair_freqs的贡献
                for i in range(len(word)-1):
                    pair_freqs[(word[i], word[i+1])] -= freq
                    if pair_freqs[(word[i], word[i+1])] <= 0:
                        del pair_freqs[(word[i], word[i+1])]
                # 添加新word对pair_freqs的贡献
                for i in range(len(new_word)-1):
                    pair_freqs[(new_word[i], new_word[i+1])] += freq
                # 更新word_freqs
                new_word_freqs[new_word] += freq
            # 如果合并后word没有变化,则直接更新word_freqs
            else:
                new_word_freqs[word] += freq
        word_freqs = new_word_freqs

    return vocab, merges
