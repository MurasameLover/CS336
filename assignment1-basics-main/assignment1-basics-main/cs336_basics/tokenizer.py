import regex as re
from typing import Iterator

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab      # token id -> bytes
        self._bytes_to_id = {v: k for k, v in vocab.items()}    # bytes -> token id
        self.merges = merges
        self.special_tokens = special_tokens or []

    # 解码器简单，先实现解码器
    """将 token ID 序列解码回文本"""
    def decode(self, ids: list[int]) -> str:
        sequence_bytes = b"".join([self.vocab[id] for id in ids])
        return sequence_bytes.decode("utf-8", errors="replace")

    """ 
    开始实现encode
    
    先实现“按special_token切分”和“按merges合并token”
    再实现“将输入文本编码成token id序列“
    再实现流式编码,节省内存
    """
    # “按special_token切分”
    @staticmethod   #Python 会把 self 隐式传进来作为第一个参数，所以 text 收到的其实是 self，会报错。所以加上 @staticmethod
    def _split_by_special_tokens(text: str, special_tokens: list[str]) -> Iterator[str]:
        """按 special token 切分，每个 special token 作为独立片段保留"""
        if not special_tokens:
            yield text
            return
        
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_tokens]
        pattern = f"({'|'.join(escaped)})"
        for part in re.split(pattern, text):
            if part:    #过滤空字符串
                yield part
        
    # “按merges合并token”
    @staticmethod
    def _merge_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """将 word 中所有相邻的 (pair[0], pair[1]) 合并为一个 token"""
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                result.append(pair[0] + pair[1])
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)

    # 开始实现encode
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        ids = []    # 存储编码后的token id序列
        for segment in self._split_by_special_tokens(text, self.special_tokens):
            if segment in self.special_tokens:
                ids.append(self._bytes_to_id[segment.encode('utf-8')])
                continue

            for match in re.finditer(PAT, segment):
                # 取出预分词后的word，例如“hello”
                word_str = match.group(0)
                # 将word转换为byte tuple，例如（b'l', b'o', b'w', b'e', b'r'）
                word_tuple = tuple(bytes([b]) for b in word_str.encode('utf-8'))
                # 按顺序遍历merges，合并相邻的pair
                for best_pair in self.merges:
                    word_tuple = self._merge_pair(word_tuple, best_pair)
                # 将合并后的文本序列转换为token id
                for token in word_tuple:
                    ids.append(self._bytes_to_id[token])
        
        return ids

    """ 实现encode之后,为了节省内存,进一步升级为流式编码 """
    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[list[int]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        buffer = ""
        for piece in iterable:
            buffer += piece

            last_safe_index = 0
            for m in re.finditer(PAT, buffer):
                if m.end() == len(buffer):
                    break
                last_safe_index = m.end()
            
            if last_safe_index > 0:
                yield from self.encode(buffer[:last_safe_index])
                buffer = buffer[last_safe_index:]

        if buffer:
            yield from self.encode(buffer)