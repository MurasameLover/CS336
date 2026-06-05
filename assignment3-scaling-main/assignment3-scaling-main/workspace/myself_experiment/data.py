"""
Load Shakespeare dataset, train a BPE tokenizer, and provide tokenized data.
The raw text file is expected at data_cache/input.txt (downloaded separately).
"""

from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────
# Use the Gutenberg books corpus (larger than Shakespeare alone)
# Falls back to Shakespeare if the Gutenberg corpus is not available.
DATA_SOURCES = [
    # Primary: Gutenberg books corpus
    "big_text.txt",
    # Fallback: Shakespeare
    "input.txt",
]
VOCAB_SIZE = 5000


def load_source_text() -> str:
    """Load text from the best available source (Gutenberg corpus first)."""
    for fname in DATA_SOURCES:
        path = CACHE_DIR / fname
        if path.exists():
            print(f"Loading text from {path} ({path.stat().st_size:,} bytes)")
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No text source found in {CACHE_DIR}. "
        f"Place at least one of {DATA_SOURCES}"
    )


def train_bpe_tokenizer(text: str):
    """Train a byte-level BPE tokenizer from HuggingFace tokenizers library."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer.add_special_tokens(["<pad>", "<bos>", "<eos>", "<unk>"])

    special_tokens = {
        "pad": tokenizer.token_to_id("<pad>"),
        "bos": tokenizer.token_to_id("<bos>"),
        "eos": tokenizer.token_to_id("<eos>"),
        "unk": tokenizer.token_to_id("<unk>"),
    }
    return tokenizer, special_tokens


def tokenize_text(tokenizer, text: str) -> list[int]:
    """Tokenize text via the tokenizer library."""
    result = tokenizer.encode(text)
    return result.ids


def decode_tokens(tokenizer, ids: list[int]) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(ids)


def prepare_data(
    seq_len: int = 256,
    tokenizer=None,
    special_tokens: dict | None = None,
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    """
    Load Shakespeare, train tokenizer, tokenize, return train/val splits.

    Returns:
        tokenizer, train_tokens, val_tokens, meta
    """
    text = load_source_text()
    print(f"Loaded {len(text):,} characters")

    if tokenizer is None:
        print("Training BPE tokenizer...")
        tokenizer, special_tokens = train_bpe_tokenizer(text)
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Tokenize
    print("Tokenizing...")
    ids = np.array(tokenize_text(tokenizer, text), dtype=np.uint16)
    print(f"Total tokens: {len(ids):,}")

    # Split into train/val (90/10), trim to multiple of seq_len
    split = int(len(ids) * 0.9)
    train_len = (split // seq_len) * seq_len
    val_len = ((len(ids) - split) // seq_len) * seq_len

    if train_len == 0 or val_len == 0:
        raise ValueError(f"Not enough tokens for seq_len={seq_len}")

    train_tokens = ids[:train_len].reshape(-1, seq_len).copy()
    val_tokens = ids[split:split + val_len].reshape(-1, seq_len).copy()

    print(f"Train: {train_tokens.shape[0]:,} sequences ({train_tokens.shape[0] * seq_len:,} tokens)")
    print(f"Val:   {val_tokens.shape[0]:,} sequences ({val_tokens.shape[0] * seq_len:,} tokens)")

    meta = {"vocab_size": tokenizer.get_vocab_size(), "seq_len": seq_len, "special_tokens": special_tokens}
    return tokenizer, train_tokens, val_tokens, meta


if __name__ == "__main__":
    tok, train, val, meta = prepare_data()
    print(f"\nSample decode: {decode_tokens(tok, train[0, :50].tolist())!r}")
