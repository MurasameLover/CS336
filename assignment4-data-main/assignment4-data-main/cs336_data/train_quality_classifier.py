import random
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.extract_txt import extract_text_from_html_bytes
from cs336_data.language_identification import language_identification
from cs336_data.gopher_quality import gopher_quality_filter

warc_path = "example.warc.gz"
model_path = "cs336_data/quality_classifier.bin"

# Step 1: 从 WARC 提取文本并分类
wiki_texts = []
cc_texts = []

# 把已有的测试用例也加进去
with open("tests/fixtures/high_quality_wiki_reference.txt", encoding="utf-8") as f:
    wiki_texts.append(f.read())
with open("tests/fixtures/low_quality_cc.txt", encoding="utf-8") as f:
    cc_texts.append(f.read())
with open("tests/fixtures/moby_extracted.txt", encoding="utf-8") as f:
    wiki_texts.append(f.read())

with open(warc_path, "rb") as f:
    for record in tqdm(ArchiveIterator(f, WarcRecordType.response)):
        if record.http_headers is None:
            continue

        html_bytes = record.reader.read()
        # 修复坏字节
        try:
            html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            html_bytes = html_bytes.decode("utf-8", errors="replace").encode("utf-8")

        text = extract_text_from_html_bytes(html_bytes)
        if not text or len(text.strip()) < 50:
            continue

        # 只保留英文
        lang, score = language_identification(text)
        if lang != "en" or score < 0.5:
            continue

        # 用 Gopher 规则判断质量
        if gopher_quality_filter(text):
            # 通过 Gopher = 相对高质量 → 打 "wiki" 标签
            wiki_texts.append(text)
        else:
            # 不通过 Gopher = 低质量 → 打 "cc" 标签
            cc_texts.append(text)

print(f"Wiki texts: {len(wiki_texts)}, CC texts: {len(cc_texts)}")

# Step 2: 写入 fastText 训练文件
train_path = Path("cs336_data/quality_train.txt")
with open(train_path, "w", encoding="utf-8") as f:
    for t in wiki_texts:
        # fastText 格式: __label__标签 文本内容（需去掉换行符）
        line = t.replace("\n", " ").replace("\r", " ")
        f.write(f"__label__wiki {line}\n")
    for t in cc_texts:
        line = t.replace("\n", " ").replace("\r", " ")
        f.write(f"__label__cc {line}\n")

print(f"Training data written: {train_path}")

# Step 3: 训练 fastText 分类器
model = fasttext.train_supervised(
    input=str(train_path),
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    dim=100,
)
model.save_model(model_path)
print(f"Model saved: {model_path}")

# Step 4: 快速验证测试用例
for label, test_path in [
    ("wiki", "tests/fixtures/high_quality_wiki_reference.txt"),
    ("cc", "tests/fixtures/low_quality_cc.txt"),
]:
    with open(test_path, encoding="utf-8") as f:
        text = f.read().replace("\n", " ").replace("\r", " ")
    labels, scores = model.predict(text)
    pred = labels[0].replace("__label__", "")
    print(f"Expected {label}: got {pred} ({scores[0]:.4f})")