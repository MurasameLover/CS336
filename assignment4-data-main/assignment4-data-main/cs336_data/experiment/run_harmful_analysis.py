"""
2.5节harmful_content第（d）题：
将的有害内容分类器运行在从 WARC 文件提取出的文本上（使用前面实现的 HTML 文本提取函数）。
然后：
    随机抽取 20 个样本；
    人工判断其是否有害；
    与分类器预测进行比较；
    记录分类错误。
"""


from __future__ import annotations

import random

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.extract_txt import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech

warc_path = "example.warc.gz"

results = []
MAX_RECORDS = 500

with open(warc_path, "rb") as f:
    for i, record in enumerate(tqdm(ArchiveIterator(f, WarcRecordType.response))):
        if i >= MAX_RECORDS:
            break

        if record.http_headers is None:
            continue

        url = record.headers.get("WARC-Target-URI", "")
        html_bytes = record.reader.read()

        # 预清理：修复无效 UTF-8 字节（有些 WARC 记录含坏字节）
        try:
            html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            html_bytes = html_bytes.decode("utf-8", errors="replace").encode("utf-8")

        text = extract_text_from_html_bytes(html_bytes)
        if not text or not text.strip():
            continue

        # fastText predict 不接受换行符，替换为空格
        text_one_line = text.replace("\n", " ").replace("\r", " ")

        nsfw_label, nsfw_score = classify_nsfw(text_one_line)
        toxic_label, toxic_score = classify_toxic_speech(text_one_line)

        results.append((
            url,
            text[:100],
            nsfw_label, nsfw_score,
            toxic_label, toxic_score,
        ))

# 统计
total = len(results)
nsfw_count = sum(1 for r in results if r[2] == "nsfw")
toxic_count = sum(1 for r in results if r[2] == "toxic")

print(f"\n===== 统计 =====")
print(f"处理记录数: {total}")
print(f"NSFW: {nsfw_count} ({nsfw_count/total*100:.1f}%)")
print(f"Toxic: {toxic_count} ({toxic_count/total*100:.1f}%)")

def safe_print(text: str):
    """安全打印，避免 GBK 编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk"))


# 抽样 20 条
sample = random.sample(results, min(20, len(results)))
print(f"\n===== 随机抽样 20 条 =====")
for i, (url, preview, nsfw_l, nsfw_s, toxic_l, toxic_s) in enumerate(sample):
    safe_print(f"\n--- 样例 {i+1} ---")
    safe_print(f"URL: {url}")
    safe_print(f"文本预览: {preview}")
    safe_print(f"NSFW: {nsfw_l} ({nsfw_s:.4f})")
    safe_print(f"Toxic: {toxic_l} ({toxic_s:.4f})")
