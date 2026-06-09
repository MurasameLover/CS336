"""
实现精确去重：仅保留唯一出现的行

step1：
    统计整个语料中每一行的出现次数

step2：
    重写文件，仅保留唯一出现的行

tips：
    为了节省内存，dict[str, int]保存所有行的出现频率时，
    不把一整句话作为key，而是保存一整行的hash值

"""
import mmh3     # 包含hash函数
from pathlib import Path

def exact_line_deduplication(
    input_path: list[str],
    output_dir: str
):
    line_counts: dict[str, int] = {}

    """ 
    input输入的是一系列文本路径:
        a/1.txt
        a/2.txt
    output:
        b/
    
    最终应生成：
        b/1.txt
        b/2.txt

    """
    # 第一遍遍历，统计所有语料中每一行的出现次数
    for filepath in input_path:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line_hash = str(mmh3.hash(line, seed=20070610))
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    # 第二遍遍历，重写整个语料，仅保留唯一出现的行
    for filepath in input_path:
        output_path = Path(output_dir) / Path(filepath).name
        with open(filepath, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line_hash = str(mmh3.hash(line, seed=20070610))
                if line_counts[line_hash] == 1:
                    fout.write(line)