"""
2.6节 Quality Rules
过滤规则：（合格的）
    单词数 50~100000
    平均词长 3~10 字符
    ≤30% 的行以 ... 结尾
    ≥80% 的单词含至少一个字母
"""

""" 输入文本，输出bool，判断是否合格 """
def gopher_quality_filter(text: str) -> bool:
    # 根据pdf的提示，使用NLTK进行分词
    import nltk
    nltk.download("punkt")
    words = nltk.word_tokenize(text)

    # 规则1：单词数 50~100000
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False

    # 规则2：平均词长 3~10 字符
    mean_len = sum(len(word) for word in words) / word_count
    if mean_len < 3 or mean_len > 10:
        return False

    # 规则3：≤30% 的行以 ... 结尾
    lines = text.split("\n")
    """
    下一行代码的语法：
        rstrip() — Python 字符串方法，移除字符串右端的空白字符（空格  、制表符 \t、换行 \n
    等）。不加参数时默认移除所有空白。这里用 rstrip() 是因为像 "...\n" 或 "... " 这种以空白结尾的行，直接
    endswith("...") 会返回 False，但其实行内容是 ...
    """
    ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
    if ellipsis_lines / len(lines) > 0.3:
        return False

    # 规则4：≥80% 的单词含至少一个字母
    """
    下一行代码的语法：
        c.isalpha() — 判断字符 c 是否是字母
        any(c.isalpha() for c in w) — 对单词 w 的每个字符 c 检查是否字母，any() 在遇到第一个 True 时立刻返回True
    """
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / word_count < 0.8:
        return False

    return True