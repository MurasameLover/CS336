import fasttext

_model = fasttext.load_model("cs336_data/lid.176.bin")

def language_identification(text: str) -> tuple[str, float]:
    # 清理文本（fasttext对空字符串会报错）
    text = text.strip()

    if text:
        """
        跑英文测试时报错了：
            fasttext不能处理 \n
        """
        text = text.replace("\n", "")

        labels, scores = _model.predict(text)

        label = labels[0]
        score = scores[0]

        """
        测试代码假定:
            英语 -> "en"
            中文 -> "zh"
        而 fastText 的输出标签可能并不完全一致
        例如可能输出 __label__zh 或 __label__zh-cn
        需要在适配器中进行适当的映射
        """

        language = label.replace("__label__", "")

        return (language, float(score))