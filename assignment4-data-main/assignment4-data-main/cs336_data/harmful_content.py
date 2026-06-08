import fasttext

_nsfw_model = fasttext.load_model("cs336_data/dolma_fasttext_nsfw_jigsaw_model.bin")
_toxic_model = fasttext.load_model("cs336_data/dolma_fasttext_hatespeech_jigsaw_model.bin")


def classify_nsfw(text: str) -> tuple[str, float]:
    text = text.strip()
    if not text:
        return ("non-nsfw", 0.0)

    labels, scores = _nsfw_model.predict(text)
    label = labels[0].replace("__label__", "")  # 去掉前缀
    return (label, float(scores[0]))


def classify_toxic_speech(text: str) -> tuple[str, float]:
    text = text.strip()
    if not text:
        return ("non-toxic", 0.0)

    labels, scores = _toxic_model.predict(text)
    label = labels[0].replace("__label__", "")
    return (label, float(scores[0]))