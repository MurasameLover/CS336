import fasttext

_model = fasttext.load_model("cs336_data/quality_classifier.bin")


def classify_quality(text: str) -> tuple[str, float]:
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if not text:
        return ("cc", 0.0)

    labels, scores = _model.predict(text)
    label = labels[0].replace("__label__", "")
    return (label, float(scores[0]))