import spacy

_nlp_model = None
#DET ADP AUX
STOP_POS = {"DET", "ADP", "AUX"}

def get_nlp_model():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("pt_core_news_sm")
    return _nlp_model