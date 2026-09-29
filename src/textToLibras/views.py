from utils.nlp import get_nlp_model, STOP_POS

class TextToLibras:

    def __init__(self):
        self.time_words = ["ONTEM", "HOJE", "AMANHÃ", "AGORA", "PASSADO", "FUTURO"]
        self.time_words_expt = ["MANHÃ", "TARDE", "NOITE"]

    def _get_word_sign(self, term: str) -> str:
        return term

    def _translate_to_gloss(self, text: str) -> list:
        nlp_model = get_nlp_model()
        doc = nlp_model(text.lower())

        normal_tokens = []
        time_tokens = []
        negation_tokens = []

        for token in doc:
            lema = token.lemma_.upper()
            print(token, lema, token.pos_, token.tag_)

            if token.is_punct or token.pos_ in STOP_POS:
                continue

            if (token.pos_ == "ADV" and lema in self.time_words) or \
                (token.pos_ == "NOUN" and lema in self.time_words_expt):
                time_tokens.append(lema)
            elif lema == "NÃO":
                negation_tokens.append(lema)
            else:
                normal_tokens.append(lema)

        return time_tokens + normal_tokens + negation_tokens

    def process_text(self, text: str) -> list:
        try:
            terms = self._translate_to_gloss(text)
            translated = []
            for term in terms:
                translated.append(self._get_word_sign(term))
            return translated
        except Exception as e:
            print(e)
            return []