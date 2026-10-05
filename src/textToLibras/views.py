from utils.nlp import get_nlp_model, STOP_POS
import requests

class TextToLibras:

    def __init__(self):
        self.time_words = ["ONTEM", "HOJE", "AMANHÃ", "AGORA", "PASSADO", "FUTURO"]
        self.time_words_expt = ["MANHÃ", "TARDE", "NOITE"]
        self.aux_expt = ["IR"]

    def _get_word_sign(self, term: str) -> str:
        term_json = requests.get(
            f"https://api-signbank.levantelab.com.br/api/tabela_sinais/sinais?page=1&search={term}&search_type=start_with"
        ).json()
        try:
            return term_json["data"][0]["json"]["video"]["url"]
        except:
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

            # if token.is_punct or token.pos_ in STOP_POS:
            #     continue
            if (token.pos_ == "ADV" and lema in self.time_words) or \
                (token.pos_ == "NOUN" and lema in self.time_words_expt):
                time_tokens.append(lema)
            elif token.pos_ == "AUX" and lema in self.aux_expt:
                normal_tokens.append(lema)
            elif lema == "NÃO":
                negation_tokens.append(lema)
            elif token.is_punct or token.pos_ in STOP_POS:
                continue
            else:
                normal_tokens.append(lema)
        return time_tokens + normal_tokens + negation_tokens

    def process_text(self, text: str) -> list:
        try:
            terms = self._translate_to_gloss(text)
            translated = []
            for term in terms:
                translated.append(self._get_word_sign(term))
            print(terms, translated)
            self._test_view_sings(translated)
            return translated
        except Exception as e:
            print(e)
            return []

    def _test_view_sings(self, translated: list) -> None:
        import cv2
        try:
            for url in translated:
                if "https://" not in url:
                    import numpy as np
                    import time
                    largura = 1280
                    altura = 720
                    frame = np.zeros((altura, largura, 3), dtype=np.uint8)
                    fonte = cv2.FONT_HERSHEY_SIMPLEX
                    escala = 1.5
                    espessura = 3
                    cor = (255, 255, 255)
                    texto = str(url)
                    tamanho_texto, _ = cv2.getTextSize(texto, fonte, escala, espessura)
                    largura_texto, altura_texto = tamanho_texto
                    pos_x = (largura - largura_texto) // 2
                    pos_y = (altura + altura_texto) // 2
                    cv2.putText(frame, texto, (pos_x, pos_y), fonte, escala, cor, espessura, cv2.LINE_AA)
                    tempo_inicial = time.time()
                    while time.time() - tempo_inicial < 2:
                        cv2.imshow("Missing word", frame)
                        key = cv2.waitKey(100) & 0xFF
                        if key in [27, ord('q')]:
                            break
                    continue

                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print(f"Erro ao abrir o vídeo: {url}")
                    continue
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow("Translation", frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                cap.release()
        finally:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)