import cv2
import joblib
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Carregar o Modelo Treinado (.pkl)
MODEL_PKL_PATH = "modelo_libras_letras.pkl"
try:
    model = joblib.load(MODEL_PKL_PATH)
    print("Modelo de classificação de LIBRAS carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo '{MODEL_PKL_PATH}' não foi encontrado. Execute o script de treino primeiro.")
    exit()

# 2. Configurar o MediaPipe Tasks Hand Landmarker
MODEL_TASK_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)


def calcular_distancia(p1, p2):
    """Calcula a distância euclidiana 3D entre dois pontos (x, y, z)"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def extrair_e_normalizar(hand_landmarks):
    """Extrai os 21 landmarks, normaliza em relação ao pulso (Landmark 0)
    e adiciona as features geométricas avançadas (Mudança A)"""
    pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]

    # Landmark 0 é o pulso
    base_x, base_y, base_z = pts[0][0], pts[0][1], pts[0][2]

    # 1. Landmarks normalizados em relação ao pulso
    norm_landmarks = []
    for p in pts:
        norm_landmarks.extend([
            p[0] - base_x,
            p[1] - base_y,
            p[2] - base_z
        ])

    # 2. Mudança A: Engenharia de Recursos Geométricos
    # Tamanho da mão para invariância de escala (distância do pulso 0 até a base do dedo médio 9)
    tamanho_mao = calcular_distancia(pts[0], pts[9])
    if tamanho_mao == 0:
        tamanho_mao = 1.0  # Evita divisão por zero

    # Distâncias relevantes
    dist_thumb_index = calcular_distancia(pts[4], pts[8]) / tamanho_mao   # Polegar ao Indicador
    dist_thumb_middle = calcular_distancia(pts[4], pts[12]) / tamanho_mao  # Polegar ao Médio
    dist_index_middle = calcular_distancia(pts[8], pts[12]) / tamanho_mao  # Indicador ao Médio

    features_extra = [dist_thumb_index, dist_thumb_middle, dist_index_middle]

    return norm_landmarks + features_extra


# 3. Inicializar a Webcam (macOS)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Iniciando detecção em tempo real... Pressione 'Q' para sair.")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    # Espelhar frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # Processar frame com MediaPipe Tasks
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    letra_detectada = "Nenhuma mao"
    confianca_texto = ""

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        # Desenhar articulações da mão na tela
        h, w, _ = frame.shape
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Extrair landmarks normalizados e features geométricas (67 colunas)
        features = extrair_e_normalizar(hand_landmarks)

        # Obter probabilidades de cada classe
        probs = model.predict_proba([features])[0]
        max_prob_idx = np.argmax(probs)
        confianca = probs[max_prob_idx] * 100

        # Filtro de confiança para evitar falsos positivos
        if confianca > 60.0:
            letra_detectada = model.classes_[max_prob_idx]
            confianca_texto = f"({confianca:.1f}%)"
        else:
            letra_detectada = "Incerto"

    # Exibir a letra na tela em destaque
    cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)  # Fundo escuro para legibilidade
    cv2.putText(
        frame,
        f"Letra: {letra_detectada} {confianca_texto}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow("Detector de LIBRAS - Tempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()