import cv2
import requests
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configurar o MediaPipe Tasks Hand Landmarker
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

# 2. Inicializar a Webcam
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# --- CONFIGURAÇÃO DA REGIÃO DE INTERESSE (ÁREA DA MÃO) ---
box_size = 260
offset_x = -50
offset_y = -50

box_x1 = ((FRAME_WIDTH - box_size) // 2) + offset_x
box_y1 = ((FRAME_HEIGHT - box_size) // 2) + offset_y
box_x2 = box_x1 + box_size
box_y2 = box_y1 + box_size

print("Iniciando soletração com reset automático... Pressione 'Q' para sair.")
start_time = time.time()

buffer_frames = []
max_buffer_size = 8
api_url = "http://127.0.0.1:8000/librasToText/translate"

palavra_formada = ""
ultimo_gesto_aceito = None
ultimo_tempo_gesto = 0
cooldown_segundos = 1.2

# Timer de inatividade realista para troca de palavra (ex: 3.5 segundos sem detectar mão/gesto válido)
ultimo_movimento_tempo = time.time()
tempo_limite_inatividade = 3.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    box_color = (0, 0, 255)  # Vermelho padrão
    current_time = time.time()

    mao_na_caixa = False

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        h, w, _ = frame.shape

        pulso = hand_landmarks[0]
        px, py = int(pulso.x * w), int(pulso.y * h)
        dentro_da_caixa = (box_x1 <= px <= box_x2) and (box_y1 <= py <= box_y2)

        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        if dentro_da_caixa:
            mao_na_caixa = True
            box_color = (0, 255, 0)  # Verde

            # Renova o tempo de inatividade pois a mão está ativa no quadrado
            ultimo_movimento_tempo = current_time

            frame_pts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
            buffer_frames.append(frame_pts)

            if len(buffer_frames) >= max_buffer_size:
                payload = {"frames": buffer_frames}
                try:
                    response = requests.post(api_url, json=payload)
                    print("Status Code:", response.status_code)
                    print("Resposta bruta da API:", response.text)
                    if response.status_code == 200:
                        data = response.json()
                        result_data = data.get("data", {})
                        if isinstance(result_data, dict):
                            traducao_lote = result_data.get("translated_phrase", "")

                            if traducao_lote:
                                nova_letra = traducao_lote[0]

                                if nova_letra != ultimo_gesto_aceito or (
                                        current_time - ultimo_tempo_gesto > cooldown_segundos):
                                    palavra_formada += nova_letra
                                    ultimo_gesto_aceito = nova_letra
                                    ultimo_tempo_gesto = current_time
                                    print(f"Letra adicionada: {nova_letra} | Palavra atual: {palavra_formada}")
                except Exception as e:
                    print("Erro na requisição:", e)

                buffer_frames = buffer_frames[-2:]

    # Verifica se o tempo sem interações ultrapassou o limite realista para resetar a palavra
    if palavra_formada and (current_time - ultimo_movimento_tempo > tempo_limite_inatividade):
        print(f"Inatividade detectada ({tempo_limite_inatividade}s). Palavra anterior apagada: {palavra_formada}")
        palavra_formada = ""
        ultimo_gesto_aceito = None
        buffer_frames = []

    # Desenha o quadrado guia
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)

    # --- PAINEL DE RESULTADO ---
    box_res_x1 = FRAME_WIDTH - 320
    box_res_y1 = 10
    box_res_x2 = FRAME_WIDTH - 10
    box_res_y2 = 70

    cv2.rectangle(frame, (box_res_x1, box_res_y1), (box_res_x2, box_res_y2), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Palavra: {palavra_formada}",
        (box_res_x1 + 10, box_res_y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(frame, "Pressione 'R' para limpar palavra", (box_x1 - 20, max(20, box_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

    cv2.imshow("Soletração Acumulativa LIBRAS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') or key == ord('R'):
        palavra_formada = ""
        ultimo_gesto_aceito = None
        print("Palavra resetada manualmente!")

cap.release()
cv2.destroyAllWindows()
detector.close()
detector.close()