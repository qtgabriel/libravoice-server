import cv2
import csv
import os
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuração do Modelo MediaPipe Tasks
MODEL_PATH = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)

# Inicializa o detector de mãos
detector = vision.HandLandmarker.create_from_options(options)

# 2. Configuração do CSV
OUTPUT_CSV = "dataset_libras_landmarks.csv"

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        header.extend(['dist_thumb_index', 'dist_thumb_middle', 'dist_index_middle'])
        writer.writerow(header)


def calcular_distancia(p1, p2):
    """Calcula a distância euclidiana 3D entre dois pontos (x, y, z)"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def extrair_e_normalizar(hand_landmarks):
    """Extrai os 21 landmarks, normaliza em relação ao pulso (Landmark 0)
    e adiciona features geométricas avançadas"""
    pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]

    base_x, base_y, base_z = pts[0][0], pts[0][1], pts[0][2]

    norm_landmarks = []
    for p in pts:
        norm_landmarks.extend([
            p[0] - base_x,
            p[1] - base_y,
            p[2] - base_z
        ])

    tamanho_mao = calcular_distancia(pts[0], pts[9])
    if tamanho_mao == 0:
        tamanho_mao = 1.0

    dist_thumb_index = calcular_distancia(pts[4], pts[8]) / tamanho_mao
    dist_thumb_middle = calcular_distancia(pts[4], pts[12]) / tamanho_mao
    dist_index_middle = calcular_distancia(pts[8], pts[12]) / tamanho_mao

    features_extra = [dist_thumb_index, dist_thumb_middle, dist_index_middle]

    return norm_landmarks + features_extra


# 3. Captura de Vídeo (macOS Webcam)
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# --- CONFIGURAÇÃO DA REGIÃO DE INTERESSE (ÁREA DA MÃO) ---
box_size = 260

# Ajustes de deslocamento (Offset X e Y):
# offset_x: Positivo move para a direita, negativo para a esquerda.
# offset_y: Positivo move para baixo, negativo para cima.
offset_x = -50
offset_y = -50

box_x1 = ((FRAME_WIDTH - box_size) // 2) + offset_x
box_y1 = ((FRAME_HEIGHT - box_size) // 2) + offset_y

box_x2 = box_x1 + box_size
box_y2 = box_y1 + box_size

print("=== COLETOR DE LANDMARKS LIBRAS (MEDIAPIPE TASKS) ===")
letra = input("Digite a letra ou classe que vai gravar agora (ex: A, B, NEUTRO): ").strip().upper()
total_amostras = 300

print("\n--- INSTRUÇÕES ---")
print("1. Posicione sua mão confortavelmente dentro do quadrado vermelho.")
print("2. Pressione 'G' para INICIAR a gravação.")
print("3. Mova levemente a mão dentro da área enquanto o contador avança.")
print("4. Pressione 'Q' para interromper e sair.")

gravando = False
contador = 0
start_time = time.time()

with open(OUTPUT_CSV, mode='a', newline='') as f:
    writer = csv.writer(f)

    while cap.isOpened() and contador < total_amostras:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera do Mac.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Desenha o quadrado guia com os offsets aplicados
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)

        if detection_system_result := detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            h, w, _ = frame.shape

            pulso = hand_landmarks[0]
            px, py = int(pulso.x * w), int(pulso.y * h)
            dentro_da_caixa = (box_x1 <= px <= box_x2) and (box_y1 <= py <= box_y2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            if gravando and dentro_da_caixa:
                norm_landmarks = extrair_e_normalizar(hand_landmarks)
                writer.writerow([letra] + norm_landmarks)
                contador += 1

        status_texto = f"GRAVANDO: {contador}/{total_amostras}" if gravando else "Pressione 'G' para comecar"
        cor_status = (0, 0, 255) if gravando else (255, 255, 0)

        cv2.putText(frame, f"Classe: {letra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status_texto, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)
        cv2.putText(frame, "Apoie a mao no quadrado", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1)

        cv2.imshow("Coletor Libras - Tasks API (macOS)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('g') or key == ord('G'):
            gravando = True
        elif key == ord('q') or key == ord('Q'):
            break

cap.release()
cv2.destroyAllWindows()
detector.close()
print(f"\nConcluído! {contador} amostras salvas para '{letra}' no arquivo {OUTPUT_CSV}.")