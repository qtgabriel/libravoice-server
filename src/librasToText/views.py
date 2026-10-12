from utils.librasModel import get_pkl_model
import numpy as np
import math

class LibrasToText:

    def __init__(self, confidence_threshold=0.65):
        self.confidence_threshold = confidence_threshold

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _normalize_landmarks(self, landmarks: list) -> list:
        pts = []
        for lm in landmarks:
            if isinstance(lm, dict):
                pts.append([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
            else:
                pts.append([lm.x, lm.y, lm.z])

        base_x, base_y, base_z = pts[0][0], pts[0][1], pts[0][2]
        norm_landmarks = []
        for p in pts:
            norm_landmarks.extend([
                p[0] - base_x,
                p[1] - base_y,
                p[2] - base_z
            ])

        hand_size = self._calculate_distance(pts[0], pts[9])
        if hand_size == 0:
            hand_size = 1.0

        dist_thumb_index = self._calculate_distance(pts[4], pts[8]) / hand_size
        dist_thumb_middle = self._calculate_distance(pts[4], pts[12]) / hand_size
        dist_index_middle = self._calculate_distance(pts[8], pts[12]) / hand_size

        features_extra = [dist_thumb_index, dist_thumb_middle, dist_index_middle]

        return norm_landmarks + features_extra

    def _predict_single(self, landmarks: list) -> dict:
        features = self._normalize_landmarks(landmarks)
        model = get_pkl_model()
        probs = model.predict_proba([features])[0]
        max_prob_idx = np.argmax(probs)
        gesture = model.classes_[max_prob_idx]
        confidence = float(probs[max_prob_idx])
        if confidence < self.confidence_threshold:
            gesture = ""
        return {
            "gesture": str(gesture),
            "confidence": confidence
        }

    def process_points(self, payload) -> dict:
        if isinstance(payload, dict):
            if "frames" in payload:
                data_list = payload["frames"]
            elif "landmarks" in payload:
                content = payload["landmarks"]
                if content and isinstance(content[0], list) and isinstance(content[0][0], (dict, list)):
                    data_list = content
                else:
                    return self._predict_single(content)
            else:
                raise ValueError("Payload inválido. Chaves esperadas: 'frames' ou 'landmarks'.")
        elif isinstance(payload, list):
            if payload and isinstance(payload[0], list):
                data_list = payload
            else:
                return self._predict_single(payload)
        else:
            raise ValueError("Formato de dados não suportado.")

        predictions = []
        for frame_landmarks in data_list:
            pred = self._predict_single(frame_landmarks)
            predictions.append(pred)

        filtered_gestures = []
        last_gesture = None
        for item in predictions:
            current_gesture = item["gesture"]
            if current_gesture and current_gesture != last_gesture:
                filtered_gestures.append(current_gesture)
                last_gesture = current_gesture

        final_translation = "".join(filtered_gestures)

        return {"predictions_detail": predictions, "translated_phrase": final_translation}