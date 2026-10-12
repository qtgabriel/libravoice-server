import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL_PATH = os.path.join(BASE_DIR, "modelo_libras_letras.pkl")

def get_pkl_model():
    return joblib.load(MODEL_PKL_PATH)