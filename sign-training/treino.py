import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar o Dataset salvo pelo coletor
CSV_FILE = "dataset_libras_landmarks.csv"

try:
    data = pd.read_csv(CSV_FILE)
    print(f"Dataset carregado com sucesso! Total de amostras: {len(data)}")
except FileNotFoundError:
    print(f"Erro: O arquivo '{CSV_FILE}' não foi encontrado. Execute o coletor primeiro.")
    exit()

# 2. Separar as Features (x, y, z dos 21 pontos) e o Target (label/letra)
X = data.drop(columns=['label'])
y = data['label']

print(f"Classes encontradas: {y.unique()}")

# 3. Divisão dos Dados em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Treinar o Classificador (Random Forest)
print("\nTreinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Avaliação do Modelo
y_pred = model.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("\n=== RESULTADOS DO TREINAMENTO ===")
print(f"Acurácia Geral: {acuracia * 100:.2f}%\n")
print("Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))

# 6. Salvar o Modelo Treinado
MODEL_OUTPUT = "modelo_libras_letras.pkl"
joblib.dump(model, MODEL_OUTPUT)
print(f"Modelo salvo com sucesso em: '{MODEL_OUTPUT}'")