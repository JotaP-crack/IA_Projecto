# recommendation_system.py
# Sistema de recomendación con scikit-learn usando KNeighborsClassifier

# Importar las librerías necesarias
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

CSV_PATH = "products.csv"


def generar_dataset_ejemplo(ruta, n=300):
    """Genera un dataset de productos de ejemplo si no existe el CSV."""
    rng = np.random.default_rng(42)
    feature1 = rng.uniform(0, 10, n)   # ej: precio normalizado
    feature2 = rng.uniform(0, 10, n)   # ej: popularidad
    feature3 = rng.uniform(0, 10, n)   # ej: calificación
    # La etiqueta (categoría recomendada) depende de las características
    score = feature1 + feature2 + feature3
    label = np.where(score < 12, 0, np.where(score < 18, 1, 2))
    df = pd.DataFrame({
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "label": label,
    })
    df.to_csv(ruta, index=False)
    print(f"Dataset de ejemplo generado en '{ruta}'")


# Cargar los datos (dataset de productos)
if not os.path.exists(CSV_PATH):
    generar_dataset_ejemplo(CSV_PATH)
data = pd.read_csv(CSV_PATH)

# Preprocesamiento de datos
# Seleccionar las características y la etiqueta
features = data[["feature1", "feature2", "feature3"]]
labels = data["label"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Función de recomendación
def recommend(product_features):
    """Recibe [feature1, feature2, feature3] y devuelve la categoría recomendada."""
    entrada = pd.DataFrame([product_features], columns=features.columns)
    prediction = model.predict(entrada)
    return prediction[0]


# Ejemplo de uso de la función de recomendación
example_product = [1.0, 2.0, 3.0]
recommended_product = recommend(example_product)
print(f"Recommended Product: {recommended_product}")
