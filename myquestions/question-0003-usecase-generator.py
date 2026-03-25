Para el reto de Clasificación Multiclase y Matriz de Confusión, aquí tienes la función generar_caso_de_uso_clasificacion.

Esta función es más sofisticada que una simple generación aleatoria: crea grupos de puntos (clusters) para que el modelo tenga algo que aprender, pero con suficiente solapamiento para que la matriz de confusión no sea perfecta y muestre errores interesantes.

Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def generar_caso_de_uso_clasificacion():
    """
    Genera un caso de uso aleatorio (input y output esperado) 
    para la función evaluar_clasificador_cultivos.
    """
    # 1. Definir parámetros aleatorios para el dataset
    n_muestras = np.random.randint(100, 200)
    
    # Crear 3 centros para las clases (Trigo, Maíz, Cebada)
    # Cada clase tendrá una "posición" diferente en el espacio de sensores
    centros = {
        'Trigo': [0.3, 6.5],
        'Maíz': [0.7, 5.8],
        'Cebada': [0.5, 6.2]
    }
    
    datos = []
    clases = ['Trigo', 'Maíz', 'Cebada']
    
    for _ in range(n_muestras):
        clase = np.random.choice(clases)
        centro = centros[clase]
        # Añadir ruido normal para que los puntos se dispersen y se solapen
        humedad = centro[0] + np.random.normal(0, 0.15)
        ph = centro[1] + np.random.normal(0, 0.3)
        datos.append([humedad, ph, clase])
    
    df = pd.DataFrame(datos, columns=['sensor_humedad', 'sensor_ph', 'cultivo'])
    
    # --- CÁLCULO DEL OUTPUT ESPERADO (siguiendo las reglas de la misión) ---
    X = df[['sensor_humedad', 'sensor_ph']]
    y = df['cultivo']
    
    # Split estratificado al 70/30 con random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar RandomForest con 50 estimadores
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predecir y evaluar
    y_pred = rf.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # Calcular errores totales (suma de la matriz - traza de la matriz)
    # La traza (np.trace) es la suma de la diagonal principal (aciertos)
    total_errores = int(np.sum(matriz) - np.trace(matriz))
    
    # 2. Estructurar el retorno
    input_dict = {
        "df": df,
        "target_col": "cultivo"
    }
    
    output_dict = {
        "accuracy": acc,
        "matriz_confusion": matriz,
        "total_errores": total_errores
    }
    
    return input_dict, output_dict
