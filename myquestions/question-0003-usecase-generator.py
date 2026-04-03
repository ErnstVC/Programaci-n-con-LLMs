
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def generar_caso_de_uso_clasificacion():
    """
    Genera un caso de uso (input y output esperado) 
    para la función evaluar_clasificador_cultivos.
    
    Incluye:
    - Clases desbalanceadas
    - Ruido y solapamiento
    - Algunos outliers
    - Evaluación completa
    """
    
    # 1. Tamaño del dataset
    n_muestras = np.random.randint(120, 200)
    
    # 2. Definir centros por clase
    centros = {
        'Trigo': [0.3, 6.5],
        'Maíz': [0.7, 5.8],
        'Cebada': [0.5, 6.2]
    }
    
    clases = ['Trigo', 'Maíz', 'Cebada']
    probabilidades = [0.6, 0.3, 0.1]  # 🔥 DESBALANCE REAL
    
    datos = []
    
    for _ in range(n_muestras):
        clase = np.random.choice(clases, p=probabilidades)
        centro = centros[clase]
        
        # Ruido normal (solapamiento)
        humedad = centro[0] + np.random.normal(0, 0.15)
        ph = centro[1] + np.random.normal(0, 0.3)
        
        # 🔥 Outliers ocasionales
        if np.random.rand() < 0.05:
            humedad += np.random.normal(0, 1)
            ph += np.random.normal(0, 1)
        
        datos.append([humedad, ph, clase])
    
    df = pd.DataFrame(datos, columns=['sensor_humedad', 'sensor_ph', 'cultivo'])
    
    # ---------------- OUTPUT ESPERADO ----------------
    
    X = df[['sensor_humedad', 'sensor_ph']]
    y = df['cultivo']
    
    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 4. Modelo
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # 5. Predicción
    y_pred = rf.predict(X_test)
    
    # 6. Matriz de confusión (orden fijo 🔥)
    matriz = confusion_matrix(
        y_test, y_pred, labels=['Trigo', 'Maíz', 'Cebada']
    )
    
    # 7. Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # 8. Errores totales
    total_errores = int(np.sum(matriz) - np.trace(matriz))
    
    # 9. Estructura final
    input_dict = {
        "df": df,
        "target_col": "cultivo"
    }
    
    output_dict = {
        "accuracy": acc,
        "confusion_matrix": matriz,
        "total_errores": total_errores
    }
    
    return input_dict, output_dict


# Ejemplo de ejecución
entrada, salida_esperada = generar_caso_de_uso_clasificacion()

print("---- INPUT ----")
print(entrada["df"].head())

print("\n---- OUTPUT ----")
print("Accuracy:", salida_esperada["accuracy"])
print("Matriz de Confusión:\n", salida_esperada["confusion_matrix"])
print("Errores Totales:", salida_esperada["total_errores"])
