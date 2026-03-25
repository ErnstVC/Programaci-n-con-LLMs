
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_moons

def generar_caso_de_uso_tuning():
    """
    Genera un caso de uso aleatorio (input y output esperado) 
    para la función optimizar_svm_grid.
    """
    # 1. Generar datos con una estructura no lineal (forma de medias lunas)
    # Esto hará que el kernel 'rbf' suela ganar sobre el 'linear'
    n_muestras = np.random.randint(60, 100)
    ruido = np.random.uniform(0.1, 0.2)
    X_np, y_np = make_moons(n_samples=n_muestras, noise=ruido, random_state=42)
    
    # Convertir a DataFrame de Pandas para el input
    df_X = pd.DataFrame(X_np, columns=['sensor_a', 'sensor_b'])
    
    # --- CÁLCULO DEL OUTPUT ESPERADO ---
    # Definimos la misma rejilla que debe usar la función del usuario
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    
    # Ejecutamos la búsqueda para conocer la respuesta correcta
    # Usamos cv=3 como se especificó en la misión
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_np, y_np)
    
    # 2. Estructurar el retorno
    input_dict = {
        "X": df_X,
        "y": y_np
    }
    
    output_dict = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_
    }
    
    return input_dict, output_dict
