
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

def generar_caso_de_uso_seleccion_caracteristicas():
    """
    Genera un caso de uso aleatorio para la función seleccionar_mejores_caracteristicas.
    Crea un set de datos con variables predictoras reales, ruido y valores nulos.
    """
    n_rows = np.random.randint(10, 20)
    
    # 1. Crear variables con relación real al target
    # Generamos una relación lineal: target = 2*v1 + 0.5*v2 + ruido_bajo
    v1 = np.linspace(0, 100, n_rows) + np.random.normal(0, 5, n_rows)
    v2 = np.linspace(50, 0, n_rows) + np.random.normal(0, 2, n_rows)
    
    # 2. Crear variables de "ruido" (sin correlación)
    ruido_aleatorio = np.random.uniform(0, 1000, n_rows)
    constante = np.ones(n_rows) * np.random.randint(1, 100)
    
    # 3. Mezclar en un DataFrame
    df = pd.DataFrame({
        'feature_util_1': v1,
        'feature_util_2': v2,
        'ruido_v3': ruido_aleatorio,
        'constante_v4': constante,
        'redundante_v5': np.random.normal(10, 1, n_rows)
    })
    
    # Target basado principalmente en v1 y v2
    target = (2 * v1) - (1.5 * v2) + np.random.normal(0, 10, n_rows)
    df['target_real'] = target

    # 4. Introducir nulos aleatorios en las columnas de características
    for col in df.columns:
        if col != 'target_real':
            # Insertar 1 o 2 nulos por columna
            idx_nulos = np.random.choice(df.index, size=np.random.randint(1, 3), replace=False)
            df.loc[idx_nulos, col] = np.nan

    # --- CÁLCULO DEL OUTPUT ESPERADO ---
    k = 2
    X = df.drop(columns=['target_real'])
    y = df['target_real']
    
    # Simular la lógica de la función: Imputar con 0 y seleccionar
    X_filled = X.fillna(0)
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X_filled, y)
    
    # Obtener nombres de columnas seleccionadas
    cols_seleccionadas = X.columns[selector.get_support()].tolist()
    
    # 5. Formatear Input y Output
    input_dict = {
        "df": df,
        "target_col": "target_real",
        "k": k
    }
    
    output_tuple = (X_new, cols_seleccionadas)
    
    return input_dict, output_tuple
