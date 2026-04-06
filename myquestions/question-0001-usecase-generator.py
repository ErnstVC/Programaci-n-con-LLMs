
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso_anomalias():
    """
    Genera un caso de prueba para la función detectar_anomalias_energeticas.
    
    INPUT:
        - DataFrame con:
            consumo_kwh (con algunos nulos)
            temperatura_ambiente (con algunos nulos)
            tipo_tarifa (categórica)
    
    OUTPUT:
        - DataFrame con columna 'es_anomalia'
    """
    
    # 1. Tamaño del dataset
    n_rows = np.random.randint(15, 26)
    
    # 2. Generar datos
    data = {
        "consumo_kwh": np.random.normal(300, 50, n_rows),
        "temperatura_ambiente": np.random.normal(25, 5, n_rows),
        "tipo_tarifa": np.random.choice(
            ["residencial", "comercial", "industrial"], n_rows
        )
    }
    
    df = pd.DataFrame(data)
    
    # 3. Introducir nulos
    # consumo_kwh (para probar eliminación)
    nan_indices = np.random.choice(n_rows, size=2, replace=False)
    df.loc[nan_indices, "consumo_kwh"] = np.nan
    
    # temperatura (para imputación)
    nan_indices_temp = np.random.choice(n_rows, size=2, replace=False)
    df.loc[nan_indices_temp, "temperatura_ambiente"] = np.nan
    
    # 4. Introducir anomalías (valores extremos)
    n_outliers = max(1, n_rows // 10)
    outlier_indices = np.random.choice(n_rows, size=n_outliers, replace=False)
    df.loc[outlier_indices, "consumo_kwh"] *= 5  # consumo exagerado
    
    # ---------------- OUTPUT ESPERADO ----------------
    
    # 5. Limpieza
    df_clean = df.dropna(subset=["consumo_kwh"]).copy()
    
    # 6. Columnas
    num_col = ["temperatura_ambiente"]
    cat_col = ["tipo_tarifa"]
    
    # 7. Transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), num_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col)
        ]
    )
    
    X = preprocessor.fit_transform(df_clean)
    
    # 8. Modelo de detección de anomalías
    modelo = IsolationForest(random_state=42)
    preds = modelo.fit_predict(X)
    
    # 9. Agregar resultado
    df_clean["es_anomalia"] = preds
    
    # 10. Estructura final
    input_dict = {
        "df": df
    }
    
    output_df = df_clean
    
    return input_dict, output_df


# Ejemplo de uso
entrada, salida_esperada = generar_caso_detectar_anomalias()

print("---- INPUT ----")
print(entrada["df"])

print("\n---- OUTPUT ----")
print(salida_esperada.head())
