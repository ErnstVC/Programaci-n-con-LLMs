
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio (input/output) para la función preparar_datos.
    El input contiene un DataFrame con nulos y escalas variadas.
    El output contiene los arrays de numpy procesados (Imputación + Escalado).
    """
    # 1. Configuración aleatoria del dataset
    n_rows = np.random.randint(5, 11)  # Entre 5 y 10 filas
    n_cols = np.random.randint(2, 5)   # Entre 2 y 4 características + 1 target
    
    # Crear nombres de columnas
    cols = [f'feature_{i}' for i in range(n_cols)]
    target_col = 'target'
    
    # 2. Generar datos numéricos aleatorios con diferentes escalas
    # Usamos una media y desviación estándar aleatoria para cada columna
    data = {}
    for col in cols:
        mu = np.random.uniform(10, 100)
        sigma = np.random.uniform(1, 10)
        values = np.random.normal(mu, sigma, n_rows)
        
        # Introducir 1 o 2 valores NaN aleatoriamente por columna
        nan_indices = np.random.choice(n_rows, size=np.random.randint(1, 3), replace=False)
        values[nan_indices] = np.nan
        data[col] = values
    
    # Generar el target (sin nulos, según estándares de ML)
    data[target_col] = np.random.randint(0, 2, n_rows)
    
    df_input = pd.DataFrame(data)
    
    # --- CÁLCULO DEL OUTPUT ESPERADO ---
    # Separar X e y
    X = df_input[cols].values
    y = df_input[target_col].values
    
    # Imputar (Media)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Escalar (Standard)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 3. Estructurar el retorno
    input_dict = {
        "df": df_input,
        "target_col": target_col
    }
    
    output_tuple = (X_scaled, y)
    
    return input_dict, output_tuple

# Ejemplo de ejecución
entrada, salida_esperada = generar_caso_de_uso_preparar_datos()

print("--- INPUT (DataFrame con nulos) ---")
print(entrada['df'])
print("\n--- OUTPUT (X_scaled primera fila) ---")
print(salida_esperada[0][0])
