import pandas as pd
import numpy as np

def analizar_elasticidad_precio(df_precios):
    P1, P2 = df_precios['precio'].iloc[0], df_precios['precio'].iloc[1]
    Q1, Q2 = df_precios['cantidad'].iloc[0], df_precios['cantidad'].iloc[1]
    
    numerador = (Q2 - Q1) / ((Q2 + Q1) / 2)
    denominador = (P2 - P1) / ((P2 + P1) / 2)
    
    if denominador == 0:
        elasticidad = np.inf
    else:
        elasticidad = numerador / denominador
    
    categoria = "Elástico" if abs(elasticidad) > 1 else "Inelástico"
    
    return {
        'elasticidad': float(elasticidad),
        'categoria': categoria
    }
