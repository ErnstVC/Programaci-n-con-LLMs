
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

def estimar_densidad_kde(df, value_col, bandwidth=0.5, num_points=100):
    valores = df[value_col].dropna().to_numpy().reshape(-1, 1)
    
    grid = np.linspace(valores.min(), valores.max(), num_points).reshape(-1, 1)
    
    modelo = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    modelo.fit(valores)
    
    log_density = modelo.score_samples(grid)
    density = np.exp(log_density)
    
    resultado = pd.DataFrame({
        value_col: grid.ravel(),
        "density": density
    })
    
    return resultado.sort_values(by=value_col).reset_index(drop=True)
