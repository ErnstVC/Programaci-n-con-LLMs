import pandas as pd
import numpy as np


def matriz_transicion(df, user_col, time_col, state_col):
    # Copia para no modificar el DataFrame original
    data = df.copy()

    # 1) Convertir a datetime si hace falta
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")

    # 2) Ordenar por usuario y tiempo
    data = data.sort_values(
        by=[user_col, time_col],
        ascending=[True, True]
    ).reset_index(drop=True)

    # 3) Obtener siguiente estado por usuario
    data["next_state"] = (
        data.groupby(user_col)[state_col]
        .shift(-1)
    )

    # 4) Eliminar últimos eventos (sin next_state)
    transitions = data.dropna(subset=["next_state"])

    # 5) Contar transiciones
    counts = (
        transitions
        .groupby([state_col, "next_state"])
        .size()
        .rename("count")
        .reset_index()
    )

    # 6) Construir matriz de conteos
    count_matrix = (
        counts
        .pivot(
            index=state_col,
            columns="next_state",
            values="count"
        )
        .fillna(0.0)
        .astype(float)
    )

    # 7) Asegurar matriz cuadrada con todos los estados
    all_states = sorted(
        set(data[state_col].dropna().unique()).union(
            set(transitions["next_state"].dropna().unique())
        )
    )

    count_matrix = count_matrix.reindex(
        index=all_states,
        columns=all_states,
        fill_value=0.0
    )

    # 8) Convertir conteos a probabilidades
    row_sums = count_matrix.sum(axis=1)

    prob_matrix = (
        count_matrix.div(
            row_sums.replace(0.0, np.nan),
            axis=0
        )
        .fillna(0.0)
    )

    # 9) Orden alfabético
    prob_matrix = (
        prob_matrix
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    return prob_matrix
