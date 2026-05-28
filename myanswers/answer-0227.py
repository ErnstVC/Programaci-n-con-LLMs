
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

def entrenar_modelo_calibrado_texto_numerico(df, text_col, num_cols, target_col, test_size=0.2, random_state=42):
    data = df.dropna(subset=[target_col]).copy()
    X = data[[text_col] + num_cols]
    y = data[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pre = ColumnTransformer(transformers=[
        ("txt", TfidfVectorizer(max_features=300), text_col),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols)
    ])

    base  = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier":   float(brier_score_loss(y_test, proba)),
        "modelo_calibrado": model
    }
