import pandas as pd
from typing import Tuple

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza inicial de columnas innecesarias.
    Basado en el notebook: data.iloc[:, :-23] y drop de columnas.
    """
    # Recortar columnas (asumiendo que las últimas 23 son targets secundarios o ruido)
    data_new = df.iloc[:, :-23]
    
    cols_to_drop = [
        "track_name", "id", "artist_name", "release_date", "year", 
        "genres", "subgenre_1", "subgenre_2", "genres_artists", "artists", "name"
    ]
    
    # Eliminar solo si existen para evitar errores
    cols_to_drop = [c for c in cols_to_drop if c in data_new.columns]
    if cols_to_drop:
        data_new = data_new.drop(columns=cols_to_drop)
        
    return data_new

def get_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa el dataset en la matriz de features (X) y el vector objetivo (y).
    Además codifica variables categóricas (OneHot) para que el modelo pueda procesarlas.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Codificar variables categóricas como géneros o subgéneros
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y
