import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga los datos crudos desde un archivo Parquet.
    """
    return pd.read_parquet(filepath)
