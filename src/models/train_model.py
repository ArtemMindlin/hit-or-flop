from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def get_model_pipeline() -> Pipeline:
    """
    Define el pipeline de Machine Learning.
    Se utiliza un RandomForestRegressor optimizado (max_depth=20, n_estimators=200).
    """
    model = Pipeline([
        ("rf", RandomForestRegressor(max_depth=20, n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return model

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Entrena el pipeline y evalúa el rendimiento sobre el set de prueba.
    Devuelve el modelo entrenado y sus métricas principales (MAE, R2).
    """
    pipeline = get_model_pipeline()
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return pipeline, mae, r2

def save_model(pipeline, filepath: str):
    """
    Guarda el modelo serializado en la ruta especificada.
    """
    joblib.dump(pipeline, filepath)
