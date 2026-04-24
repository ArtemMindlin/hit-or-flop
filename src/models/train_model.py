from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow

def get_model_pipeline(max_depth=20, n_estimators=200) -> Pipeline:
    """
    Define el pipeline de Machine Learning.
    Se utiliza un RandomForestRegressor con parámetros configurables.
    """
    model = Pipeline([
        ("rf", RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1))
    ])
    return model

def train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=20, n_estimators=200):
    """
    Entrena el pipeline y evalúa el rendimiento sobre el set de prueba.
    Devuelve el modelo entrenado y sus métricas principales (MAE, R2).
    Registra todo automáticamente en MLflow.
    """
    # 1. Registrar hiperparámetros en MLflow
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("model_type", "RandomForestRegressor")

    pipeline = get_model_pipeline(max_depth, n_estimators)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 2. Registrar métricas de rendimiento en MLflow
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    return pipeline, mae, r2

def save_model(pipeline, filepath: str):
    """
    Guarda el modelo serializado de forma local tradicional.
    """
    joblib.dump(pipeline, filepath)
