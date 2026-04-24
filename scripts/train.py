import argparse
import sys
import os
import mlflow

# Añadir la carpeta raíz al PATH de python para poder importar 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data, get_features_target
from src.models.train_model import train_and_evaluate, save_model

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo Hit-or-Flop con MLflow")
    parser.add_argument("--data_path", type=str, default="data/processed/minable_view_w_imputed_genres.parquet", help="Ruta al dataset Parquet de origen")
    parser.add_argument("--model_path", type=str, default="data/models/rf_popularity_model.joblib", help="Ruta de guardado del modelo local")
    # Añadimos soporte para probar hiperparámetros desde la terminal
    parser.add_argument("--max_depth", type=int, default=20, help="Profundidad máxima del árbol")
    parser.add_argument("--n_estimators", type=int, default=200, help="Número de árboles")
    args = parser.parse_args()

    # 1. Configurar MLflow para guardar todo en una base de datos local (sqlite) en lugar de una carpeta suelta
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Hit-or-Flop Popularity Prediction")

    print(f"[*] Cargando datos desde {args.data_path}...")
    df = load_data(args.data_path)

    print("[*] Limpiando y preprocesando datos...")
    df_clean = preprocess_data(df)
    
    print("[*] Generando Features (X) y Target (y)...")
    X, y = get_features_target(df_clean, target_col="new_popularity_2025")
    
    print("[*] Dividiendo datos (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[*] Entrenando Random Forest con MLflow... (Train shape: {X_train.shape})")
    
    # 2. Envolver el entrenamiento en un 'run' de MLflow
    with mlflow.start_run(run_name="RandomForest Run"):
        # Registrar dimensiones de los datos
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        
        # Entrenar y evaluar
        model, mae, r2 = train_and_evaluate(
            X_train, X_test, y_train, y_test, 
            max_depth=args.max_depth, 
            n_estimators=args.n_estimators
        )

        print("\n[+] 🚀 ¡Modelo entrenado y registrado con éxito en MLflow!")
        print(f"    - Error Absoluto Medio (MAE): {mae:.3f}")
        print(f"    - R2 Score:                   {r2:.3f}\n")

        print(f"[*] Guardando modelo en MLflow Artifacts y en {args.model_path}...")
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        save_model(model, args.model_path)
        
        # Guardar el modelo físico dentro de MLflow (para versionado y descarga fácil)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
    print("\n[+] Todo completado.")
    print("👉 Para ver el dashboard interactivo de tus modelos, ejecuta en la terminal:")
    print("   uv run mlflow ui --backend-store-uri sqlite:///mlruns.db")

if __name__ == "__main__":
    main()
