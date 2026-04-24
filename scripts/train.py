import argparse
import sys
import os

# Añadir la carpeta raíz al PATH de python para poder importar 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data, get_features_target
from src.models.train_model import train_and_evaluate, save_model

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo Hit-or-Flop")
    parser.add_argument("--data_path", type=str, default="data/raw/data.parquet", help="Ruta al dataset Parquet de origen")
    parser.add_argument("--model_path", type=str, default="data/models/rf_popularity_model.joblib", help="Ruta de guardado del modelo")
    args = parser.parse_args()

    print(f"[*] 1. Cargando datos desde {args.data_path}...")
    df = load_data(args.data_path)

    print("[*] 2. Limpiando y preprocesando datos...")
    df_clean = preprocess_data(df)
    
    print("[*] 3. Generando Features (X) y Target (y)...")
    # 'new_popularity_2025' era el target principal en tu notebook
    X, y = get_features_target(df_clean, target_col="new_popularity_2025")
    
    print("[*] 4. Dividiendo datos (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[*] 5. Entrenando Random Forest... (Train shape: {X_train.shape})")
    print("       (Esto puede tardar unos segundos/minutos)")
    model, mae, r2 = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n[+] 🚀 ¡Modelo entrenado con éxito!")
    print(f"    - Error Absoluto Medio (MAE): {mae:.3f}")
    print(f"    - R2 Score:                   {r2:.3f}\n")

    print(f"[*] 6. Guardando modelo en {args.model_path}...")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    save_model(model, args.model_path)
    print("[+] Todo completado.")

if __name__ == "__main__":
    main()
