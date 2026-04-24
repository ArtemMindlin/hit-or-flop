.PHONY: help sync train ui lint clean

help:
	@echo "🛠️  Hit-or-Flop Commands:"
	@echo "  make sync     - Instala/Sincroniza las dependencias con uv"
	@echo "  make train    - Entrena el modelo Random Forest y lo registra en MLflow"
	@echo "  make ui       - Levanta el dashboard de MLflow (http://localhost:5000)"
	@echo "  make lint     - Revisa y formatea el código fuente usando ruff"
	@echo "  make clean    - Elimina archivos temporales y caché de Python"

sync:
	uv sync

train:
	uv run scripts/train.py

ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

lint:
	uv run ruff check src/ scripts/
	uv run ruff format src/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .ruff_cache/
