# 🤖 AI Agent Harnessing (AGENTS.md)

Welcome to the **Hit-or-Flop** project repository. 
If you are an AI assistant, coding agent, or LLM (like Gemini, Copilot, or Cursor) interacting with this codebase, you **MUST** strictly adhere to the following rules and context. These are foundational mandates.

## 1. Project Context
*   **Goal**: Predict the commercial success ("Hit" or "Flop") of songs using Spotify audio features and NLP on scraped lyrics.
*   **Tone/Level**: This is a production-ready Machine Learning / MLOps project. Treat all code and infrastructure choices as Senior Data Engineering/ML Engineer level.

## 2. Architectural Mandates
*   **Notebooks are for EDA only**: Do NOT write production model training code or complex data pipelines inside the `notebooks/` directory. 
*   **Modular Source Code**: All reusable functions, data loaders, feature engineering classes, and ML pipelines MUST reside in the `src/` directory (`src/data/`, `src/features/`, `src/models/`).
*   **Entrypoints**: Executable scripts must be placed in the `scripts/` directory (e.g., `scripts/train.py`, `scripts/apiazlyrics.py`).

## 3. Data & Storage Rules
*   **Format**: **NEVER** use `.csv` for large datasets. Always use and default to **Apache Parquet** (`pd.read_parquet`, `df.to_parquet`) with Snappy compression for performance and size efficiency.
*   **Location**: Data must strictly follow the DVC (Data Version Control) style layout:
    *   `data/raw/`: Immutable, original data.
    *   `data/processed/`: Cleaned, feature-engineered data ready for ML.
    *   `data/models/`: Serialized models (`.joblib`, `.pkl`).
*   **Paths**: NEVER use absolute or hardcoded OS paths (e.g., `C:/Users/...`). Always use relative paths based on the project root.

## 4. Technology & MLOps Stack
*   **Dependency Management**: This project uses **`uv`**. Do not suggest `pip install` or `conda install`. Use `uv add <package>` or `uv run <script>`. Update `pyproject.toml` accordingly.
*   **Execution Harness**: Always prefer running commands via the provided **`Makefile`** (e.g., `make train`, `make lint`).
*   **Experiment Tracking**: All model training MUST be instrumented with **MLflow**. Log parameters, metrics (MAE, R2), and artifacts (models) using the local SQLite backend (`sqlite:///mlruns.db`).

## 5. Coding Standards
*   **Type Hinting**: All new Python functions must include static type hints (e.g., `def process(df: pd.DataFrame) -> pd.DataFrame:`).
*   **Documentation**: Include concise NumPy or Google style docstrings for every class and function in `src/`.
*   **Linting & Formatting**: The project uses **`ruff`**. Before declaring a task complete, run `make lint` to ensure code quality and formatting compliance.
*   **Memory Management**: When handling Pandas DataFrames, drop high-cardinality text columns (like raw lyrics or genres) before applying One-Hot Encoding to prevent Out-Of-Memory (OOM) errors.

## 6. Git Workflow
*   **Commits**: Follow Conventional Commits format (e.g., `feat(ml): ...`, `fix(data): ...`, `docs: ...`, `chore: ...`).
*   **Secrets**: NEVER commit API keys, absolute paths, or the `.venv` directory.
