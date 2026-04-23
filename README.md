# 🎵 Hit-or-Flop: Predicting Musical Success with ML

<p align="center">
  <img src="docs/presentations/Iconos/logo_spotify.png" width="100" alt="Spotify Logo">
</p>

This project, developed during the 3rd year of the Data Science Degree (UPV), earned a **9.8/10** grade. Its goal is to predict whether a song will be a "Hit" or a "Flop" by combining Spotify's audio metrics, lyric analysis via web scraping, and advanced Machine Learning models.

---

## 🚀 Key Differentiators
Beyond basic popularity analysis, this project implements:
*   **Professional Data Pipeline**: Migration from CSV to **Apache Parquet** to optimize storage and read speeds.
*   **NLP & Scraping**: A dynamic lyric extraction engine from AZLyrics with built-in text normalization logic.
*   **Statistical Modeling**: Implementation of **Zero-Inflated** models to handle the high variance in popularity data.
*   **Modern Tooling**: Dependency and environment management via **`uv`**, the high-performance Python standard.

---

## 🛠️ Tech Stack
*   **Languages**: Python 3.14+, R.
*   **Project Management**: `uv` (Fast Python package manager).
*   **Data Processing**: Pandas, NumPy, PyArrow (Parquet).
*   **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, CatBoost.
*   **Visualization & EDA**: Seaborn, Matplotlib, Association Rules (R).
*   **NLP & Scraping**: BeautifulSoup4, Requests.

---

## 📁 Project Structure
```text
hit-or-flop/
├── data/              # Data management (optimized in Parquet)
│   ├── raw/           # Original Spotify, Last.fm, and raw lyrics.
│   ├── processed/     # Cleaned datasets and final feature engineering.
│   └── models/        # Trained models (.pkl, .joblib).
├── notebooks/         # Research & experimentation
│   ├── popularity/    # Hit prediction and Zero-Inflated models.
│   ├── genres/        # Automatic musical genre classification.
│   └── lyrics/        # Text processing and sentiment analysis.
├── scripts/           # Extraction tools and production utilities.
├── docs/              # Technical documentation, presentations, and "State of the Art".
├── pyproject.toml     # Dependency definition with uv.
└── README.md
```

---

## 💻 Installation & Usage
This project uses `uv` to ensure maximum speed and reproducibility.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ArtemMindlin/hit-or-flop.git
   cd hit-or-flop
   ```

2. **Sync the environment**:
   ```bash
   uv sync
   ```

3. **Run the Lyric Scraper**:
   ```bash
   uv run scripts/apiazlyrics.py
   ```

---

## 📈 Methodology & Results
The project approaches the problem from three angles:
1.  **Feature Engineering**: Using intrinsic metrics (danceability, energy, tempo) and extrinsic metrics (genres, historical popularity).
2.  **Lyric Analysis**: Automatic extraction to identify linguistic patterns correlated with commercial success.
3.  **Model Evaluation**: Rigorous comparison between Gradient Boosting and Random Forest, optimizing for MAE and R² metrics.

---

## ✍️ Authors
*   **Artem Mindlin** - *Lead Data Engineer & ML Developer*
*   University Project developed by a 3rd-year Data Science team.

---
<p align="center">
  <i>Final Grade: 9.8/10</i>
</p>
