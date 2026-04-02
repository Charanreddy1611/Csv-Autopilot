# CSV Autopilot

> Drop a CSV, get an instant deep-dive EDA report.

Part of the **30 Days / 30 Vibe-Coded Projects** challenge.

## What It Does

Upload any CSV file and CSV Autopilot will automatically:

- **Infer semantic types** — goes beyond pandas dtypes to detect booleans, identifiers, datetime strings, categorical integers, and free text.
- **Profile every column** — mean, std, skewness, kurtosis, coefficient of variation, entropy, mode, top values.
- **Visualize distributions** — histograms with marginal box plots, Q-Q plots against normal, and categorical bar charts.
- **Compute correlations** — Pearson & Spearman heatmaps, auto-detect highly-correlated pairs (|r| >= 0.8), scatter plots with trendlines.
- **Detect outliers** — three methods side-by-side: IQR, Z-score, and Isolation Forest (multivariate). Inspect flagged rows interactively.
- **Analyze missing values** — per-column counts, missing-value co-occurrence matrix, row completeness distribution, nullity heatmap.

## Quick Start

```bash
cd csv-autopilot
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), upload a CSV, and explore.

## Project Structure

```
csv-autopilot/
├── app.py              # Streamlit UI — tabs, layout, interactivity
├── analyzer.py         # Core analysis engine — type inference, profiling, correlations, outliers, missing patterns
├── visualizations.py   # Plotly chart builders
├── requirements.txt    # Python dependencies
└── README.md
```

## Tech Stack

- **Streamlit** — interactive UI
- **Pandas / NumPy / SciPy** — data analysis
- **scikit-learn** — Isolation Forest outlier detection
- **Plotly** — interactive visualizations
