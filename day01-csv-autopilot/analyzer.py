"""
CSV Autopilot — Core Analysis Engine
Handles type inference, statistical profiling, correlation analysis,
outlier detection, and missing-value pattern analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Type Inference ──────────────────────────────────────────────────────────

SEMANTIC_TYPES = [
    "boolean", "integer", "float", "categorical",
    "datetime", "text", "identifier",
]


def infer_semantic_type(series: pd.Series) -> str:
    """Infer the semantic type of a pandas Series beyond its raw dtype."""
    s = series.dropna()
    if s.empty:
        return "empty"

    if s.dtype == "bool" or set(s.unique()) <= {0, 1, True, False, "0", "1"}:
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique()
        if np.issubdtype(s.dtype, np.integer) and nunique < 20:
            return "categorical"
        if np.issubdtype(s.dtype, np.integer):
            return "integer"
        return "float"

    # string-like columns
    try:
        pd.to_datetime(s, infer_datetime_format=True)
        return "datetime"
    except (ValueError, TypeError):
        pass

    nunique = s.nunique()
    ratio = nunique / len(s) if len(s) > 0 else 0

    if ratio > 0.9:
        return "identifier"
    if nunique <= 50 or ratio < 0.05:
        return "categorical"
    return "text"


def build_type_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame mapping each column to its inferred semantic type."""
    rows = []
    for col in df.columns:
        sem_type = infer_semantic_type(df[col])
        rows.append({
            "column": col,
            "pandas_dtype": str(df[col].dtype),
            "semantic_type": sem_type,
            "nunique": df[col].nunique(),
            "pct_missing": round(df[col].isna().mean() * 100, 2),
        })
    return pd.DataFrame(rows)


# ── Statistical Profiling ───────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    stats: dict[str, Any] = field(default_factory=dict)


def profile_numeric(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    desc = s.describe()
    return {
        "mean": round(float(desc["mean"]), 4),
        "std": round(float(desc["std"]), 4),
        "min": float(desc["min"]),
        "25%": float(desc["25%"]),
        "50%": float(desc["50%"]),
        "75%": float(desc["75%"]),
        "max": float(desc["max"]),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurtosis()), 4),
        "cv": round(float(s.std() / s.mean()), 4) if s.mean() != 0 else None,
    }


def profile_categorical(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    vc = s.value_counts()
    return {
        "mode": str(vc.index[0]),
        "mode_count": int(vc.iloc[0]),
        "mode_pct": round(float(vc.iloc[0] / len(s) * 100), 2),
        "top_5": vc.head(5).to_dict(),
        "entropy": round(float(stats.entropy(vc.values / vc.values.sum())), 4),
    }


def profile_column(series: pd.Series, sem_type: str) -> ColumnProfile:
    numeric_types = {"integer", "float", "boolean"}
    stat_fn = profile_numeric if sem_type in numeric_types else profile_categorical

    return ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        semantic_type=sem_type,
        count=len(series),
        missing=int(series.isna().sum()),
        missing_pct=round(series.isna().mean() * 100, 2),
        unique=series.nunique(),
        stats=stat_fn(series),
    )


def profile_dataframe(df: pd.DataFrame) -> list[ColumnProfile]:
    type_report = build_type_report(df)
    profiles = []
    for _, row in type_report.iterrows():
        col, sem_type = row["column"], row["semantic_type"]
        profiles.append(profile_column(df[col], sem_type))
    return profiles


# ── Correlation Analysis ────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute Pearson, Spearman, and Kendall correlations for numeric cols."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {}
    return {
        "pearson": numeric.corr(method="pearson").round(3),
        "spearman": numeric.corr(method="spearman").round(3),
    }


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list[dict]:
    """Return pairs of features with |correlation| >= threshold."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(val, 4),
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


# ── Outlier Detection ──────────────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Flag outliers using the IQR method."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag outliers using z-score method."""
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z > threshold
    return mask


def detect_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """Run Isolation Forest on all numeric columns."""
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[0] < 10 or numeric.shape[1] < 1:
        return pd.Series(False, index=df.index)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(numeric)
    mask = pd.Series(False, index=df.index)
    mask.loc[numeric.index] = preds == -1
    return mask


def outlier_report(df: pd.DataFrame) -> dict:
    """Produce a combined outlier report across methods."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    col_reports = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        iqr_mask = detect_outliers_iqr(df[col])
        z_mask = detect_outliers_zscore(df[col])
        col_reports[col] = {
            "iqr_outliers": int(iqr_mask.sum()),
            "zscore_outliers": int(z_mask.sum()),
            "iqr_pct": round(float(iqr_mask.mean() * 100), 2),
            "zscore_pct": round(float(z_mask.mean() * 100), 2),
        }

    iso_mask = detect_outliers_isolation_forest(df)
    return {
        "per_column": col_reports,
        "isolation_forest_total": int(iso_mask.sum()),
        "isolation_forest_pct": round(float(iso_mask.mean() * 100), 2),
        "isolation_forest_mask": iso_mask,
    }


# ── Missing Value Patterns ─────────────────────────────────────────────────

def missing_value_analysis(df: pd.DataFrame) -> dict:
    """Analyze missing value patterns: counts, co-occurrence, and row completeness."""
    missing_counts = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())

    # co-occurrence: which columns tend to be missing together
    missing_binary = df.isna().astype(int)
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
    co_occurrence = None
    if len(cols_with_missing) >= 2:
        co_occurrence = missing_binary[cols_with_missing].corr().round(3)

    # row completeness
    row_completeness = (1 - df.isna().mean(axis=1)) * 100
    completeness_bins = pd.cut(
        row_completeness,
        bins=[0, 25, 50, 75, 99.9, 100],
        labels=["0-25%", "25-50%", "50-75%", "75-99%", "100%"],
    ).value_counts().sort_index()

    return {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "total_missing_pct": round(total_missing / total_cells * 100, 2) if total_cells else 0,
        "per_column": pd.DataFrame({"missing": missing_counts, "pct": missing_pct}).to_dict("index"),
        "co_occurrence": co_occurrence,
        "row_completeness_distribution": completeness_bins.to_dict(),
        "fully_complete_rows": int((row_completeness == 100).sum()),
        "fully_complete_rows_pct": round(float((row_completeness == 100).mean() * 100), 2),
    }


# ── Full Analysis Orchestrator ─────────────────────────────────────────────

@dataclass
class FullReport:
    shape: tuple[int, int]
    memory_mb: float
    type_report: pd.DataFrame
    profiles: list[ColumnProfile]
    correlations: dict[str, pd.DataFrame]
    high_correlations: list[dict]
    outlier_report: dict
    missing_analysis: dict
    duplicate_rows: int
    duplicate_pct: float


def run_full_analysis(df: pd.DataFrame) -> FullReport:
    """Run the complete EDA pipeline on a DataFrame."""
    type_rep = build_type_report(df)
    profiles = profile_dataframe(df)
    corrs = compute_correlations(df)
    high_corrs = []
    if "pearson" in corrs:
        high_corrs = find_high_correlations(corrs["pearson"])
    outliers = outlier_report(df)
    missing = missing_value_analysis(df)
    dupes = int(df.duplicated().sum())

    return FullReport(
        shape=df.shape,
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        type_report=type_rep,
        profiles=profiles,
        correlations=corrs,
        high_correlations=high_corrs,
        outlier_report=outliers,
        missing_analysis=missing,
        duplicate_rows=dupes,
        duplicate_pct=round(dupes / len(df) * 100, 2) if len(df) else 0,
    )
