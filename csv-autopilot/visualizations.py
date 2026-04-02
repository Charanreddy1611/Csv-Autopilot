"""
CSV Autopilot — Visualization Module
All charts are Plotly-based for interactivity inside Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

PALETTE = px.colors.qualitative.Set2


# ── Distribution Plots ──────────────────────────────────────────────────────

def plot_histogram(series: pd.Series, nbins: int = 50) -> go.Figure:
    s = series.dropna()
    fig = px.histogram(
        s, nbins=nbins, marginal="box",
        title=f"Distribution — {series.name}",
        color_discrete_sequence=[PALETTE[0]],
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


def plot_categorical_bar(series: pd.Series, top_n: int = 20) -> go.Figure:
    vc = series.value_counts().head(top_n)
    fig = px.bar(
        x=vc.index.astype(str), y=vc.values,
        title=f"Top {min(top_n, len(vc))} Values — {series.name}",
        labels={"x": series.name, "y": "Count"},
        color_discrete_sequence=[PALETTE[1]],
        template="plotly_white",
    )
    fig.update_layout(height=420)
    return fig


def plot_qq(series: pd.Series) -> go.Figure:
    """Q-Q plot against normal distribution."""
    s = series.dropna().values
    theoretical = np.sort(sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(s))))
    observed = np.sort(s)
    # subsample if too large
    if len(observed) > 2000:
        idx = np.linspace(0, len(observed) - 1, 2000, dtype=int)
        theoretical = theoretical[idx]
        observed = observed[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=observed, mode="markers",
                             marker=dict(size=3, color=PALETTE[2]), name="Data"))
    mn, mx = min(theoretical.min(), observed.min()), max(theoretical.max(), observed.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Normal"))
    fig.update_layout(
        title=f"Q-Q Plot — {series.name}", template="plotly_white",
        xaxis_title="Theoretical Quantiles", yaxis_title="Observed Quantiles",
        height=420, showlegend=False,
    )
    return fig


# ── Correlation Heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Pearson Correlation") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        title=title, template="plotly_white",
        height=max(400, 50 * len(corr_matrix.columns)),
        width=max(500, 50 * len(corr_matrix.columns)),
    )
    return fig


def plot_scatter_pair(df: pd.DataFrame, col_x: str, col_y: str) -> go.Figure:
    fig = px.scatter(
        df, x=col_x, y=col_y,
        trendline="ols",
        title=f"{col_x} vs {col_y}",
        color_discrete_sequence=[PALETTE[3]],
        template="plotly_white",
    )
    fig.update_layout(height=420)
    return fig


# ── Outlier Visualization ──────────────────────────────────────────────────

def plot_box_strip(series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=series.dropna(), name=series.name,
        boxpoints="outliers", marker_color=PALETTE[0],
        line_color=PALETTE[0],
    ))
    fig.update_layout(
        title=f"Box Plot — {series.name}", template="plotly_white", height=420,
    )
    return fig


def plot_outlier_overview(outlier_data: dict) -> go.Figure:
    """Bar chart showing outlier % per column (IQR vs Z-score)."""
    per_col = outlier_data.get("per_column", {})
    if not per_col:
        return go.Figure().update_layout(title="No numeric columns for outlier analysis")

    cols = list(per_col.keys())
    iqr_pcts = [per_col[c]["iqr_pct"] for c in cols]
    z_pcts = [per_col[c]["zscore_pct"] for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="IQR Method", x=cols, y=iqr_pcts, marker_color=PALETTE[4]))
    fig.add_trace(go.Bar(name="Z-Score Method", x=cols, y=z_pcts, marker_color=PALETTE[5]))
    fig.update_layout(
        barmode="group", title="Outlier Percentage by Column",
        yaxis_title="% Outliers", template="plotly_white", height=420,
    )
    return fig


# ── Missing Value Visualization ─────────────────────────────────────────────

def plot_missing_matrix(df: pd.DataFrame) -> go.Figure:
    """Nullity heatmap — white = present, colored = missing."""
    sample = df.sample(min(len(df), 500), random_state=42).sort_index() if len(df) > 500 else df
    missing = sample.isna().astype(int)

    fig = go.Figure(data=go.Heatmap(
        z=missing.values,
        x=missing.columns.tolist(),
        y=list(range(len(missing))),
        colorscale=[[0, "#eaeaea"], [1, "#e45756"]],
        showscale=False,
    ))
    fig.update_layout(
        title="Missing Value Matrix (white = present, red = missing)",
        template="plotly_white",
        yaxis=dict(title="Row Index", autorange="reversed"),
        height=max(400, len(missing) * 0.8),
    )
    return fig


def plot_missing_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of missing % per column."""
    pct = (df.isna().mean() * 100).sort_values(ascending=True)
    pct = pct[pct > 0]
    if pct.empty:
        fig = go.Figure()
        fig.update_layout(title="No Missing Values Detected")
        return fig

    colors = [PALETTE[4] if v < 10 else (PALETTE[1] if v < 40 else "#e45756") for v in pct.values]
    fig = go.Figure(go.Bar(
        x=pct.values, y=pct.index.tolist(),
        orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in pct.values], textposition="outside",
    ))
    fig.update_layout(
        title="Missing Values by Column (%)", template="plotly_white",
        xaxis_title="% Missing", height=max(300, len(pct) * 35),
    )
    return fig


def plot_missing_co_occurrence(co_occ: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=co_occ.values,
        x=co_occ.columns.tolist(),
        y=co_occ.index.tolist(),
        colorscale="Purples",
        zmin=-1, zmax=1,
        text=co_occ.values.round(2),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Missing Value Co-occurrence (correlation of missingness)",
        template="plotly_white",
        height=max(350, 50 * len(co_occ)),
    )
    return fig


def plot_row_completeness(dist: dict) -> go.Figure:
    labels = list(dist.keys())
    values = list(dist.values())
    fig = px.pie(
        names=labels, values=values,
        title="Row Completeness Distribution",
        color_discrete_sequence=PALETTE,
        template="plotly_white",
    )
    fig.update_layout(height=400)
    return fig


# ── Imputation Before / After ───────────────────────────────────────────────

def plot_before_after_histogram(
    original: pd.Series,
    imputed: pd.Series,
    col_name: str,
) -> go.Figure:
    """Overlaid histograms comparing a column before and after imputation."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=original.dropna(), name="Before (non-null)",
        marker_color=PALETTE[0], opacity=0.6, nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=imputed, name="After Imputation",
        marker_color=PALETTE[3], opacity=0.6, nbinsx=50,
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Before vs After — {col_name}",
        template="plotly_white", height=420,
        xaxis_title=col_name, yaxis_title="Count",
    )
    return fig


def plot_before_after_bar(
    original: pd.Series,
    imputed: pd.Series,
    col_name: str,
    top_n: int = 15,
) -> go.Figure:
    """Side-by-side bar chart for categorical columns before/after imputation."""
    vc_before = original.dropna().value_counts().head(top_n)
    vc_after = imputed.value_counts().head(top_n)
    all_cats = list(dict.fromkeys(list(vc_before.index) + list(vc_after.index)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in all_cats],
        y=[vc_before.get(c, 0) for c in all_cats],
        name="Before", marker_color=PALETTE[0],
    ))
    fig.add_trace(go.Bar(
        x=[str(c) for c in all_cats],
        y=[vc_after.get(c, 0) for c in all_cats],
        name="After", marker_color=PALETTE[3],
    ))
    fig.update_layout(
        barmode="group",
        title=f"Before vs After — {col_name}",
        template="plotly_white", height=420,
        xaxis_title=col_name, yaxis_title="Count",
    )
    return fig


def plot_imputation_summary(missing_before: dict, missing_after: dict) -> go.Figure:
    """Bar chart showing missing count per column before vs after imputation."""
    cols = list(missing_before.keys())
    before_vals = [missing_before[c] for c in cols]
    after_vals = [missing_after.get(c, 0) for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cols, y=before_vals, name="Before", marker_color="#e45756"))
    fig.add_trace(go.Bar(x=cols, y=after_vals, name="After", marker_color="#43e97b"))
    fig.update_layout(
        barmode="group",
        title="Missing Values: Before vs After Imputation",
        yaxis_title="Missing Count",
        template="plotly_white", height=420,
    )
    return fig
