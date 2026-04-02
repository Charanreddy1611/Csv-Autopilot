"""
CSV Autopilot — Report Generator
Builds a self-contained HTML report (with embedded Plotly charts as base64 images)
and optionally converts to PDF.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from analyzer import FullReport
from visualizations import (
    plot_histogram,
    plot_categorical_bar,
    plot_correlation_heatmap,
    plot_outlier_overview,
    plot_missing_bar,
    plot_missing_matrix,
    plot_box_strip,
)


def _fig_to_base64(fig: go.Figure, width: int = 900, height: int = 450) -> str:
    """Render a Plotly figure to a base64-encoded PNG string."""
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")


def _fig_to_html_interactive(fig: go.Figure) -> str:
    """Render a Plotly figure as an inline interactive HTML div."""
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_stat_table(profiles: list, max_cols: int = 50) -> str:
    """Build an HTML table summarizing column profiles."""
    rows_html = ""
    for p in profiles[:max_cols]:
        missing_cls = ' class="warn"' if p.missing_pct > 5 else ""
        key_stats = ""
        if p.semantic_type in ("integer", "float", "boolean") and p.stats:
            parts = []
            for k in ("mean", "std", "skewness", "kurtosis"):
                if k in p.stats and p.stats[k] is not None:
                    parts.append(f"{k}: {p.stats[k]}")
            key_stats = " · ".join(parts)
        elif p.stats:
            if "mode" in p.stats:
                key_stats = f'mode: {p.stats["mode"]}'
            if "entropy" in p.stats:
                key_stats += f' · entropy: {p.stats["entropy"]}'

        rows_html += f"""
        <tr>
            <td><strong>{p.name}</strong></td>
            <td>{p.semantic_type}</td>
            <td>{p.dtype}</td>
            <td>{p.unique}</td>
            <td{missing_cls}>{p.missing} ({p.missing_pct}%)</td>
            <td class="small">{key_stats}</td>
        </tr>"""

    return f"""
    <table>
        <thead>
            <tr>
                <th>Column</th><th>Semantic Type</th><th>Dtype</th>
                <th>Unique</th><th>Missing</th><th>Key Statistics</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def _build_correlation_pairs_table(pairs: list[dict]) -> str:
    if not pairs:
        return "<p>No highly correlated pairs found (|r| &ge; 0.8).</p>"
    rows = ""
    for p in pairs:
        rows += f'<tr><td>{p["feature_1"]}</td><td>{p["feature_2"]}</td><td>{p["correlation"]}</td></tr>'
    return f"""
    <table>
        <thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _build_outlier_table(outlier_data: dict) -> str:
    per_col = outlier_data.get("per_column", {})
    if not per_col:
        return "<p>No numeric columns for outlier analysis.</p>"
    rows = ""
    for col, d in per_col.items():
        rows += f'<tr><td>{col}</td><td>{d["iqr_outliers"]} ({d["iqr_pct"]}%)</td><td>{d["zscore_outliers"]} ({d["zscore_pct"]}%)</td></tr>'
    iso_total = outlier_data.get("isolation_forest_total", 0)
    iso_pct = outlier_data.get("isolation_forest_pct", 0)
    return f"""
    <table>
        <thead><tr><th>Column</th><th>IQR Outliers</th><th>Z-Score Outliers</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <p><strong>Isolation Forest:</strong> {iso_total} global outliers ({iso_pct}%) across all numeric features.</p>"""


def generate_html_report(
    df: pd.DataFrame,
    report: FullReport,
    filename: str = "dataset",
    interactive_charts: bool = True,
) -> str:
    """Generate a complete HTML EDA report.

    Args:
        df: The analyzed DataFrame.
        report: The FullReport from run_full_analysis.
        filename: Original file name for the report header.
        interactive_charts: If True, embed interactive Plotly charts.
                           If False, embed static PNG images (needed for PDF).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Build chart sections ────────────────────────────────────────────
    charts_html = ""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [p.name for p in report.profiles if p.semantic_type in ("categorical", "boolean")]

    def embed(fig: go.Figure, w=900, h=450) -> str:
        if interactive_charts:
            return f'<div class="chart">{_fig_to_html_interactive(fig)}</div>'
        b64 = _fig_to_base64(fig, w, h)
        return f'<div class="chart"><img src="data:image/png;base64,{b64}" /></div>'

    # Distribution charts (first 10 numeric, first 5 categorical)
    dist_charts = ""
    for col in numeric_cols[:10]:
        dist_charts += embed(plot_histogram(df[col]))
    for col in cat_cols[:5]:
        dist_charts += embed(plot_categorical_bar(df[col]))

    # Correlation heatmap
    corr_chart = ""
    if "pearson" in report.correlations:
        corr_chart = embed(plot_correlation_heatmap(report.correlations["pearson"]), w=800, h=700)

    # Outlier overview
    outlier_chart = embed(plot_outlier_overview(report.outlier_report))

    # Missing value charts
    missing_bar_chart = embed(plot_missing_bar(df))
    missing_matrix_chart = embed(plot_missing_matrix(df), w=900, h=500)

    # Box plots for top 6 numeric
    box_charts = ""
    for col in numeric_cols[:6]:
        box_charts += embed(plot_box_strip(df[col]))

    # ── Missing value summary ───────────────────────────────────────────
    ma = report.missing_analysis
    missing_summary = f"""
    <div class="stats-row">
        <div class="stat-box"><div class="stat-val">{ma['total_missing']:,}</div><div class="stat-label">Total Missing Cells</div></div>
        <div class="stat-box"><div class="stat-val">{ma['total_missing_pct']}%</div><div class="stat-label">Overall Missing %</div></div>
        <div class="stat-box"><div class="stat-val">{ma['fully_complete_rows']:,}</div><div class="stat-label">Fully Complete Rows</div></div>
        <div class="stat-box"><div class="stat-val">{ma['fully_complete_rows_pct']}%</div><div class="stat-label">Complete Rows %</div></div>
    </div>"""

    plotly_js = ""
    if interactive_charts:
        plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report — {filename}</title>
    {plotly_js}
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #f5f7fa; color: #2d3748; line-height: 1.6;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 20px 30px; }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px 30px; border-radius: 12px; margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 5px; }}
        .header p {{ opacity: 0.85; font-size: 0.95rem; }}
        .section {{
            background: white; border-radius: 10px; padding: 25px 30px;
            margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            font-size: 1.35rem; color: #4a5568; margin-bottom: 16px;
            padding-bottom: 8px; border-bottom: 2px solid #667eea;
        }}
        .stats-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
        .stat-box {{
            flex: 1; min-width: 140px; background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 10px; padding: 18px; text-align: center; color: white;
        }}
        .stat-val {{ font-size: 1.6rem; font-weight: 700; }}
        .stat-label {{ font-size: 0.8rem; opacity: 0.85; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9rem; }}
        th {{ background: #667eea; color: white; padding: 10px 14px; text-align: left; }}
        td {{ padding: 8px 14px; border-bottom: 1px solid #e2e8f0; }}
        tr:nth-child(even) {{ background: #f7fafc; }}
        .warn {{ color: #e53e3e; font-weight: 600; }}
        .small {{ font-size: 0.8rem; color: #718096; }}
        .chart {{ margin: 16px 0; text-align: center; }}
        .chart img {{ max-width: 100%; height: auto; border-radius: 6px; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .footer {{ text-align: center; color: #a0aec0; font-size: 0.8rem; padding: 20px; }}
        @media print {{
            body {{ background: white; }}
            .section {{ box-shadow: none; break-inside: avoid; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>EDA Report — {filename}</h1>
        <p>Generated on {timestamp} by CSV Autopilot</p>
    </div>

    <!-- Overview -->
    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="stats-row">
            <div class="stat-box"><div class="stat-val">{report.shape[0]:,}</div><div class="stat-label">Rows</div></div>
            <div class="stat-box"><div class="stat-val">{report.shape[1]}</div><div class="stat-label">Columns</div></div>
            <div class="stat-box" style="background:linear-gradient(135deg,#4facfe,#00f2fe)"><div class="stat-val">{report.memory_mb} MB</div><div class="stat-label">Memory</div></div>
            <div class="stat-box" style="background:linear-gradient(135deg,#43e97b,#38f9d7)"><div class="stat-val">{report.duplicate_rows:,}</div><div class="stat-label">Duplicate Rows ({report.duplicate_pct}%)</div></div>
        </div>
    </div>

    <!-- Column Profiles -->
    <div class="section">
        <h2>Column Profiles</h2>
        {_build_stat_table(report.profiles)}
    </div>

    <!-- Distributions -->
    <div class="section">
        <h2>Distributions</h2>
        <div class="chart-grid">{dist_charts}</div>
    </div>

    <!-- Correlations -->
    <div class="section">
        <h2>Correlation Analysis</h2>
        {corr_chart}
        <h3 style="margin-top:16px;font-size:1.1rem;">Highly Correlated Pairs</h3>
        {_build_correlation_pairs_table(report.high_correlations)}
    </div>

    <!-- Outliers -->
    <div class="section">
        <h2>Outlier Detection</h2>
        {outlier_chart}
        {_build_outlier_table(report.outlier_report)}
    </div>

    <!-- Box Plots -->
    <div class="section">
        <h2>Box Plots (Numeric Columns)</h2>
        <div class="chart-grid">{box_charts}</div>
    </div>

    <!-- Missing Values -->
    <div class="section">
        <h2>Missing Value Analysis</h2>
        {missing_summary}
        {missing_bar_chart}
        {missing_matrix_chart}
    </div>

    <div class="footer">
        CSV Autopilot · Generated {timestamp}
    </div>

</div>
</body>
</html>"""

    return html


def html_to_pdf(html_content: str) -> bytes | None:
    """Convert HTML string to PDF bytes using available backend.

    Tries pdfkit (wkhtmltopdf) first, then falls back to weasyprint.
    Returns None if neither is available.
    """
    # Try pdfkit
    try:
        import pdfkit
        options = {
            "page-size": "A4",
            "orientation": "Landscape",
            "margin-top": "10mm",
            "margin-bottom": "10mm",
            "margin-left": "10mm",
            "margin-right": "10mm",
            "encoding": "UTF-8",
            "enable-local-file-access": "",
            "no-outline": None,
        }
        return pdfkit.from_string(html_content, False, options=options)
    except Exception:
        pass

    # Try weasyprint
    try:
        from weasyprint import HTML as WeasyprintHTML
        buf = io.BytesIO()
        WeasyprintHTML(string=html_content).write_pdf(buf)
        return buf.getvalue()
    except Exception:
        pass

    return None
