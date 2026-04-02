"""
CSV Autopilot — Drop a CSV, get an instant EDA report.
Drop a CSV → get an instant, thorough EDA report.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from analyzer import (
    run_full_analysis,
    FullReport,
    detect_outliers_iqr,
    detect_outliers_zscore,
    IMPUTATION_STRATEGIES,
    impute_column,
    impute_multiple,
    suggest_strategy,
    imputation_impact_stats,
    infer_semantic_type,
)
from visualizations import (
    plot_histogram,
    plot_categorical_bar,
    plot_qq,
    plot_correlation_heatmap,
    plot_scatter_pair,
    plot_box_strip,
    plot_outlier_overview,
    plot_missing_matrix,
    plot_missing_bar,
    plot_missing_co_occurrence,
    plot_row_completeness,
    plot_before_after_histogram,
    plot_before_after_bar,
    plot_imputation_summary,
)

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CSV Autopilot",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin-bottom: 10px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    .section-header {
        border-left: 4px solid #667eea; padding-left: 12px;
        margin-top: 1.5rem; margin-bottom: 0.8rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value, color_idx: int = 0):
    colors = [
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    ]
    bg = colors[color_idx % len(colors)]
    st.markdown(
        f'<div class="metric-card" style="background:{bg}">'
        f'<h2>{value}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛩️ CSV Autopilot")
    st.caption("30 Vibe-Coded Projects")
    st.divider()
    uploaded = st.file_uploader("Upload a CSV file", type=["csv", "tsv", "txt"])
    st.divider()
    if uploaded:
        sep = st.selectbox("Delimiter", [",", ";", "\\t", "|"], index=0)
        sep = "\t" if sep == "\\t" else sep
        sample_rows = st.slider("Preview rows", 5, 100, 20)
    st.divider()
    st.markdown("Built with Streamlit + Plotly")


# ── Main ────────────────────────────────────────────────────────────────────

if not uploaded:
    st.markdown("## 🛩️ CSV Autopilot")
    st.markdown(
        "> **Drop a CSV** in the sidebar and get an instant, interactive EDA report — "
        "type inference, distributions, correlations, outlier detection, and missing-value analysis."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔍 Smart Type Detection")
        st.markdown("Goes beyond pandas dtypes — detects booleans, identifiers, datetime strings, "
                     "categorical ints, and free text.")
    with col2:
        st.markdown("#### 📊 Deep Profiling")
        st.markdown("Skewness, kurtosis, CV, entropy, top values, Q-Q plots — "
                     "all computed automatically for every column.")
    with col3:
        st.markdown("#### 🚨 Outlier & Missing Analysis")
        st.markdown("IQR, Z-score, and Isolation Forest outlier detection. "
                     "Missing-value co-occurrence and row completeness breakdown.")
    st.stop()


# ── Load Data ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Parsing CSV…")
def load(file_bytes: bytes, separator: str) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), sep=separator, low_memory=False)


df = load(uploaded.getvalue(), sep)


@st.cache_data(show_spinner="Running full analysis…")
def analyze(file_bytes: bytes, separator: str) -> FullReport:
    from io import BytesIO
    _df = pd.read_csv(BytesIO(file_bytes), sep=separator, low_memory=False)
    return run_full_analysis(_df)


report = analyze(uploaded.getvalue(), sep)


# ── Overview Cards ──────────────────────────────────────────────────────────

st.markdown("## Analysis Report")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: metric_card("Rows", f"{report.shape[0]:,}", 0)
with c2: metric_card("Columns", report.shape[1], 1)
with c3: metric_card("Memory", f"{report.memory_mb} MB", 2)
with c4: metric_card("Duplicates", f"{report.duplicate_rows:,} ({report.duplicate_pct}%)", 3)
with c5: metric_card("Missing Cells", f"{report.missing_analysis['total_missing']:,}", 4)
with c6: metric_card("Missing %", f"{report.missing_analysis['total_missing_pct']}%", 5)


# ── Tabs ────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📋 Data Preview",
    "🧬 Type Report",
    "📊 Distributions",
    "🔗 Correlations",
    "🚨 Outliers",
    "❓ Missing Values",
    "🔧 Impute & Clean",
    "📈 Column Deep-Dive",
])

# ── Tab 0: Data Preview ────────────────────────────────────────────────────

with tabs[0]:
    st.dataframe(df.head(sample_rows), use_container_width=True, height=500)

# ── Tab 1: Type Report ─────────────────────────────────────────────────────

with tabs[1]:
    st.markdown('<div class="section-header"><h3>Inferred Column Types</h3></div>', unsafe_allow_html=True)
    type_df = report.type_report.copy()

    type_color = {
        "integer": "🟦", "float": "🟩", "categorical": "🟧",
        "boolean": "🟪", "datetime": "🟫", "text": "🟨", "identifier": "⬜",
    }
    type_df["icon"] = type_df["semantic_type"].map(lambda t: type_color.get(t, "⬛"))
    type_df["display_type"] = type_df["icon"] + " " + type_df["semantic_type"]
    st.dataframe(
        type_df[["column", "display_type", "pandas_dtype", "nunique", "pct_missing"]],
        use_container_width=True, hide_index=True, height=min(600, 40 + 35 * len(type_df)),
    )

    st.markdown("**Type breakdown:**")
    breakdown = type_df["semantic_type"].value_counts()
    cols = st.columns(min(len(breakdown), 6))
    for i, (t, cnt) in enumerate(breakdown.items()):
        with cols[i % len(cols)]:
            st.metric(f"{type_color.get(t, '')} {t}", cnt)

# ── Tab 2: Distributions ───────────────────────────────────────────────────

with tabs[2]:
    st.markdown('<div class="section-header"><h3>Column Distributions</h3></div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [p.name for p in report.profiles if p.semantic_type in ("categorical", "boolean", "text")]

    sub_tab = st.radio("Column kind", ["Numeric", "Categorical"], horizontal=True)
    if sub_tab == "Numeric":
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            sel = st.selectbox("Select numeric column", numeric_cols, key="dist_num")
            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(plot_histogram(df[sel]), use_container_width=True)
            with col_r:
                st.plotly_chart(plot_qq(df[sel]), use_container_width=True)

            profile = next((p for p in report.profiles if p.name == sel), None)
            if profile and profile.stats:
                st.markdown("**Statistics**")
                stat_cols = st.columns(5)
                items = list(profile.stats.items())
                for i, (k, v) in enumerate(items):
                    with stat_cols[i % 5]:
                        st.metric(k, v if v is not None else "N/A")
    else:
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            sel = st.selectbox("Select categorical column", cat_cols, key="dist_cat")
            st.plotly_chart(plot_categorical_bar(df[sel]), use_container_width=True)
            profile = next((p for p in report.profiles if p.name == sel), None)
            if profile and profile.stats:
                st.markdown("**Statistics**")
                scols = st.columns(4)
                for i, (k, v) in enumerate(list(profile.stats.items())[:8]):
                    if k == "top_5":
                        continue
                    with scols[i % 4]:
                        st.metric(k, v)

# ── Tab 3: Correlations ────────────────────────────────────────────────────

with tabs[3]:
    st.markdown('<div class="section-header"><h3>Correlation Analysis</h3></div>', unsafe_allow_html=True)
    if not report.correlations:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        corr_mat = report.correlations[method]
        st.plotly_chart(plot_correlation_heatmap(corr_mat, title=f"{method.title()} Correlation"),
                        use_container_width=True)

        if report.high_correlations:
            st.markdown("**Highly correlated pairs (|r| >= 0.8):**")
            for pair in report.high_correlations:
                emoji = "🔴" if abs(pair["correlation"]) >= 0.95 else "🟡"
                st.write(f'{emoji} **{pair["feature_1"]}** ↔ **{pair["feature_2"]}**: `{pair["correlation"]}`')

            st.markdown("---")
            st.markdown("**Scatter plot for a pair:**")
            pair_opts = [f'{p["feature_1"]} vs {p["feature_2"]}' for p in report.high_correlations]
            if pair_opts:
                chosen = st.selectbox("Select pair", pair_opts)
                idx = pair_opts.index(chosen)
                p = report.high_correlations[idx]
                st.plotly_chart(plot_scatter_pair(df, p["feature_1"], p["feature_2"]),
                                use_container_width=True)

# ── Tab 4: Outliers ────────────────────────────────────────────────────────

with tabs[4]:
    st.markdown('<div class="section-header"><h3>Outlier Detection</h3></div>', unsafe_allow_html=True)
    st.plotly_chart(plot_outlier_overview(report.outlier_report), use_container_width=True)

    iso_total = report.outlier_report.get("isolation_forest_total", 0)
    iso_pct = report.outlier_report.get("isolation_forest_pct", 0)
    st.info(f"**Isolation Forest** flagged **{iso_total}** rows ({iso_pct}%) as global outliers "
            f"(across all numeric features jointly).")

    if numeric_cols:
        st.markdown("---")
        sel_out = st.selectbox("Deep-dive column", numeric_cols, key="outlier_col")
        ocol1, ocol2 = st.columns(2)
        with ocol1:
            st.plotly_chart(plot_box_strip(df[sel_out]), use_container_width=True)
        with ocol2:
            st.plotly_chart(plot_histogram(df[sel_out], nbins=80), use_container_width=True)

        iqr_mask = detect_outliers_iqr(df[sel_out])
        z_mask = detect_outliers_zscore(df[sel_out])
        combined = iqr_mask | z_mask
        if combined.any():
            with st.expander(f"View {combined.sum()} outlier rows"):
                st.dataframe(df[combined].head(200), use_container_width=True)

# ── Tab 5: Missing Values ──────────────────────────────────────────────────

with tabs[5]:
    st.markdown('<div class="section-header"><h3>Missing Value Analysis</h3></div>', unsafe_allow_html=True)
    ma = report.missing_analysis

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Total Missing Cells", f"{ma['total_missing']:,}")
    with m2: st.metric("Overall Missing %", f"{ma['total_missing_pct']}%")
    with m3: st.metric("Fully Complete Rows", f"{ma['fully_complete_rows']:,} ({ma['fully_complete_rows_pct']}%)")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.plotly_chart(plot_missing_bar(df), use_container_width=True)
    with mcol2:
        st.plotly_chart(plot_row_completeness(ma["row_completeness_distribution"]),
                        use_container_width=True)

    st.plotly_chart(plot_missing_matrix(df), use_container_width=True)

    if ma["co_occurrence"] is not None:
        st.markdown("**Missingness co-occurrence** — high values mean columns tend to be missing together:")
        st.plotly_chart(plot_missing_co_occurrence(ma["co_occurrence"]), use_container_width=True)

# ── Tab 6: Impute & Clean ──────────────────────────────────────────────────

with tabs[6]:
    st.markdown('<div class="section-header"><h3>Impute Missing Values</h3></div>', unsafe_allow_html=True)

    cols_with_missing = [
        col for col in df.columns if df[col].isna().sum() > 0
    ]

    if not cols_with_missing:
        st.success("No missing values found in this dataset — nothing to impute!")
    else:
        st.markdown(
            f"**{len(cols_with_missing)}** column(s) have missing values. "
            "Configure a strategy per column, preview the result, and download the cleaned CSV."
        )

        imp_mode = st.radio(
            "Imputation mode",
            ["Single Column", "Bulk (all missing columns)"],
            horizontal=True,
            key="imp_mode",
        )

        if imp_mode == "Single Column":
            sel_col = st.selectbox(
                "Select column to impute",
                cols_with_missing,
                key="imp_col",
            )
            n_miss = int(df[sel_col].isna().sum())
            pct_miss = round(n_miss / len(df) * 100, 2)
            sem_type = infer_semantic_type(df[sel_col])
            suggested = suggest_strategy(df[sel_col], sem_type)

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.metric("Missing", f"{n_miss:,} ({pct_miss}%)")
            with ic2:
                st.metric("Type", sem_type)
            with ic3:
                st.metric("Suggested Strategy", suggested)

            is_numeric = pd.api.types.is_numeric_dtype(df[sel_col])
            available = IMPUTATION_STRATEGIES.copy()
            if not is_numeric:
                available = [s for s in available if s not in ("Mean", "Median", "Interpolate (Linear)")]

            default_idx = available.index(suggested) if suggested in available else 0
            strategy = st.selectbox(
                "Imputation strategy",
                available,
                index=default_idx,
                key="imp_strategy",
            )

            custom_val = None
            if strategy == "Custom Value":
                custom_val = st.text_input("Enter fill value", key="imp_custom")
                if is_numeric and custom_val:
                    try:
                        custom_val = float(custom_val)
                    except ValueError:
                        st.warning("Column is numeric — enter a valid number.")
                        custom_val = None

            if st.button("Apply Imputation", type="primary", key="imp_apply"):
                imputed_df = impute_column(df, sel_col, strategy, custom_val)
                st.session_state["imputed_df"] = imputed_df
                st.session_state["imp_col_name"] = sel_col
                st.session_state["imp_strategy_name"] = strategy

            if "imputed_df" in st.session_state and st.session_state.get("imp_col_name") == sel_col:
                imputed_df = st.session_state["imputed_df"]
                strat_name = st.session_state["imp_strategy_name"]

                st.markdown(f"---")
                st.markdown(f"**Result: `{sel_col}` imputed with `{strat_name}`**")

                impact = imputation_impact_stats(df[sel_col], imputed_df[sel_col])
                ri1, ri2, ri3, ri4 = st.columns(4)
                with ri1:
                    st.metric("Missing Before", impact["missing_before"])
                with ri2:
                    st.metric("Missing After", impact["missing_after"],
                              delta=impact["missing_after"] - impact["missing_before"])
                with ri3:
                    st.metric("Rows Before", impact["rows_before"])
                with ri4:
                    st.metric("Rows After", impact["rows_after"],
                              delta=impact["rows_after"] - impact["rows_before"]
                              if impact["rows_after"] != impact["rows_before"] else None)

                if "mean_before" in impact:
                    si1, si2 = st.columns(2)
                    with si1:
                        st.metric("Mean Before", impact["mean_before"])
                        st.metric("Std Before", impact["std_before"])
                    with si2:
                        st.metric("Mean After", impact["mean_after"])
                        st.metric("Std After", impact["std_after"])

                st.markdown("**Distribution Comparison**")
                if is_numeric:
                    st.plotly_chart(
                        plot_before_after_histogram(df[sel_col], imputed_df[sel_col], sel_col),
                        use_container_width=True,
                    )
                else:
                    st.plotly_chart(
                        plot_before_after_bar(df[sel_col], imputed_df[sel_col], sel_col),
                        use_container_width=True,
                    )

                csv_bytes = imputed_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Imputed CSV",
                    data=csv_bytes,
                    file_name=f"imputed_{sel_col}.csv",
                    mime="text/csv",
                    type="primary",
                )

        else:  # Bulk mode
            st.markdown("**Configure strategy per column:**")

            bulk_strategies = {}
            bulk_custom_vals = {}

            for col in cols_with_missing:
                n_miss = int(df[col].isna().sum())
                sem_type = infer_semantic_type(df[col])
                suggested = suggest_strategy(df[col], sem_type)
                is_numeric = pd.api.types.is_numeric_dtype(df[col])

                available = IMPUTATION_STRATEGIES.copy()
                if not is_numeric:
                    available = [s for s in available if s not in ("Mean", "Median", "Interpolate (Linear)")]

                with st.expander(f"**{col}** — {n_miss:,} missing ({round(n_miss / len(df) * 100, 1)}%) · {sem_type}"):
                    default_idx = available.index(suggested) if suggested in available else 0
                    strat = st.selectbox(
                        "Strategy",
                        available,
                        index=default_idx,
                        key=f"bulk_{col}",
                    )
                    bulk_strategies[col] = strat

                    if strat == "Custom Value":
                        cv = st.text_input("Fill value", key=f"bulk_cv_{col}")
                        if is_numeric and cv:
                            try:
                                cv = float(cv)
                            except ValueError:
                                st.warning("Numeric column — enter a number.")
                                cv = None
                        bulk_custom_vals[col] = cv
                    else:
                        bulk_custom_vals[col] = None

            if st.button("Apply All Imputations", type="primary", key="bulk_apply"):
                col_strats = {
                    col: (bulk_strategies[col], bulk_custom_vals[col])
                    for col in cols_with_missing
                }
                imputed_df = impute_multiple(df, col_strats)
                st.session_state["bulk_imputed_df"] = imputed_df
                st.session_state["bulk_strategies"] = bulk_strategies

            if "bulk_imputed_df" in st.session_state:
                imputed_df = st.session_state["bulk_imputed_df"]
                bulk_strats = st.session_state["bulk_strategies"]

                st.markdown("---")
                st.markdown("**Imputation Results**")

                missing_before = {col: int(df[col].isna().sum()) for col in cols_with_missing}
                missing_after = {col: int(imputed_df[col].isna().sum()) for col in cols_with_missing}

                st.plotly_chart(
                    plot_imputation_summary(missing_before, missing_after),
                    use_container_width=True,
                )

                results_data = []
                for col in cols_with_missing:
                    results_data.append({
                        "Column": col,
                        "Strategy": bulk_strats.get(col, "—"),
                        "Missing Before": missing_before[col],
                        "Missing After": missing_after.get(col, 0),
                        "Resolved": missing_before[col] - missing_after.get(col, 0),
                    })
                st.dataframe(
                    pd.DataFrame(results_data),
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Compare distributions per column"):
                    for col in cols_with_missing:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            st.plotly_chart(
                                plot_before_after_histogram(df[col], imputed_df[col], col),
                                use_container_width=True,
                            )
                        else:
                            st.plotly_chart(
                                plot_before_after_bar(df[col], imputed_df[col], col),
                                use_container_width=True,
                            )

                csv_bytes = imputed_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Fully Imputed CSV",
                    data=csv_bytes,
                    file_name="imputed_full.csv",
                    mime="text/csv",
                    type="primary",
                    key="bulk_download",
                )

# ── Tab 7: Column Deep-Dive ────────────────────────────────────────────────

with tabs[7]:
    st.markdown('<div class="section-header"><h3>Column Deep-Dive</h3></div>', unsafe_allow_html=True)
    all_cols = df.columns.tolist()
    picked = st.selectbox("Choose a column", all_cols, key="deep_dive")
    profile = next((p for p in report.profiles if p.name == picked), None)

    if profile:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Semantic Type", profile.semantic_type)
        with d2: st.metric("Missing", f"{profile.missing} ({profile.missing_pct}%)")
        with d3: st.metric("Unique Values", profile.unique)
        with d4: st.metric("Pandas dtype", profile.dtype)

        if profile.stats:
            st.markdown("**Full Statistics**")
            scols = st.columns(5)
            for i, (k, v) in enumerate(profile.stats.items()):
                if k == "top_5":
                    continue
                with scols[i % 5]:
                    st.metric(k, v if v is not None else "—")

            if "top_5" in profile.stats:
                st.markdown("**Top Values**")
                top5_df = pd.DataFrame(
                    list(profile.stats["top_5"].items()),
                    columns=["Value", "Count"],
                )
                st.dataframe(top5_df, use_container_width=True, hide_index=True)

        if profile.semantic_type in ("integer", "float"):
            c_l, c_r = st.columns(2)
            with c_l:
                st.plotly_chart(plot_histogram(df[picked]), use_container_width=True)
            with c_r:
                st.plotly_chart(plot_box_strip(df[picked]), use_container_width=True)
        elif profile.semantic_type in ("categorical", "boolean"):
            st.plotly_chart(plot_categorical_bar(df[picked]), use_container_width=True)
