from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib


# -------------------------
# Paths & configuration
# -------------------------

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Customer Segmentation & Churn",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Data & model loading
# -------------------------

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """
    Try to load processed data from a few common filenames.
    Supports both parquet and csv.
    """
    candidates = [
        "churn_with_segments.parquet",
        "churn_with_segments.csv",
        "churn_ready.parquet",
        "churn_ready.csv",
    ]

    for fname in candidates:
        path = DATA_PROCESSED / fname
        if path.exists():
            st.sidebar.success(f"Loaded data: {fname}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_csv(path)

    st.error(
        "No processed data found in data/processed/. "
        "Expected one of: churn_with_segments.(parquet|csv), churn_ready.(parquet|csv)."
    )
    return None


@st.cache_resource
def load_model():
    """
    Try to load a trained churn model pipeline.
    Expects a full sklearn Pipeline saved with joblib.
    """
    candidates = [
        "best_churn_pipeline.joblib",
        "best_churn_model.pkl",
        "best_model.joblib",
    ]

    for fname in candidates:
        path = MODELS_DIR / fname
        if path.exists():
            st.sidebar.success(f"Loaded model: {fname}")
            return joblib.load(path)

    st.sidebar.warning(
        "No trained model found in models/. "
        "Train in your modelling notebook and save with joblib.dump(pipeline, 'models/best_churn_pipeline.joblib')."
    )
    return None


# -------------------------
# Helpers
# -------------------------

NUMERIC_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "clv_proxy",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
    # "segment",  # you can add segment as categorical if you used it in training
]


def compute_kpis(df: pd.DataFrame, prob_col: Optional[str] = None):
    total_customers = len(df)
    churn_rate = df["Churn"].mean() if "Churn" in df.columns else np.nan
    avg_clv = df["clv_proxy"].mean() if "clv_proxy" in df.columns else np.nan
    avg_churn_prob = df[prob_col].mean() if prob_col and prob_col in df.columns else np.nan
    return total_customers, churn_rate, avg_clv, avg_churn_prob


def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Add churn probability and prediction columns to df if model is available.
    Only uses rows with all required features non-null.
    """
    df = df.copy()
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]

    # we don't want to fail if some features are missing
    if len(feature_cols) == 0:
        st.warning("No overlap between expected feature columns and data columns.")
        return df

    df_model = df.dropna(subset=feature_cols).copy()
    if df_model.empty:
        st.warning("No rows with complete features for prediction.")
        return df

    proba = model.predict_proba(df_model[feature_cols])[:, 1]
    df_model["churn_proba"] = proba
    df_model["churn_pred"] = (df_model["churn_proba"] >= 0.5).astype(int)

    # merge back into full df
    df = df.merge(
        df_model[["CustomerID", "churn_proba", "churn_pred"]],
        on="CustomerID",
        how="left",
    )
    return df


# -------------------------
# Main app
# -------------------------

def main():
    st.title("📊 Customer Segmentation & Churn Dashboard")

    st.markdown(
        """
        This dashboard summarizes customer segments and churn risk for an e-commerce retailer.
        Use the tabs below to explore:
        - **Overview:** high-level KPIs and churn distribution  
        - **Segments:** RFM-style segments and their churn/CLV profiles  
        - **Churn Explorer:** filter and inspect high-risk customers  
        """
    )

    df = load_data()
    if df is None:
        st.stop()

    model = load_model()

    if model is not None:
        df = add_predictions(df, model)
        prob_col = "churn_proba"
    else:
        prob_col = None

    # Basic sanity: ensure CustomerID exists
    if "CustomerID" not in df.columns:
        st.error("Expected a 'CustomerID' column in the data.")
        st.stop()

    # KPIs
    total_customers, churn_rate, avg_clv, avg_churn_prob = compute_kpis(df, prob_col)

    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total customers", f"{total_customers:,}")
    if not np.isnan(churn_rate):
        col2.metric("Churn rate", f"{churn_rate * 100:.1f}%")
    else:
        col2.metric("Churn rate", "N/A")

    if not np.isnan(avg_clv):
        col3.metric("Avg CLV proxy", f"{avg_clv:,.0f}")
    else:
        col3.metric("Avg CLV proxy", "N/A")

    if prob_col and not np.isnan(avg_churn_prob):
        col4.metric("Avg predicted churn prob", f"{avg_churn_prob * 100:.1f}%")
    else:
        col4.metric("Avg predicted churn prob", "N/A")

    # Tabs
    tab_overview, tab_segments, tab_churn = st.tabs(
        ["Overview", "Segments", "Churn Explorer"]
    )

    # -------------------------
    # Overview tab
    # -------------------------
    with tab_overview:
        st.markdown("### Churn distribution")

        if "Churn" in df.columns:
            fig = px.histogram(
                df,
                x="Churn",
                nbins=2,
                text_auto=True,
                labels={"Churn": "Churn (0 = no, 1 = yes)"},
                title="Churn class counts",
            )
            st.plotly_chart(fig, use_container_width=True)

        if prob_col and prob_col in df.columns:
            st.markdown("### Predicted churn probability distribution")
            fig2 = px.histogram(
                df,
                x=prob_col,
                nbins=30,
                labels={prob_col: "Predicted churn probability"},
                title="Distribution of predicted churn probabilities",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Feature overview (numeric)")
        numeric_cols_available = [c for c in NUMERIC_FEATURES if c in df.columns]
        st.dataframe(df[numeric_cols_available].describe().T)

    # -------------------------
    # Segments tab
    # -------------------------
    with tab_segments:
        st.markdown("### Segment performance")

        if "segment" not in df.columns:
            st.info("No 'segment' column found. Run your segmentation notebook to add segments.")
        else:
            seg_group = df.groupby("segment").agg(
                customers=("CustomerID", "nunique"),
                churn_rate=("Churn", "mean") if "Churn" in df.columns else ("CustomerID", "size"),
                avg_clv=("clv_proxy", "mean") if "clv_proxy" in df.columns else ("CustomerID", "size"),
                avg_churn_proba=(prob_col, "mean") if prob_col and prob_col in df.columns else ("CustomerID", "size"),
            )

            seg_group = seg_group.reset_index()

            st.dataframe(seg_group.style.format({
                "churn_rate": "{:.2%}",
                "avg_clv": "{:,.0f}",
                "avg_churn_proba": "{:.2%}",
            }))

            colA, colB = st.columns(2)

            with colA:
                if "avg_clv" in seg_group.columns:
                    fig_seg_clv = px.bar(
                        seg_group,
                        x="segment",
                        y="avg_clv",
                        title="Average CLV proxy by segment",
                        labels={"avg_clv": "Avg CLV proxy"},
                    )
                    st.plotly_chart(fig_seg_clv, use_container_width=True)

            with colB:
                if "churn_rate" in seg_group.columns:
                    fig_seg_churn = px.bar(
                        seg_group,
                        x="segment",
                        y="churn_rate",
                        title="Churn rate by segment",
                        labels={"churn_rate": "Churn rate"},
                    )
                    st.plotly_chart(fig_seg_churn, use_container_width=True)

            # Optional: per-segment distribution of predicted probs
            if prob_col and prob_col in df.columns:
                st.markdown("### Predicted churn probability by segment")
                fig_box = px.box(
                    df.dropna(subset=[prob_col]),
                    x="segment",
                    y=prob_col,
                    labels={prob_col: "Predicted churn probability"},
                )
                st.plotly_chart(fig_box, use_container_width=True)

    # -------------------------
    # Churn Explorer tab
    # -------------------------
    with tab_churn:
        st.markdown("### Filter and inspect high-risk customers")

        if prob_col is None or prob_col not in df.columns:
            st.info("No model predictions available yet. Train and save a model to use this tab.")
        else:
            # Sidebar-like controls within this tab
            left, right = st.columns([1, 3])

            with left:
                threshold = st.slider(
                    "Churn probability threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                )

                segments_available: List[int] | List[str]
                if "segment" in df.columns:
                    segments_available = sorted(df["segment"].dropna().unique())
                    selected_segments = st.multiselect(
                        "Filter by segment",
                        options=segments_available,
                        default=segments_available,
                    )
                else:
                    selected_segments = None

                regions_available = (
                    sorted(df["region"].dropna().unique())
                    if "region" in df.columns
                    else []
                )
                if regions_available:
                    selected_regions = st.multiselect(
                        "Filter by region",
                        options=regions_available,
                        default=regions_available,
                    )
                else:
                    selected_regions = None

                devices_available = (
                    sorted(df["PreferredLoginDevice"].dropna().unique())
                    if "PreferredLoginDevice" in df.columns
                    else []
                )
                if devices_available:
                    selected_devices = st.multiselect(
                        "Filter by device",
                        options=devices_available,
                        default=devices_available,
                    )
                else:
                    selected_devices = None

                top_n = st.number_input(
                    "Show top N high-risk customers",
                    min_value=5,
                    max_value=500,
                    value=50,
                    step=5,
                )

            with right:
                filtered = df.copy()
                filtered = filtered[filtered[prob_col] >= threshold]

                if selected_segments is not None:
                    filtered = filtered[filtered["segment"].isin(selected_segments)]

                if selected_regions is not None and len(selected_regions) > 0 and "region" in filtered.columns:
                    filtered = filtered[filtered["region"].isin(selected_regions)]

                if selected_devices is not None and len(selected_devices) > 0 and "PreferredLoginDevice" in filtered.columns:
                    filtered = filtered[filtered["PreferredLoginDevice"].isin(selected_devices)]

                if filtered.empty:
                    st.warning("No customers match the current filters.")
                else:
                    filtered = filtered.sort_values(prob_col, ascending=False).head(top_n)

                    display_cols = ["CustomerID", prob_col, "Churn"] if "Churn" in filtered.columns else ["CustomerID", prob_col]
                    for extra in ["Tenure", "OrderCount", "CashbackAmount", "clv_proxy", "segment"]:
                        if extra in filtered.columns:
                            display_cols.append(extra)

                    st.dataframe(
                        filtered[display_cols].style.format(
                            {
                                prob_col: "{:.1%}",
                                "clv_proxy": "{:,.0f}",
                            }
                        )
                    )

                    # Optional: select a single customer for details
                    st.markdown("#### Customer details")
                    selected_customer = st.selectbox(
                        "Select customer ID",
                        options=filtered["CustomerID"].tolist(),
                    )

                    cust_row = df[df["CustomerID"] == selected_customer].iloc[0]
                    st.write("Basic info:")
                    st.json(
                        {
                            "CustomerID": int(cust_row["CustomerID"]),
                            "Churn": int(cust_row["Churn"]) if "Churn" in cust_row else None,
                            "Predicted churn probability": float(cust_row.get(prob_col, np.nan)),
                            "Tenure": float(cust_row.get("Tenure", np.nan)),
                            "OrderCount": float(cust_row.get("OrderCount", np.nan)),
                            "CashbackAmount": float(cust_row.get("CashbackAmount", np.nan)),
                            "CLV proxy": float(cust_row.get("clv_proxy", np.nan)),
                            "Segment": int(cust_row.get("segment")) if "segment" in cust_row.index and not pd.isna(cust_row["segment"]) else None,
                            "PreferredLoginDevice": cust_row.get("PreferredLoginDevice"),
                            "PreferredPaymentMode": cust_row.get("PreferredPaymentMode"),
                            "Gender": cust_row.get("Gender"),
                            "PreferedOrderCat": cust_row.get("PreferedOrderCat"),
                        }
                    )



if __name__ == "__main__":
    main()
