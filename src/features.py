# src/features.py
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def create_rfm_features_from_transactions(
    transactions: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    amount_col: str,
    snapshot_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) from transactional data.

    transactions: one row per transaction/order.
    snapshot_date: reference date; if None, use max(invoice_date) + 1 day.
    """
    df = transactions.copy()
    df[invoice_date_col] = pd.to_datetime(df[invoice_date_col])

    if snapshot_date is None:
        snapshot_date = df[invoice_date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id_col).agg(
        recency_days=(invoice_date_col, lambda x: (snapshot_date - x.max()).days),
        frequency=(invoice_date_col, "nunique"),
        monetary=(amount_col, "sum"),
    )

    rfm = rfm.reset_index()
    return rfm


def add_simple_clv_proxy(
    df_customers: pd.DataFrame,
    monetary_col: str,
    frequency_col: str,
    clv_col_name: str = "clv_proxy",
) -> pd.DataFrame:
    """
    Add a simple CLV proxy: monetary * frequency (you can refine later).
    """
    df = df_customers.copy()
    df[clv_col_name] = df[monetary_col] * df[frequency_col]
    return df


def add_basic_behavioral_features(
    df: pd.DataFrame,
    total_orders_col: Optional[str] = None,
    total_spend_col: Optional[str] = None,
    last_order_date_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add generic behavioural features if the columns exist.

    This is intentionally simple/generic; adapt in notebooks as needed.
    """
    df = df.copy()

    if total_orders_col and total_orders_col in df.columns:
        df["avg_orders_per_month"] = df[total_orders_col] / np.maximum(
            df.get("tenure_months", 1), 1
        )

    if (
        total_orders_col
        and total_spend_col
        and total_orders_col in df.columns
        and total_spend_col in df.columns
    ):
        df["avg_order_value"] = df[total_spend_col] / df[total_orders_col].replace(0, np.nan)
        df["avg_order_value"].fillna(0, inplace=True)

    if last_order_date_col and last_order_date_col in df.columns:
        df[last_order_date_col] = pd.to_datetime(df[last_order_date_col])
        snapshot = df[last_order_date_col].max() + pd.Timedelta(days=1)
        df["recency_days"] = (snapshot - df[last_order_date_col]).dt.days

    return df
