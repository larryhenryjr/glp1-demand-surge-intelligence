"""Focused tests for the GLP-1 demand-surge analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import choose_threshold, market_era, product_age


def test_market_era_boundaries() -> None:
    """Market-era labels should change at the documented launch boundaries."""
    assert market_era(pd.Series({"year": 2021, "quarter": 2})) == "Pre-Wegovy"
    assert market_era(pd.Series({"year": 2021, "quarter": 3})) == "Wegovy expansion"
    assert market_era(pd.Series({"year": 2023, "quarter": 4})) == "Zepbound expansion"


def test_product_age_never_negative() -> None:
    """Product age should be expressed as nonnegative elapsed quarters."""
    row = pd.Series({"brand": "OZEMPIC", "year": 2017, "quarter": 4})
    assert product_age(row) == 0


def test_choose_threshold_returns_candidate_table() -> None:
    """Threshold selection should return a valid cutoff and audit table."""
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.35, 0.55, 0.70, 0.95])
    threshold, table = choose_threshold(y_true, probabilities)
    assert 0.10 <= threshold <= 0.90
    assert {"threshold", "accuracy", "recall", "specificity", "f1"} <= set(table.columns)
