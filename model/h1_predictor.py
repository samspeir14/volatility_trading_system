"""Routing predictor for the h=1 deviation forecast.

Routes between the pooled LightGBM, the HAR-RV benchmark, and their 50/50
blend based on the retrain job's acceptance gate (whichever wins OOS
within-ticker deviation R²; the decision is stored as `route` in
latest_retrain_r2.json).
HAR is a tiny artifact and always loadable, so it doubles as the fallback
when the LightGBM artifact is missing or fails to produce a finite value.
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from model.har_model import HARRVPredictor
from model.lightgbm_model import LightGBMVolPredictor

logger = logging.getLogger(__name__)

VALID_ROUTES = ("lgbm", "har", "blend")


class H1DeviationPredictor:
    def __init__(
        self,
        lgbm: LightGBMVolPredictor | None,
        har: HARRVPredictor | None,
        route: str = "har",
    ):
        if route not in VALID_ROUTES:
            raise ValueError(f"route must be one of {VALID_ROUTES}, got {route!r}")
        if lgbm is None and har is None:
            raise ValueError("need at least one of lgbm / har")
        if route == "blend" and (lgbm is None or har is None):
            route = "har" if lgbm is None else "lgbm"
            logger.warning(
                "route=blend needs both artifacts — falling back to %s", route,
            )
        if route == "lgbm" and lgbm is None:
            logger.warning(
                "route=lgbm requested but no LightGBM artifact loaded — "
                "falling back to HAR"
            )
            route = "har"
        if route == "har" and har is None:
            logger.warning(
                "route=har requested but no HAR artifact loaded — using LightGBM"
            )
            route = "lgbm"
        self._lgbm = lgbm
        self._har = har
        self._route = route

    @property
    def active_model(self) -> str:
        return self._route

    def predict_deviation(self, X_row: pd.DataFrame) -> float:
        """Predict tomorrow's log-vol deviation from a single feature row.
        route=blend averages the two models (degrading to whichever side is
        finite); lgbm/har fall back to the other model if the routed one
        yields a non-finite value (and the other exists)."""
        if self._route == "blend":
            values = [
                float(m.predict(X_row)[0]) for m in (self._lgbm, self._har)
            ]
            finite = [v for v in values if math.isfinite(v)]
            if len(finite) < 2:
                logger.warning(
                    "blend: %d of 2 models non-finite", 2 - len(finite),
                )
            return sum(finite) / len(finite) if finite else float("nan")
        primary = self._lgbm if self._route == "lgbm" else self._har
        fallback = self._har if self._route == "lgbm" else self._lgbm
        value = float(primary.predict(X_row)[0])
        if math.isfinite(value) or fallback is None:
            return value
        logger.warning(
            "%s produced non-finite deviation; falling back", self._route,
        )
        return float(fallback.predict(X_row)[0])
