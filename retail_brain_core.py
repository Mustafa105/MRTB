"""Retail Brain core engine (no Streamlit dependency, fully unit-testable).

Contents
--------
1. Data loading and validation for the retail operations CSV schema.
2. DemandModel: a time-aware demand forecaster trained on history only.
     demand = context_rate(x) * traffic * commercial_lever_adjustment
   - context_rate: XGBoost model of demand per visitor (weather, events,
     calendar, staffing, private-label presence, category). Explained with SHAP.
   - commercial levers (price position, promotion): estimated from the data
     when the data can identify them, otherwise an explicit, editable
     assumption. The model never silently pretends to have learned an effect
     the data cannot support.
   - split-conformal prediction intervals (80% by default).
   - fill-rate curve (delivery delay -> share of demand served), monotone,
     learned from the data with pool-adjacent-violators.
3. Backtest/validation against naive baselines on a held-out time period.
4. Nine decision-support agents that communicate over an in-process event bus.
"""
from __future__ import annotations

import hashlib
import io
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Date", "Category", "Event_Season", "Avg_Temperature_C", "Climate_Weather_Condition",
    "Our_Active_Promotion", "Esselunga_Price_EUR", "Lidl_Price_EUR", "Conad_Price_EUR",
    "Carrefour_Price_EUR", "Private_Label_Price_EUR", "Private_Label_Share_Pct",
    "Partnership_Deal_Active", "Shelf_Share_Private_Label_Pct", "Store_Traffic_Volume",
    "Staff_Capacity_Pct", "Shelf_Sorting_Compliance_Pct", "Logistics_Delivery_Delay_Hrs",
    "Checkout_Queue_Avg_Length", "Cold_Chain_Failure_Flag", "Demand_QTY", "Actual_Sales_QTY",
]
OPTIONAL_COLUMNS = ["AI_Gross_Margin_Pct"]
NUMERIC_COLUMNS = [
    "Avg_Temperature_C", "Esselunga_Price_EUR", "Lidl_Price_EUR", "Conad_Price_EUR",
    "Carrefour_Price_EUR", "Private_Label_Price_EUR", "Private_Label_Share_Pct",
    "Shelf_Share_Private_Label_Pct", "Store_Traffic_Volume", "Staff_Capacity_Pct",
    "Shelf_Sorting_Compliance_Pct", "Logistics_Delivery_Delay_Hrs", "Checkout_Queue_Avg_Length",
    "Demand_QTY", "Actual_Sales_QTY",
]

# Business assumptions the dataset cannot provide. They are surfaced in the UI
# as assumptions, never presented as learned facts.
CATEGORY_ASSUMPTIONS = {
    "Birra": {"shelf_life_days": 120, "base_waste_rate": 0.010},
    "Prodotti da Forno": {"shelf_life_days": 2, "base_waste_rate": 0.090},
    "Ortofrutta": {"shelf_life_days": 4, "base_waste_rate": 0.120},
    "Detersivi": {"shelf_life_days": 365, "base_waste_rate": 0.005},
}
DEFAULT_ASSUMPTION = {"shelf_life_days": 30, "base_waste_rate": 0.030}
DEFAULT_MARGIN = 0.30                 # used when the CSV has no margin column
COLD_CHAIN_WRITEOFF_SHARE = 0.50      # share of a day's demand written off on a cold-chain failure
INTERVAL_COVERAGE = 0.80
DEFAULT_ASSUMED_ELASTICITY = -0.8     # elasticity of demand w.r.t. price relative to competitors
DEFAULT_ASSUMED_PROMO_UPLIFT = 0.08   # +8% demand when a campaign is active


def _read_any(source) -> pd.DataFrame:
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        return pd.read_csv(source, keep_default_na=False, na_values=[""])
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
        return pd.read_csv(io.StringIO(raw), keep_default_na=False, na_values=[""])
    raise ValueError("Unsupported data source")


def load_operations_csv(source) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate a retail operations CSV. Returns (dataframe, warnings).

    Raises ValueError with a readable message if required columns are missing
    or too few usable rows remain.
    """
    df = _read_any(source)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    warnings: list[str] = []
    n0 = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Category", "Event_Season", "Climate_Weather_Condition", "Our_Active_Promotion",
                "Partnership_Deal_Active", "Cold_Chain_Failure_Flag"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=["Date"] + NUMERIC_COLUMNS)
    df = df[(df["Store_Traffic_Volume"] > 0) & (df["Demand_QTY"] > 0)]
    dup = df.duplicated(subset=["Date", "Category"], keep="last").sum()
    if dup:
        warnings.append(f"{dup} duplicate (Date, Category) row(s) found; kept the last of each.")
        df = df.drop_duplicates(subset=["Date", "Category"], keep="last")
    dropped = n0 - len(df) - dup
    if dropped > 0:
        warnings.append(f"{dropped} row(s) dropped for missing or invalid values.")
    df = df.sort_values(["Date", "Category"]).reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"Only {len(df)} usable rows; at least 200 are needed to train a model.")
    if df["Date"].nunique() < 60:
        raise ValueError("Fewer than 60 distinct dates; at least 60 days of history are needed.")
    return df, warnings


def data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["comp_price"] = d[["Lidl_Price_EUR", "Conad_Price_EUR", "Carrefour_Price_EUR"]].mean(axis=1)
    d["rel_price"] = d["Esselunga_Price_EUR"] / d["comp_price"]
    d["promo_active"] = (d["Our_Active_Promotion"].str.lower() != "none").astype(int)
    d["partner_active"] = (d["Partnership_Deal_Active"].str.lower() == "yes").astype(int)
    d["cold_chain"] = (d["Cold_Chain_Failure_Flag"].str.lower() == "yes").astype(int)
    d["dow"] = d["Date"].dt.dayofweek
    d["month"] = d["Date"].dt.month
    d["rate"] = d["Demand_QTY"] / d["Store_Traffic_Volume"]
    d["fill"] = (d["Actual_Sales_QTY"] / d["Demand_QTY"]).clip(0, 1)
    return d


# Profile key -> human label (used by the UI and by explanations)
PROFILE_LABELS = {
    "price": "Own price (EUR)", "competitor_price": "Competitor price, mean (EUR)",
    "promo_active": "Promotion active (0/1)", "partner_active": "Partnership deal (0/1)",
    "temperature": "Temperature (C)", "traffic": "Store traffic (visitors/day)",
    "staff_capacity_pct": "Staff capacity (%)", "delay_hrs": "Delivery delay (hours)",
    "compliance_pct": "Shelf compliance (%)", "queue_len": "Checkout queue (avg length)",
    "cold_chain": "Cold-chain failure (0/1)", "pl_share": "Private-label share (%)",
    "shelf_share_pl": "Private-label shelf share (%)", "unit_cost": "Unit cost (EUR)",
}


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _pav_nonincreasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators for a non-increasing fit of y on x."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    # isotonic increasing on -y
    blocks: list[list[float]] = []  # [sum, weight, count]
    for v in -ys:
        blocks.append([v, 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, w, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w
            blocks[-1][2] += c
    fit = np.concatenate([[b[0] / b[1]] * int(b[2]) for b in blocks])
    return xs, -fit


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def estimate_levers(d: pd.DataFrame) -> dict[str, Any]:
    """Estimate price elasticity and promotion uplift from history.

    log(rate) ~ log(relative price) + promo + category/event dummies.
    Reports the estimate, standard error and whether it is statistically
    distinguishable from zero with the expected sign.
    """
    y = np.log(d["rate"].clip(lower=1e-6).to_numpy())
    cols = [np.ones(len(d)), np.log(d["rel_price"].to_numpy()), d["promo_active"].to_numpy().astype(float)]
    names = ["const", "log_rel_price", "promo"]
    for prefix, col in (("cat", "Category"), ("ev", "Event_Season")):
        dummies = pd.get_dummies(d[col], prefix=prefix, drop_first=True).astype(float)
        for c in dummies.columns:
            cols.append(dummies[c].to_numpy())
            names.append(c)
    X = np.column_stack(cols)
    beta, se = _ols(X, y)
    b_p, se_p = float(beta[1]), float(se[1])
    b_u, se_u = float(beta[2]), float(se[2])
    t_p = b_p / se_p if se_p > 0 else 0.0
    t_u = b_u / se_u if se_u > 0 else 0.0
    return {
        "elasticity_est": b_p, "elasticity_se": se_p, "elasticity_t": t_p,
        "elasticity_detectable": bool(t_p <= -2.0),
        "promo_uplift_est": math.expm1(b_u), "promo_uplift_se": se_u, "promo_t": t_u,
        "promo_detectable": bool(t_u >= 2.0),
        "rel_price_std": float(d["rel_price"].std()),
        "promo_days": int(d["promo_active"].sum()),
        "n": int(len(d)),
    }


def resolve_levers(est: dict[str, Any], assumed_elasticity: float, assumed_uplift: float) -> dict[str, Any]:
    """Use the data estimate when detectable, otherwise the explicit assumption."""
    out = {"elasticity": float(assumed_elasticity), "elasticity_source": "assumed",
           "promo_uplift": float(assumed_uplift), "promo_source": "assumed"}
    if est.get("elasticity_detectable"):
        out["elasticity"], out["elasticity_source"] = float(est["elasticity_est"]), "estimated from data"
    if est.get("promo_detectable"):
        out["promo_uplift"], out["promo_source"] = float(est["promo_uplift_est"]), "estimated from data"
    return out


# ---------------------------------------------------------------------------
# 2. Demand model
# ---------------------------------------------------------------------------
NUMERIC_CONTEXT = {
    "Avg_Temperature_C": "temperature", "Private_Label_Share_Pct": "pl_share",
    "Shelf_Share_Private_Label_Pct": "shelf_share_pl", "Staff_Capacity_Pct": "staff_capacity_pct",
    "partner_active": "partner_active", "dow": "dow", "month": "month",
}
NUMERIC_LABELS = {
    "Avg_Temperature_C": "Temperature", "Private_Label_Share_Pct": "Private-label share",
    "Shelf_Share_Private_Label_Pct": "Private-label shelf share", "Staff_Capacity_Pct": "Staff capacity",
    "partner_active": "Partnership deal", "dow": "Day of week", "month": "Month of year",
}
GROUP_PREFIXES = {"cat_": "Category", "ev_": "Event / season", "wx_": "Weather condition"}


def _feature_group(col: str) -> str:
    for p, g in GROUP_PREFIXES.items():
        if col.startswith(p):
            return g
    return NUMERIC_LABELS.get(col, col)


@dataclass
class DemandModel:
    asof: pd.Timestamp
    kind: str
    columns: list[str]
    categories: list[str]
    events: list[str]
    weathers: list[str]
    base_rate: float
    mean_traffic: float
    median_traffic: dict[str, float]
    ref_rel_price: dict[str, float]
    ref_promo: dict[str, float]
    ref_price: dict[str, float]
    margin: dict[str, float]
    cheapest_ratio: dict[str, float]
    lidl_ratio: dict[str, float]
    interval_q: float
    fill_x: np.ndarray
    fill_y: np.ndarray
    lever_est: dict[str, Any]
    validation: dict[str, Any]
    n_train: int
    history_start: pd.Timestamp
    history_end: pd.Timestamp
    cat_rate: dict[str, float]
    _model: Any = None
    _explainer: Any = None

    # ----- feature construction ------------------------------------------
    def context_frame(self, profiles: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for p in profiles:
            r = {c: 0.0 for c in self.columns}
            for col, key in NUMERIC_CONTEXT.items():
                r[col] = float(p[key])
            for prefix, key in (("cat_", "category"), ("ev_", "event"), ("wx_", "weather")):
                name = f"{prefix}{p[key]}"
                if name in r:
                    r[name] = 1.0
            rows.append(r)
        return pd.DataFrame(rows)[self.columns]

    def _rate(self, frame: pd.DataFrame, profiles: list[dict[str, Any]]) -> np.ndarray:
        if self.kind == "xgboost" and self._model is not None:
            return np.maximum(self._model.predict(frame), 0.01)
        return np.array([self.cat_rate.get(p["category"], self.base_rate) for p in profiles])

    # ----- prediction -------------------------------------------------------
    def lever_adjustment(self, p: dict[str, Any], levers: dict[str, Any]) -> tuple[float, float]:
        cat = p["category"]
        rel = p["price"] / max(p["competitor_price"], 0.01)
        ref = self.ref_rel_price.get(cat, 1.0)
        price_adj = float((rel / ref) ** levers["elasticity"])
        promo_ref = self.ref_promo.get(cat, 0.0)
        promo_adj = float((1 + levers["promo_uplift"] * p["promo_active"]) /
                          (1 + levers["promo_uplift"] * promo_ref))
        return price_adj, promo_adj

    def predict(self, p: dict[str, Any], levers: dict[str, Any]) -> dict[str, float]:
        frame = self.context_frame([p])
        rate = float(self._rate(frame, [p])[0])
        ctx = rate * float(p["traffic"])
        price_adj, promo_adj = self.lever_adjustment(p, levers)
        point = ctx * price_adj * promo_adj
        lo = max(0.0, ctx * (1 - self.interval_q)) * price_adj * promo_adj
        hi = ctx * (1 + self.interval_q) * price_adj * promo_adj
        return {"demand": point, "low": lo, "high": hi, "rate": rate, "ctx_demand": ctx,
                "price_adj": price_adj, "promo_adj": promo_adj}

    def fill_rate(self, delay_hrs: float) -> float:
        return float(np.clip(np.interp(delay_hrs, self.fill_x, self.fill_y), 0.0, 1.0))

    # ----- explanation --------------------------------------------------------
    def explain(self, p: dict[str, Any], levers: dict[str, Any]) -> pd.DataFrame:
        """Additive 'demand bridge' from a baseline to the forecast, in units/day.

        base + traffic effect + context drivers (SHAP) + price lever + promo lever
        sums exactly to the point forecast.
        """
        frame = self.context_frame([p])
        traffic = float(p["traffic"])
        pred = self.predict(p, levers)
        base_rate = self.base_rate
        rows: list[dict[str, Any]] = []
        shap_ok = False
        contrib = {}
        if self.kind == "xgboost" and self._explainer is not None:
            try:
                sv = np.ravel(self._explainer.shap_values(frame))
                ev = self._explainer.expected_value
                base_rate = float(np.ravel(ev)[0]) if hasattr(ev, "__len__") else float(ev)
                for col, val in zip(self.columns, sv):
                    g = _feature_group(col)
                    contrib[g] = contrib.get(g, 0.0) + float(val) * traffic
                shap_ok = True
            except Exception:
                shap_ok = False
        if not shap_ok:
            base_rate = self.cat_rate.get(p["category"], self.base_rate)
        rows.append({"Driver": "Baseline: typical traffic, average conditions",
                     "Contribution (units/day)": base_rate * self.mean_traffic, "Type": "base"})
        rows.append({"Driver": "Store traffic today vs typical",
                     "Contribution (units/day)": base_rate * (traffic - self.mean_traffic), "Type": "traffic"})
        if shap_ok:
            for g, v in contrib.items():
                rows.append({"Driver": g, "Contribution (units/day)": v, "Type": "context"})
        else:
            resid = pred["ctx_demand"] - base_rate * traffic
            rows.append({"Driver": "Category level", "Contribution (units/day)": resid, "Type": "context"})
        ctx = pred["ctx_demand"]
        rows.append({"Driver": "Price position vs competitors (lever)",
                     "Contribution (units/day)": ctx * (pred["price_adj"] - 1), "Type": "lever"})
        rows.append({"Driver": "Promotion (lever)",
                     "Contribution (units/day)": ctx * pred["price_adj"] * (pred["promo_adj"] - 1), "Type": "lever"})
        out = pd.DataFrame(rows)
        out["Absolute"] = out["Contribution (units/day)"].abs()
        out.attrs["forecast"] = float(pred["demand"])
        out.attrs["sum_of_parts"] = float(out["Contribution (units/day)"].sum())
        out.attrs["shap"] = shap_ok
        return out


def _conformal_q(resid_rel: np.ndarray, coverage: float) -> float:
    n = len(resid_rel)
    if n == 0:
        return 0.25
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(resid_rel, level))


XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "min_child_weight": w}
    for d in (2, 3) for n in (60, 120, 200) for w in (10, 40)
]


def _make_xgb(seed: int = 17, params: dict[str, Any] | None = None):
    p = {"n_estimators": 200, "max_depth": 3, "min_child_weight": 10}
    p.update(params or {})
    return XGBRegressor(learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
                        objective="reg:squarederror", random_state=seed, n_jobs=2, **p)


def _design(d: pd.DataFrame, categories, events, weathers) -> pd.DataFrame:
    cols = {}
    for col in NUMERIC_CONTEXT:
        cols[col] = d[col].astype(float).to_numpy()
    for c in categories:
        cols[f"cat_{c}"] = (d["Category"] == c).astype(float).to_numpy()
    for e in events:
        cols[f"ev_{e}"] = (d["Event_Season"] == e).astype(float).to_numpy()
    for w in weathers:
        cols[f"wx_{w}"] = (d["Climate_Weather_Condition"] == w).astype(float).to_numpy()
    return pd.DataFrame(cols)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "MAPE %": float(np.mean(np.abs(err) / y) * 100),
        "MAE (units)": float(np.mean(np.abs(err))),
        "RMSE (units)": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - (err @ err) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def train_model(history: pd.DataFrame, asof: pd.Timestamp | None = None, seed: int = 17) -> DemandModel:
    """Train on rows strictly before `asof` (no look-ahead)."""
    d = engineer(history)
    if asof is None:
        asof = d["Date"].max() + pd.Timedelta(days=1)
    asof = pd.Timestamp(asof)
    d = d[d["Date"] < asof].copy()
    if d["Date"].nunique() < 60:
        raise ValueError("Need at least 60 days of history before the selected date.")
    categories = sorted(d["Category"].unique())
    events = sorted(d["Event_Season"].unique())
    weathers = sorted(d["Climate_Weather_Condition"].unique())
    X_all = _design(d, categories, events, weathers)
    columns = list(X_all.columns)
    y_rate = d["rate"].to_numpy()
    traffic = d["Store_Traffic_Volume"].to_numpy()
    y_dem = d["Demand_QTY"].to_numpy().astype(float)

    # --- time-ordered split by date: 65% train, 20% calibration, 15% test
    dates = np.sort(d["Date"].unique())
    c1, c2 = dates[int(len(dates) * 0.65)], dates[int(len(dates) * 0.85)]
    tr = (d["Date"] < c1).to_numpy()
    ca = ((d["Date"] >= c1) & (d["Date"] < c2)).to_numpy()
    te = (d["Date"] >= c2).to_numpy()

    kind = "xgboost" if XGBOOST_AVAILABLE else "baseline"
    cat_rate_all = d.groupby("Category")["rate"].mean().to_dict()

    # choose XGBoost settings on the calibration window only (the test window is never used for selection)
    best_params: dict[str, Any] = {}
    if kind == "xgboost":
        best_mae = float("inf")
        for prm in XGB_GRID:
            mm_ = _make_xgb(seed, prm).fit(X_all[tr], y_rate[tr])
            mae = float(np.mean(np.abs(y_dem[ca] - np.maximum(mm_.predict(X_all[ca]), 0.01) * traffic[ca])))
            if mae < best_mae:
                best_mae, best_params = mae, prm

    def fit_pred(train_mask, X_pred_df, d_pred, d_train):
        if kind == "xgboost":
            m = _make_xgb(seed, best_params).fit(X_all[train_mask], y_rate[train_mask])
            return m, np.maximum(m.predict(X_pred_df), 0.01)
        cr = d_train.groupby("Category")["rate"].mean().to_dict()
        return None, d_pred["Category"].map(cr).fillna(d_train["rate"].mean()).to_numpy()

    # validation fit (train only) -> calibration residuals and test metrics
    _, rate_ca = fit_pred(tr, X_all[ca], d[ca], d[tr])
    pred_ca = rate_ca * traffic[ca]
    q = _conformal_q(np.abs(y_dem[ca] - pred_ca) / np.maximum(pred_ca, 1e-6), INTERVAL_COVERAGE)
    m_val, rate_te = fit_pred(tr | ca, X_all[te], d[te], d[tr | ca])
    pred_te = rate_te * traffic[te]
    y_te = y_dem[te]
    dte = d[te].copy()
    dte["forecast"] = pred_te
    dte["lo"] = np.maximum(0, pred_te * (1 - q))
    dte["hi"] = pred_te * (1 + q)
    # naive baselines
    cat_mean_rate = d[tr | ca].groupby("Category")["rate"].mean()
    base_a = dte["Category"].map(cat_mean_rate).to_numpy() * traffic[te]
    last = d[tr | ca].sort_values("Date").groupby("Category")["Demand_QTY"].last()
    base_b = dte["Category"].map(last).to_numpy().astype(float)
    overall = {"Retail Brain model": _metrics(y_te, pred_te),
               "Baseline: category avg rate x traffic": _metrics(y_te, base_a),
               "Baseline: last observed demand": _metrics(y_te, base_b)}
    per_cat = []
    for c in categories:
        mask = (dte["Category"] == c).to_numpy()
        if mask.sum() >= 5:
            per_cat.append({"Category": c, "Days": int(mask.sum()),
                            "Model MAPE %": _metrics(y_te[mask], pred_te[mask])["MAPE %"],
                            "Baseline MAPE %": _metrics(y_te[mask], base_a[mask])["MAPE %"]})
    coverage = float(np.mean((y_te >= dte["lo"].to_numpy()) & (y_te <= dte["hi"].to_numpy())))

    # permutation importance on the test period, grouped by feature group
    perm_rows = []
    if kind == "xgboost" and m_val is not None:
        rng = np.random.default_rng(seed)
        base_err = _metrics(y_te, (np.maximum(m_val.predict(X_all[te]), 0.01)) * traffic[te])["MAE (units)"]
        groups: dict[str, list[str]] = {}
        for c in columns:
            groups.setdefault(_feature_group(c), []).append(c)
        for g, cols in groups.items():
            incs = []
            for _ in range(8):
                Xp = X_all[te].copy()
                idx = rng.permutation(len(Xp))
                for c in cols:
                    Xp[c] = Xp[c].to_numpy()[idx]
                err = _metrics(y_te, np.maximum(m_val.predict(Xp), 0.01) * traffic[te])["MAE (units)"]
                incs.append(err - base_err)
            perm_rows.append({"Feature group": g, "Importance (MAE increase, units)": float(np.mean(incs)),
                              "Spread": float(np.std(incs))})
    perm = pd.DataFrame(perm_rows).sort_values("Importance (MAE increase, units)", ascending=False) if perm_rows else pd.DataFrame()

    series = dte[["Date", "Category", "Demand_QTY", "forecast", "lo", "hi"]].rename(columns={"Demand_QTY": "actual"})

    # deployed model: refit on all history before asof
    model_obj, explainer = None, None
    if kind == "xgboost":
        model_obj = _make_xgb(seed, best_params).fit(X_all, y_rate)
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model_obj)
            except Exception:
                explainer = None
    base_rate = float(np.mean(model_obj.predict(X_all))) if model_obj is not None else float(d["rate"].mean())
    if explainer is not None:
        try:
            ev = explainer.expected_value
            base_rate = float(np.ravel(ev)[0])
        except Exception:
            pass

    # fill-rate curve vs delay
    fx, fy = _pav_nonincreasing(d["Logistics_Delivery_Delay_Hrs"].to_numpy(), d["fill"].to_numpy())

    est = estimate_levers(d)
    med_price = d.groupby("Category")["Esselunga_Price_EUR"].median().to_dict()
    if "AI_Gross_Margin_Pct" in d.columns and d["AI_Gross_Margin_Pct"].notna().any():
        margin = (d.groupby("Category")["AI_Gross_Margin_Pct"].median() / 100.0).to_dict()
    else:
        margin = {c: DEFAULT_MARGIN for c in categories}
    comp_min = d[["Lidl_Price_EUR", "Conad_Price_EUR", "Carrefour_Price_EUR"]].min(axis=1)
    cheapest = (comp_min / d["comp_price"]).groupby(d["Category"]).median().to_dict()
    lidl = (d["Lidl_Price_EUR"] / d["comp_price"]).groupby(d["Category"]).median().to_dict()

    validation = {"overall": overall, "per_category": pd.DataFrame(per_cat), "coverage": coverage,
                  "target_coverage": INTERVAL_COVERAGE, "importance": perm, "series": series,
                  "test_start": pd.Timestamp(c2), "test_end": pd.Timestamp(d["Date"].max()),
                  "test_days": int(dte["Date"].nunique()), "interval_q": q, "xgb_params": best_params}

    return DemandModel(
        asof=asof, kind=kind, columns=columns, categories=categories, events=events, weathers=weathers,
        base_rate=base_rate, mean_traffic=float(d["Store_Traffic_Volume"].mean()),
        median_traffic=d.groupby("Category")["Store_Traffic_Volume"].median().to_dict(),
        ref_rel_price=d.groupby("Category")["rel_price"].mean().to_dict(),
        ref_promo=d.groupby("Category")["promo_active"].mean().to_dict(),
        ref_price=med_price, margin=margin, cheapest_ratio=cheapest, lidl_ratio=lidl,
        interval_q=q, fill_x=fx, fill_y=fy, lever_est=est, validation=validation,
        n_train=int(len(d)), history_start=pd.Timestamp(d["Date"].min()), history_end=pd.Timestamp(d["Date"].max()),
        cat_rate=cat_rate_all, _model=model_obj, _explainer=explainer)


# ---------------------------------------------------------------------------
# Profiles and scenario changes
# ---------------------------------------------------------------------------
def profile_from_row(row: pd.Series, model: DemandModel) -> dict[str, Any]:
    cat = str(row["Category"])
    asm = CATEGORY_ASSUMPTIONS.get(cat, DEFAULT_ASSUMPTION)
    comp = float(np.mean([row["Lidl_Price_EUR"], row["Conad_Price_EUR"], row["Carrefour_Price_EUR"]]))
    price_ref = model.ref_price.get(cat, float(row["Esselunga_Price_EUR"]))
    margin = model.margin.get(cat, DEFAULT_MARGIN)
    d = pd.Timestamp(row["Date"])
    return {
        "category": cat, "date": d.date().isoformat(),
        "price": float(row["Esselunga_Price_EUR"]), "competitor_price": comp,
        "private_label_price": float(row["Private_Label_Price_EUR"]),
        "promo_active": int(str(row["Our_Active_Promotion"]).lower() != "none"),
        "partner_active": int(str(row["Partnership_Deal_Active"]).lower() == "yes"),
        "temperature": float(row["Avg_Temperature_C"]), "traffic": float(row["Store_Traffic_Volume"]),
        "staff_capacity_pct": float(row["Staff_Capacity_Pct"]), "delay_hrs": float(row["Logistics_Delivery_Delay_Hrs"]),
        "compliance_pct": float(row["Shelf_Sorting_Compliance_Pct"]), "queue_len": float(row["Checkout_Queue_Avg_Length"]),
        "cold_chain": int(str(row["Cold_Chain_Failure_Flag"]).lower() == "yes"),
        "event": str(row["Event_Season"]), "weather": str(row["Climate_Weather_Condition"]),
        "pl_share": float(row["Private_Label_Share_Pct"]), "shelf_share_pl": float(row["Shelf_Share_Private_Label_Pct"]),
        "unit_cost": float(price_ref * (1 - margin)),
        "shelf_life_days": asm["shelf_life_days"], "base_waste_rate": asm["base_waste_rate"],
        "dow": int(d.dayofweek), "month": int(d.month),
        "lidl_ratio": float(model.cheapest_ratio.get(cat, 0.95)),
    }


def build_profiles(df: pd.DataFrame, model: DemandModel, as_of: pd.Timestamp) -> dict[str, dict[str, Any]]:
    out = {}
    day = df[df["Date"] == pd.Timestamp(as_of)]
    for _, row in day.iterrows():
        out[str(row["Category"])] = profile_from_row(row, model)
    return out


_BOUNDS = {
    "price": (0.10, 50.0), "competitor_price": (0.10, 50.0), "promo_active": (0, 1), "partner_active": (0, 1),
    "temperature": (-15.0, 48.0), "traffic": (50.0, 6000.0), "staff_capacity_pct": (20.0, 100.0),
    "delay_hrs": (0.0, 24.0), "compliance_pct": (0.0, 100.0), "queue_len": (0.0, 30.0), "cold_chain": (0, 1),
    "pl_share": (0.0, 100.0), "shelf_share_pl": (0.0, 100.0), "unit_cost": (0.05, 50.0),
}


def apply_changes(profile: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Apply scenario changes. Grammar: field_pct (multiplicative %), field_delta (additive), field_set."""
    out = dict(profile)
    for key, value in changes.items():
        if key.endswith("_pct"):
            f = key[:-4]
            if f in out:
                out[f] = float(out[f]) * (1 + float(value) / 100)
        elif key.endswith("_delta"):
            f = key[:-6]
            if f in out:
                out[f] = float(out[f]) + float(value)
        elif key.endswith("_set"):
            f = key[:-4]
            if f in out:
                out[f] = value
    for f, (lo, hi) in _BOUNDS.items():
        if f in out:
            out[f] = float(np.clip(out[f], lo, hi))
    for f in ("promo_active", "partner_active", "cold_chain"):
        out[f] = int(round(out[f]))
    return out


# ---------------------------------------------------------------------------
# 4. Agents and event bus
# ---------------------------------------------------------------------------
@dataclass
class Message:
    seq: int
    topic: str
    sender: str
    payload: dict[str, Any]


class EventBus:
    """Minimal synchronous in-process publish/subscribe bus.

    Agents never call each other. They subscribe to topics and publish new
    messages; the bus delivers them in order until no messages remain. This is
    genuine decoupling of responsibilities, but it runs in one process: it is
    not a distributed system, and the thesis says so.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Agent]] = {}
        self._queue: deque[Message] = deque()
        self.log: list[Message] = []
        self._seq = 0

    def subscribe(self, topic: str, agent: "Agent") -> None:
        self._subs.setdefault(topic, []).append(agent)

    def publish(self, topic: str, sender: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        msg = Message(self._seq, topic, sender, payload)
        self._queue.append(msg)
        self.log.append(msg)

    def run(self, ctx: dict[str, Any], max_steps: int = 500) -> None:
        steps = 0
        while self._queue and steps < max_steps:
            msg = self._queue.popleft()
            for agent in self._subs.get(msg.topic, []):
                agent.handle(msg, ctx, self)
            steps += 1


class Agent:
    name = "Agent"
    title = ""
    role = ""
    topics: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.report: dict[str, Any] | None = None
        self.inbox: dict[str, dict[str, Any]] = {}

    def handle(self, msg: Message, ctx: dict[str, Any], bus: EventBus) -> None:  # pragma: no cover
        raise NotImplementedError

    def _report(self, value: Any, label: str, unit: str, status: str, detail: str, **extra: Any) -> None:
        self.report = {"agent": self.name, "title": self.title, "role": self.role, "value": value,
                       "label": label, "unit": unit, "status": status, "detail": detail, **extra}


class DemandForecastAgent(Agent):
    name, title = "demand_forecast", "1. Demand forecasting"
    role = "Forecasts daily demand with an 80% prediction interval"
    topics = ("ctx.ready",)

    def handle(self, msg, ctx, bus):
        p, model, levers = ctx["profile"], ctx["model"], ctx["levers"]
        pred = model.predict(p, levers)
        self._report(pred["demand"], "Forecast demand", "units/day", "ok",
                     f"80% interval {pred['low']:.0f} to {pred['high']:.0f} units/day",
                     low=pred["low"], high=pred["high"])
        bus.publish("forecast.ready", self.name, {**pred})


class InventoryRiskAgent(Agent):
    name, title = "inventory_risk", "2. Inventory risk"
    role = "Turns delivery delay into expected unfulfilled demand"
    topics = ("forecast.ready",)

    def handle(self, msg, ctx, bus):
        f, p, model = msg.payload, ctx["profile"], ctx["model"]
        fill = model.fill_rate(p["delay_hrs"])
        lost = f["demand"] * (1 - fill)
        lost_hi = f["high"] * (1 - fill)
        sales = f["demand"] - lost
        pct = (1 - fill) * 100
        status = "alert" if pct >= 25 else "watch" if pct >= 8 else "ok"
        self._report(pct, "Stockout risk (unfulfilled demand)", "%", status,
                     f"Delay {p['delay_hrs']:.1f} h gives an expected fill rate of {fill*100:.0f}% "
                     f"(learned from {model.n_train:,} days of history)",
                     fill=fill, lost=lost)
        bus.publish("supply.risk", self.name, {"sales": sales, "lost": lost, "lost_high": lost_hi,
                                               "stockout_pct": pct, "fill": fill,
                                               "demand": f["demand"], "high": f["high"]})


class ReplenishmentAgent(Agent):
    name, title = "replenishment", "4. Replenishment"
    role = "Sizes the safety buffer from the demand interval"
    topics = ("supply.risk",)

    def handle(self, msg, ctx, bus):
        s = msg.payload
        buffer_units = max(0.0, (s["high"] - s["demand"]) + s["lost"])
        status = "alert" if s["stockout_pct"] >= 25 else "watch" if buffer_units > 0.15 * max(s["demand"], 1) else "ok"
        self._report(buffer_units, "Recommended safety buffer", "units", status,
                     "Covers the upper end of the demand interval plus units expected to be lost to delivery delay")
        bus.publish("replenishment.plan", self.name, {"buffer_units": buffer_units})


class CommercialIntelligenceAgent(Agent):
    name, title = "commercial_intelligence", "3. Commercial intelligence"
    role = "Compares own price with competitors and private label"
    topics = ("forecast.ready",)

    def handle(self, msg, ctx, bus):
        p = ctx["profile"]
        comp = p["competitor_price"]
        gap_pct = (p["price"] / comp - 1) * 100
        cheapest = comp * p.get("lidl_ratio", 0.95)
        status = "alert" if gap_pct > 10 else "watch" if gap_pct > 4 else "ok"
        self._report(gap_pct, "Price vs competitor mean", "%", status,
                     f"Own EUR {p['price']:.2f}, competitor mean EUR {comp:.2f}, cheapest about EUR {cheapest:.2f}, "
                     f"private label EUR {p['private_label_price']:.2f}", gap_eur=p["price"] - comp)
        bus.publish("price.position", self.name, {"gap_pct": gap_pct})


class PromotionDisplayAgent(Agent):
    name, title = "promotion_display", "7. Promotion & display"
    role = "Reads price position and checks whether a campaign is justified"
    topics = ("price.position",)

    def handle(self, msg, ctx, bus):
        p, levers = ctx["profile"], ctx["levers"]
        gap = msg.payload["gap_pct"]
        if p["promo_active"]:
            note = f"Campaign active; assumed uplift {levers['promo_uplift']*100:+.0f}% ({levers['promo_source']})"
            status = "ok"
        elif gap > 4:
            note = f"No campaign while priced {gap:.1f}% above competitors; a campaign could be considered"
            status = "watch"
        else:
            note = "No campaign needed on price position"
            status = "ok"
        self._report("On" if p["promo_active"] else "Off", "Promotion", "state", status,
                     note + f". Private-label shelf share {p['shelf_share_pl']:.0f}%.")


class FreshnessWasteAgent(Agent):
    name, title = "freshness_waste", "5. Freshness & waste"
    role = "Estimates waste cost from shelf life, heat and cold-chain failures"
    topics = ("supply.risk",)

    def handle(self, msg, ctx, bus):
        s, p = msg.payload, ctx["profile"]
        perishable = p["shelf_life_days"] <= 7
        rate = p["base_waste_rate"] * (1 + max(0.0, p["temperature"] - 25) * 0.04 if perishable else 1.0)
        waste_units = s["sales"] * rate
        if p["cold_chain"]:
            waste_units += COLD_CHAIN_WRITEOFF_SHARE * s["demand"]
        freshness = 1.0 if p["cold_chain"] else (
            float(np.clip((3 - p["shelf_life_days"]) / 3, 0, 1)) if perishable else 0.0)
        cost = waste_units * p["unit_cost"]
        status = "alert" if freshness >= 0.7 else "watch" if freshness >= 0.35 or cost > 0.05 * max(s["sales"] * p["price"], 1) else "ok"
        self._report(cost, "Waste cost proxy", "EUR/day", status,
                     f"Base waste rate {p['base_waste_rate']*100:.1f}% (assumption), shelf life {p['shelf_life_days']} days"
                     + ("; cold-chain failure flagged" if p["cold_chain"] else ""),
                     freshness=freshness, waste_units=waste_units)
        bus.publish("waste.estimate", self.name, {"waste_cost": cost, "freshness": freshness})


class WorkforcePlanningAgent(Agent):
    name, title = "workforce_planning", "6. Workforce planning"
    role = "Compares traffic load with staff capacity"
    topics = ("forecast.ready",)

    def handle(self, msg, ctx, bus):
        p, model = ctx["profile"], ctx["model"]
        typical = model.median_traffic.get(p["category"], model.mean_traffic)
        load = p["traffic"] / max(typical, 1.0)
        pressure = float(np.clip(load / max(p["staff_capacity_pct"] / 100.0, 0.2) - 1.0, 0, 2))
        status = "alert" if pressure >= 1.0 else "watch" if pressure >= 0.5 else "ok"
        self._report(pressure, "Staffing pressure index", "index", status,
                     f"Traffic is {load:.2f}x typical with {p['staff_capacity_pct']:.0f}% staff capacity; "
                     f"checkout queue {p['queue_len']:.1f}")
        bus.publish("staffing.pressure", self.name, {"pressure": pressure})


class ComplianceTaskAgent(Agent):
    name, title = "compliance_tasks", "9. Compliance & task management"
    role = "Raises compliance flags from shelf, cold-chain, queue and staffing signals"
    topics = ("staffing.pressure", "waste.estimate")

    def handle(self, msg, ctx, bus):
        self.inbox[msg.topic] = msg.payload
        if len(self.inbox) < 2:
            return
        p = ctx["profile"]
        flags = []
        if p["compliance_pct"] < 85:
            flags.append(f"Shelf compliance {p['compliance_pct']:.0f}% is below 85%")
        if p["cold_chain"]:
            flags.append("Cold-chain failure reported")
        if p["queue_len"] >= 5:
            flags.append(f"Checkout queue {p['queue_len']:.1f} is at or above 5")
        if self.inbox["staffing.pressure"]["pressure"] >= 1.0:
            flags.append("Staffing pressure at or above 1.0")
        critical = p["cold_chain"] or len(flags) >= 2
        status = "alert" if critical else "watch" if flags else "ok"
        self._report(len(flags), "Compliance flags", "flags", status, "; ".join(flags) if flags else "No active flags",
                     flags=flags, critical=bool(critical))
        bus.publish("compliance.flags", self.name, {"count": len(flags), "critical": bool(critical)})


class ProfitabilityAgent(Agent):
    name, title = "profitability", "8. Profitability"
    role = "Combines sales, price, cost and waste into gross profit"
    topics = ("supply.risk", "waste.estimate")

    def handle(self, msg, ctx, bus):
        self.inbox[msg.topic] = msg.payload
        if len(self.inbox) < 2:
            return
        p = ctx["profile"]
        s, w = self.inbox["supply.risk"], self.inbox["waste.estimate"]
        revenue = s["sales"] * p["price"]
        gross = s["sales"] * (p["price"] - p["unit_cost"])
        net = gross - w["waste_cost"]
        margin = (p["price"] - p["unit_cost"]) / max(p["price"], 0.01) * 100
        status = "watch" if net < 0.8 * gross else "ok"
        self._report(gross, "Gross profit", "EUR/day", status,
                     f"Revenue EUR {revenue:.2f}, margin {margin:.1f}% (unit cost is an assumption derived from category margin), "
                     f"net of waste EUR {net:.2f}", revenue=revenue, net=net, margin=margin)


AGENT_CLASSES = [DemandForecastAgent, InventoryRiskAgent, CommercialIntelligenceAgent, ReplenishmentAgent,
                 FreshnessWasteAgent, WorkforcePlanningAgent, PromotionDisplayAgent, ProfitabilityAgent,
                 ComplianceTaskAgent]


def analyze(profile: dict[str, Any], model: DemandModel, levers: dict[str, Any],
            keep_log: bool = True) -> dict[str, Any]:
    """Run all nine agents over the event bus and assemble a metrics dict."""
    bus = EventBus()
    agents = [cls() for cls in AGENT_CLASSES]
    for a in agents:
        for t in a.topics:
            bus.subscribe(t, a)
    ctx = {"profile": profile, "model": model, "levers": levers}
    bus.publish("ctx.ready", "orchestrator", {"category": profile["category"]})
    bus.run(ctx)
    rep = {a.name: a.report for a in agents if a.report}
    f, inv, prof = rep["demand_forecast"], rep["inventory_risk"], rep["profitability"]
    waste, work, comp = rep["freshness_waste"], rep["workforce_planning"], rep["compliance_tasks"]
    stockout, fresh = inv["value"], waste["freshness"]
    pressure = work["value"]
    if comp["critical"] or stockout >= 25 or fresh >= 0.7:
        status = "Critical"
    elif comp["value"] >= 1 or stockout >= 8 or fresh >= 0.35 or pressure >= 0.5:
        status = "Warning"
    else:
        status = "Adequate"
    metrics = {
        "category": profile["category"], "demand_units": f["value"], "demand_low": f["low"], "demand_high": f["high"],
        "sales_units": f["value"] - inv["lost"], "lost_sales_units": inv["lost"],
        "revenue_eur": prof["revenue"], "gross_profit_eur": prof["value"], "net_profit_eur": prof["net"],
        "margin_pct": prof["margin"], "safety_stock_units": rep["replenishment"]["value"],
        "stockout_risk_pct": stockout, "waste_cost_eur": waste["value"], "freshness_risk": fresh,
        "staffing_pressure": pressure, "compliance_flags": int(comp["value"]),
        "compliance_risk": bool(comp["value"] >= 1), "status": status,
        "price_gap_pct": rep["commercial_intelligence"]["value"],
        "agent_reports": rep,
    }
    if keep_log:
        metrics["messages"] = [{"#": m.seq, "Topic": m.topic, "From": m.sender,
                                "Payload": ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in m.payload.items())} for m in bus.log]
    return metrics


def global_shap(model: DemandModel, history: pd.DataFrame, n_days: int = 120) -> pd.DataFrame:
    """Mean absolute SHAP per feature group, in units/day, over the most recent days before the model's as-of date."""
    if model.kind != "xgboost" or model._explainer is None:
        return pd.DataFrame()
    d = engineer(history)
    d = d[d["Date"] < model.asof]
    cutoff = d["Date"].max() - pd.Timedelta(days=n_days)
    d = d[d["Date"] > cutoff]
    if d.empty:
        return pd.DataFrame()
    X = _design(d, model.categories, model.events, model.weathers).reindex(columns=model.columns, fill_value=0.0)
    sv = np.asarray(model._explainer.shap_values(X))
    scaled = np.abs(sv) * d["Store_Traffic_Volume"].to_numpy()[:, None]
    groups: dict[str, float] = {}
    for j, col in enumerate(model.columns):
        g = _feature_group(col)
        groups[g] = groups.get(g, 0.0) + float(scaled[:, j].mean())
    out = pd.DataFrame({"Feature group": list(groups), "Mean |effect| (units/day)": list(groups.values())})
    return out.sort_values("Mean |effect| (units/day)", ascending=False).reset_index(drop=True)
