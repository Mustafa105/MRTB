from __future__ import annotations

import json
import re
import random
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Retail Multi-Agent Store Operations MVP (Retail Brain)", page_icon="🛒", layout="wide")

# Keep explanatory captions and secondary labels readable in both Streamlit themes.
# Use Streamlit's theme text variable so auto/light/dark themes are respected.
st.markdown("""
<style>
/* Retail Brain analytics workspace: inspired by the supplied clean SaaS dashboard */
:root { --rb-purple: #6254c7; --rb-purple-dark: #493ca9; --rb-orange: #ff7a18; --rb-ink: #22243a; --rb-muted: #74778b; --rb-card: var(--secondary-background-color); }
.stApp { background: linear-gradient(135deg, rgba(98,84,199,.045), transparent 42%), var(--background-color); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.15rem; padding-bottom: 2.25rem; max-width: 1500px; }
.stApp h1 { font-size: clamp(1.55rem, 2.3vw, 2.15rem); font-weight: 750; letter-spacing: -.035em; color: var(--text-color); }
.stApp h2, .stApp h3 { letter-spacing: -.02em; font-weight: 700; }
.stApp [data-testid="stMetric"] { background: var(--secondary-background-color); border: 1px solid rgba(120,120,150,.16); border-radius: 14px; padding: 14px 15px; box-shadow: 0 3px 12px rgba(40,35,90,.035); min-height: 96px; }
.stApp [data-testid="stMetricLabel"] { font-size: .82rem; font-weight: 600; }
.stApp [data-testid="stMetricValue"] { font-size: clamp(1.25rem, 1.8vw, 1.8rem); font-weight: 750; color: var(--rb-purple); }
.stApp [data-testid="stMetricDelta"] { font-size: .78rem; }
.stApp [data-testid="stTabs"] [role="tablist"] { gap: 5px; border-bottom: 1px solid rgba(120,120,150,.18); }
.stApp [data-testid="stTabs"] button[role="tab"] { border-radius: 10px 10px 0 0; padding: .7rem .85rem; font-weight: 650; }
.stApp [data-testid="stTabs"] button[aria-selected="true"] { color: var(--rb-purple) !important; border-bottom: 3px solid var(--rb-purple) !important; }
.stApp [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid rgba(120,120,150,.16); }
.stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.5; }
.stApp [data-testid="stSidebar"] button[kind="primary"] { background: var(--rb-purple); border: 1px solid var(--rb-purple); color: #fff; border-radius: 11px; font-weight: 700; min-height: 42px; text-align: left; box-shadow: 0 3px 10px rgba(98,84,199,.12); }
.stApp [data-testid="stSidebar"] button[kind="secondary"] { background: transparent; border: 1px solid rgba(120,120,150,.06); border-radius: 11px; font-weight: 600; min-height: 42px; text-align: left; }
.stApp [data-testid="stSidebar"] button { transition: background .15s ease, border-color .15s ease; }
.stApp [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .45rem; }
.stApp [data-testid="stMainBlockContainer"] { padding-left: 1.45rem; padding-right: 1.45rem; }
.stApp [data-testid="stMarkdownContainer"] h3 { font-size: 1.03rem; }
.stApp [data-testid="stMarkdownContainer"] h2 { font-size: 1.18rem; }
.stApp [data-testid="stSidebar"] button[kind="secondary"]:hover { background: rgba(98,84,199,.08); border-color: rgba(98,84,199,.16); }
.stApp button[kind="primary"] { background: var(--rb-purple); border-color: var(--rb-purple); color: #fff; border-radius: 10px; font-weight: 650; }
.stApp button[kind="secondary"], .stApp button:not([kind]) { border-radius: 10px; }
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"] { border: 1px solid rgba(120,120,150,.16); border-radius: 12px; overflow: hidden; }
.stApp [data-testid="stPlotlyChart"], .stApp [data-testid="stVegaLiteChart"] { background: var(--secondary-background-color); border: 1px solid rgba(120,120,150,.13); border-radius: 14px; padding: 8px; }
.stApp [data-testid="stAlert"] { border-radius: 12px; }
.stApp hr { border-color: rgba(120,120,150,.18); }
@media (max-width: 800px) { .block-container { padding: 1rem .8rem 2rem; } .stApp [data-testid="stMetric"] { padding: 12px; } }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stMarkdown small,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stWidgetLabel"] {
    color: var(--text-color) !important;
    opacity: 1 !important;
}
/* Give secondary explanatory text sufficient contrast in light mode. */
@media (prefers-color-scheme: light) {
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #374151 !important;
        opacity: 1 !important;
    }
}
@media (prefers-color-scheme: dark) {
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration and scenario templates
# -----------------------------------------------------------------------------
CATEGORIES = {
    "Birra": {
        "category_factor": 1.00, "price": 1.29, "competitor_price": 1.35,
        "promo_pct": 5, "display_pct": 35, "temperature": 22, "traffic_idx": 100,
        "stock_units": 420, "lead_time_days": 2, "expiry_days": 120,
        "staffing_units": 3, "unit_cost": 0.72, "base_daily_demand": 115,
        "waste_rate": 0.01, "compliance_issue": False,
    },
    "Prodotti da Forno": {
        "category_factor": 0.92, "price": 1.49, "competitor_price": 1.55,
        "promo_pct": 8, "display_pct": 45, "temperature": 22, "traffic_idx": 100,
        "stock_units": 180, "lead_time_days": 1, "expiry_days": 2,
        "staffing_units": 3, "unit_cost": 0.76, "base_daily_demand": 95,
        "waste_rate": 0.09, "compliance_issue": False,
    },
    "Ortofrutta": {
        "category_factor": 1.12, "price": 2.49, "competitor_price": 2.59,
        "promo_pct": 6, "display_pct": 50, "temperature": 22, "traffic_idx": 100,
        "stock_units": 210, "lead_time_days": 1, "expiry_days": 4,
        "staffing_units": 4, "unit_cost": 1.30, "base_daily_demand": 105,
        "waste_rate": 0.12, "compliance_issue": False,
    },
    "Detersivi": {
        "category_factor": 0.78, "price": 3.49, "competitor_price": 3.69,
        "promo_pct": 4, "display_pct": 25, "temperature": 22, "traffic_idx": 100,
        "stock_units": 260, "lead_time_days": 3, "expiry_days": 365,
        "staffing_units": 2, "unit_cost": 1.95, "base_daily_demand": 58,
        "waste_rate": 0.005, "compliance_issue": False,
    },
}

FEATURES = [
    "price", "competitor_price", "promo_pct", "display_pct", "temperature",
    "traffic_idx", "stock_units", "lead_time_days", "expiry_days", "staffing_units",
    "unit_cost", "category_factor",
]
FEATURE_LABELS = {
    "price": "Own price (€)", "competitor_price": "Competitor price (€)",
    "promo_pct": "Promotion (%)", "display_pct": "Display quality (%)",
    "temperature": "Temperature (°C)", "traffic_idx": "Store traffic index",
    "stock_units": "Available stock (units)", "lead_time_days": "Lead time (days)",
    "expiry_days": "Days to expiry", "staffing_units": "Staffing capacity",
    "unit_cost": "Unit cost (€)", "category_factor": "Category demand factor",
}

SCENARIO_TEMPLATES = {
    "Competitor price reduction": {
        "description": "A competitor cuts price, potentially diverting price-sensitive demand.",
        "changes": {"competitor_price_pct": -12, "traffic_pct": -5, "promo_pct_points": 2},
        "variables": ["competitor_price", "traffic_idx", "promo_pct"],
    },
    "Demand spike and delivery delay": {
        "description": "Traffic rises while a delayed delivery tightens availability and staffing.",
        "changes": {"traffic_pct": 18, "stock_units_delta": -35, "lead_time_days_delta": 2, "staffing_units_delta": -1},
        "variables": ["traffic_idx", "stock_units", "lead_time_days", "staffing_units"],
    },
    "Promotion and display opportunity": {
        "description": "A stronger promotion and better display may lift demand but consume stock.",
        "changes": {"promo_pct_points": 12, "display_pct_points": 20, "stock_units_delta": -20, "staffing_units_delta": 1},
        "variables": ["promo_pct", "display_pct", "stock_units", "staffing_units"],
    },
    "Freshness and waste risk": {
        "description": "Shorter remaining shelf life increases markdown and waste pressure.",
        "changes": {"expiry_days_delta": -2, "traffic_pct": -8, "promo_pct_points": 5},
        "variables": ["expiry_days", "traffic_idx", "promo_pct"],
    },
    "Hot weather and traffic increase": {
        "description": "Warmer weather and more visitors can increase demand for selected categories.",
        "changes": {"temperature_delta": 7, "traffic_pct": 15, "stock_units_delta": -25},
        "variables": ["temperature", "traffic_idx", "stock_units"],
    },
    "Cost and margin pressure": {
        "description": "Own-price, competitor-price and promotion pressure affect estimated gross margin.",
        "changes": {"price_pct": 5, "competitor_price_pct": -4, "promo_pct_points": -2},
        "variables": ["price", "competitor_price", "promo_pct"],
    },
    "Staffing and compliance alert": {
        "description": "Lower staffing during higher traffic creates an operational compliance risk.",
        "changes": {"staffing_units_delta": -2, "traffic_pct": 6, "compliance_issue": True},
        "variables": ["staffing_units", "traffic_idx", "compliance_issue"],
    },
}

# -----------------------------------------------------------------------------
# Lightweight audit persistence (SQLite; path can be overridden by environment)
# -----------------------------------------------------------------------------
AUDIT_DB_PATH = Path(__import__("os").environ.get("RETAIL_BRAIN_AUDIT_DB", "retail_brain_audit.sqlite3"))

def init_audit_db() -> None:
    """Create a small local audit table. On hosted platforms, durable storage depends on the host filesystem."""
    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS decision_history (event_id TEXT, decision_at TEXT, payload TEXT NOT NULL)")
            conn.commit()
    except Exception as exc:
        st.session_state["persistence_warning"] = str(exc)

def load_persisted_decisions() -> list[dict[str, Any]]:
    try:
        init_audit_db()
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            rows = conn.execute("SELECT payload FROM decision_history ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]
    except Exception as exc:
        st.session_state["persistence_warning"] = str(exc)
        return []

def persist_decision(record: dict[str, Any]) -> None:
    try:
        init_audit_db()
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            conn.execute("INSERT INTO decision_history (event_id, decision_at, payload) VALUES (?, ?, ?)",
                         (str(record.get("event_id", "")), str(record.get("decision_at", "")), json.dumps(record, ensure_ascii=False, default=str)))
            conn.commit()
    except Exception as exc:
        st.session_state["persistence_warning"] = str(exc)

def shap_explanation(profile: dict[str, Any]):
    """Return genuine TreeSHAP values for the fitted XGBoost model, or a clear unavailable reason."""
    if not SHAP_AVAILABLE or not XGBOOST_AVAILABLE:
        return None, "Real SHAP requires both the shap and xgboost packages to be available."
    model, _ = get_model()
    if not isinstance(model, XGBRegressor):
        return None, "The active demand model is the fallback estimator; TreeSHAP is not applicable."
    try:
        row = pd.DataFrame([{key: float(profile[key]) for key in FEATURES}], columns=FEATURES)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(row)
        values = np.asarray(values).reshape(-1)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).reshape(-1)[0]
        return {"row": row, "values": values, "base_value": float(base_value),
                "prediction": float(model.predict(row)[0]), "explainer": explainer}, None
    except Exception as exc:
        return None, f"SHAP could not calculate an explanation in this environment: {exc}"

def render_shap_waterfall(profile: dict[str, Any], key: str) -> None:
    explanation, error = shap_explanation(profile)
    if explanation is None:
        st.info(error or "A SHAP explanation is unavailable.")
        return
    try:
        import matplotlib.pyplot as plt
        vals = explanation["values"]
        exp = shap.Explanation(values=vals, base_values=explanation["base_value"],
                               data=explanation["row"].iloc[0].to_numpy(dtype=float),
                               feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES])
        fig = plt.figure(figsize=(10, 5.5))
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.title("SHAP explanation for estimated daily demand")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(f"SHAP base value: {explanation['base_value']:.1f} units/day · model estimate: {explanation['prediction']:.1f} units/day. These explain the fitted model, not causal effects.")
    except Exception as exc:
        st.info(f"SHAP values were calculated, but the waterfall plot could not be rendered: {exc}")

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "events": [], "decision_history": [], "approved_event_ids": [],
        "base_profiles": {k: dict(v) for k, v in CATEGORIES.items()},
        "last_scan_at": None, "last_auto_event_at": None,
        "auto_monitor": False, "sound_enabled": False,
        "scan_count": 0, "model_error": None,
        "last_review_event_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "audit_history_loaded" not in st.session_state:
        st.session_state.decision_history = load_persisted_decisions()
        st.session_state.audit_history_loaded = True

init_state()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def current_profile(category: str, exclude_event_id: str | None = None) -> dict[str, Any]:
    """Apply approved scenario deltas to the base category profile."""
    profile = dict(st.session_state.base_profiles[category])
    # Events are stored newest-first. Applying oldest-first makes deltas deterministic.
    for event in reversed(st.session_state.events):
        if event.get("status") != "approved" or event.get("category") != category:
            continue
        if exclude_event_id and event.get("id") == exclude_event_id:
            continue
        profile = apply_changes_to_profile(profile, event.get("changes", {}))
    return profile


def apply_changes_to_profile(profile: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    for key, value in changes.items():
        if key.endswith("_pct"):
            field = key[:-4]
            if field in out:
                out[field] = float(out[field]) * (1 + float(value) / 100)
        elif key.endswith("_pct_points"):
            field = key[:-11]
            if field in out:
                out[field] = float(out[field]) + float(value)
        elif key.endswith("_delta"):
            field = key[:-6]
            if field in out:
                out[field] = float(out[field]) + float(value)
        elif key == "compliance_issue":
            out[key] = bool(value)
    # Guardrails keep simulated inputs within reasonable bounds.
    out["price"] = max(0.10, float(out["price"]))
    out["competitor_price"] = max(0.10, float(out["competitor_price"]))
    out["promo_pct"] = float(np.clip(out["promo_pct"], 0, 70))
    out["display_pct"] = float(np.clip(out["display_pct"], 0, 100))
    out["temperature"] = float(np.clip(out["temperature"], -10, 45))
    out["traffic_idx"] = float(np.clip(out["traffic_idx"], 20, 250))
    out["stock_units"] = max(0, int(round(out["stock_units"])))
    out["lead_time_days"] = max(0, int(round(out["lead_time_days"])))
    out["expiry_days"] = max(0, int(round(out["expiry_days"])))
    out["staffing_units"] = max(0, int(round(out["staffing_units"])))
    return out

# -----------------------------------------------------------------------------
# Demand estimation — XGBoost where available, transparent formula fallback
# -----------------------------------------------------------------------------
def make_training_data(n: int = 1800, seed: int = 17) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows, targets = [], []
    for category, base in CATEGORIES.items():
        for _ in range(n // len(CATEGORIES)):
            price = max(0.2, rng.normal(base["price"], base["price"] * 0.12))
            comp = max(0.2, rng.normal(base["competitor_price"], base["competitor_price"] * 0.10))
            promo = float(np.clip(rng.normal(base["promo_pct"], 8), 0, 60))
            display = float(np.clip(rng.normal(base["display_pct"], 18), 0, 100))
            temp = float(rng.normal(22, 8))
            traffic = float(np.clip(rng.normal(100, 18), 30, 220))
            stock = float(np.clip(rng.normal(base["stock_units"], max(20, base["stock_units"] * .25)), 0, 900))
            lead = float(np.clip(rng.normal(base["lead_time_days"], 1.1), 0, 8))
            expiry = float(np.clip(rng.normal(base["expiry_days"], max(1, base["expiry_days"] * .08)), 0, 400))
            staffing = float(np.clip(rng.normal(base["staffing_units"], 1), 0, 10))
            cost = max(0.05, rng.normal(base["unit_cost"], base["unit_cost"] * .06))
            factor = base["category_factor"]
            row = dict(price=price, competitor_price=comp, promo_pct=promo, display_pct=display,
                       temperature=temp, traffic_idx=traffic, stock_units=stock,
                       lead_time_days=lead, expiry_days=expiry, staffing_units=staffing,
                       unit_cost=cost, category_factor=factor)
            price_effect = np.clip((base["price"] / price) ** 0.7, .55, 1.8)
            competitive_effect = np.clip(1 + ((comp - price) / max(price, .1)) * .25, .7, 1.3)
            promo_effect = 1 + promo * .009
            display_effect = .75 + display * .005
            weather_effect = 1 + (temp - 22) * (0.004 if category in ("Birra", "Ortofrutta") else 0.001)
            traffic_effect = traffic / 100
            stock_effect = min(1, stock / max(base["base_daily_demand"] * 1.15, 1))
            staff_effect = min(1.12, .72 + staffing * .10)
            lead_effect = max(.75, 1 - max(0, lead - base["lead_time_days"]) * .025)
            expiry_effect = .75 if category in ("Ortofrutta", "Prodotti da Forno") and expiry <= 1 else 1
            demand = base["base_daily_demand"] * price_effect * competitive_effect * promo_effect * display_effect * weather_effect * traffic_effect * stock_effect * staff_effect * lead_effect * expiry_effect
            demand *= rng.normal(1, .09)
            rows.append(row)
            targets.append(max(0, demand))
    return pd.DataFrame(rows)[FEATURES], pd.Series(targets, name="daily_demand")


class LinearFallback:
    """Deterministic formula fallback used when XGBoost is unavailable."""
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearFallback":
        self.coef_ = None
        self.mean_target_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Transparent multiplicative proxy; predictions are calibrated to training mean.
        price = np.maximum(X["price"].to_numpy(), .1)
        comp = np.maximum(X["competitor_price"].to_numpy(), .1)
        promo = X["promo_pct"].to_numpy()
        display = X["display_pct"].to_numpy()
        traffic = X["traffic_idx"].to_numpy()
        stock = X["stock_units"].to_numpy()
        staffing = X["staffing_units"].to_numpy()
        lead = X["lead_time_days"].to_numpy()
        expiry = X["expiry_days"].to_numpy()
        factor = X["category_factor"].to_numpy()
        raw = (factor * 85 * (1 / price) ** .22 * (1 + np.clip((comp-price)/price, -.3, .3)*.22)
               * (1 + promo*.009) * (.75 + display*.005) * (traffic/100)
               * np.minimum(1.12, .72 + staffing*.10) * np.maximum(.75, 1-np.maximum(0, lead-2)*.025)
               * np.minimum(1, stock/90))
        raw *= np.where(expiry <= 1, .78, 1.0)
        return np.maximum(0, raw)


@st.cache_resource(show_spinner=False)
def get_model() -> tuple[Any, str]:
    X, y = make_training_data()
    if XGBOOST_AVAILABLE:
        try:
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=.06,
                                 subsample=.9, colsample_bytree=.9, objective="reg:squarederror",
                                 random_state=17, n_jobs=2)
            model.fit(X, y)
            return model, "XGBoost regressor trained on synthetic data"
        except Exception as exc:
            st.session_state.model_error = str(exc)
    model = LinearFallback().fit(X, y)
    return model, "Transparent formula fallback (XGBoost unavailable or failed)"


def predict_demand(profile: dict[str, Any]) -> float:
    model, _ = get_model()
    row = pd.DataFrame([{key: float(profile[key]) for key in FEATURES}])[FEATURES]
    try:
        return max(0.0, float(model.predict(row)[0]))
    except Exception:
        fallback = LinearFallback()
        return max(0.0, float(fallback.predict(row)[0]))


def calculate_metrics(category: str, profile: dict[str, Any]) -> dict[str, float | str]:
    demand = predict_demand(profile)
    stock = max(0, float(profile["stock_units"]))
    sales_units = min(demand, stock)
    lost_units = max(0, demand - stock)
    revenue = sales_units * float(profile["price"])
    gross_profit = sales_units * (float(profile["price"]) - float(profile["unit_cost"]))
    stock_cover = stock / max(demand, 1)
    stockout_risk = float(np.clip(lost_units / max(demand, 1), 0, 1))
    if category in ("Ortofrutta", "Prodotti da Forno"):
        freshness_risk = float(np.clip((3 - float(profile["expiry_days"])) / 3, 0, 1))
    else:
        freshness_risk = float(np.clip((2 - float(profile["expiry_days"])) / 2, 0, 1))
    waste_units = sales_units * float(profile.get("waste_rate", .02)) + (stock * freshness_risk * .12)
    waste_cost = waste_units * float(profile["unit_cost"])
    margin_pct = ((float(profile["price"]) - float(profile["unit_cost"])) / max(float(profile["price"]), .01)) * 100
    staffing_pressure = float(np.clip((demand / max(float(profile["staffing_units"])*40, 1)) - 1, 0, 2))
    compliance_risk = bool(profile.get("compliance_issue", False) or (profile["staffing_units"] <= 1 and profile["traffic_idx"] >= 110))
    if compliance_risk or stockout_risk >= .25 or freshness_risk >= .7:
        status = "Critical"
    elif stockout_risk >= .08 or freshness_risk >= .35 or staffing_pressure >= .5:
        status = "Warning"
    else:
        status = "Adequate"
    return {
        "category": category, "demand_units": demand, "sales_units": sales_units,
        "lost_sales_units": lost_units, "revenue_eur": revenue, "gross_profit_eur": gross_profit,
        "margin_pct": margin_pct, "stock_cover_days": stock_cover,
        "stockout_risk_pct": stockout_risk*100, "waste_cost_eur": waste_cost,
        "staffing_pressure": staffing_pressure, "compliance_risk": compliance_risk,
        "status": status,
    }


def metrics_frame(profiles: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    profiles = profiles or {c: current_profile(c) for c in CATEGORIES}
    rows = [calculate_metrics(c, profiles[c]) for c in CATEGORIES]
    return pd.DataFrame(rows)


def scenario_metrics(category: str, changes: dict[str, Any], exclude_event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    before_profile = current_profile(category, exclude_event_id=exclude_event_id)
    after_profile = apply_changes_to_profile(before_profile, changes)
    before = calculate_metrics(category, before_profile)
    after = calculate_metrics(category, after_profile)
    return before_profile, after_profile, before, after


def describe_changes(changes: dict[str, Any]) -> str:
    labels = {
        "price_pct": "Own price", "competitor_price_pct": "Competitor price",
        "promo_pct_points": "Promotion", "display_pct_points": "Display quality",
        "temperature_delta": "Temperature", "traffic_pct": "Store traffic",
        "stock_units_delta": "Stock", "lead_time_days_delta": "Lead time",
        "expiry_days_delta": "Days to expiry", "staffing_units_delta": "Staffing",
        "compliance_issue": "Compliance flag",
    }
    parts = []
    for key, value in changes.items():
        label = labels.get(key, key)
        if key.endswith("_pct"):
            parts.append(f"{label} {float(value):+g}%")
        elif key.endswith("_pct_points"):
            parts.append(f"{label} {float(value):+g} pp")
        elif key.endswith("_delta"):
            parts.append(f"{label} {float(value):+g}")
        else:
            parts.append(f"{label}: {value}")
    return " · ".join(parts)


def create_scenario(template_name: str, category: str, source: str = "Manual scan") -> dict[str, Any]:
    template = SCENARIO_TEMPLATES[template_name]
    changes = dict(template["changes"])
    # Make selected scenario variables category-sensitive while keeping effects interpretable.
    if template_name == "Freshness and waste risk" and category not in ("Ortofrutta", "Prodotti da Forno"):
        changes["expiry_days_delta"] = -1
    if template_name == "Hot weather and traffic increase" and category == "Detersivi":
        changes["traffic_pct"] = 8
    before_profile, after_profile, before, after = scenario_metrics(category, changes)
    delta_demand = float(after["demand_units"]) - float(before["demand_units"])
    delta_profit = float(after["gross_profit_eur"]) - float(before["gross_profit_eur"])
    risk_rank = {"Adequate": 0, "Warning": 1, "Critical": 2}
    severity = "Critical" if after["status"] == "Critical" or risk_rank[after["status"]] > risk_rank[before["status"]] else ("Warning" if after["status"] == "Warning" else "Info")
    return {
        "id": str(uuid.uuid4()), "created_at": now_iso(), "template": template_name,
        "description": template["description"], "category": category, "changes": changes,
        "variables": list(template["variables"]), "status": "pending", "source": source,
        "severity": severity, "before": before, "after": after,
        "ai_recommendation": "approve" if delta_profit >= 0 and after.get("status") != "Critical" else "reject",
        "delta_demand_units": delta_demand, "delta_gross_profit_eur": delta_profit,
        "change_summary": describe_changes(changes),
    }


def scan_scenarios(count: int = 3, source: str = "Manual scan") -> list[dict[str, Any]]:
    names = random.sample(list(SCENARIO_TEMPLATES.keys()), k=min(count, len(SCENARIO_TEMPLATES)))
    new_events = []
    for name in names:
        category = random.choice(list(CATEGORIES.keys()))
        event = create_scenario(name, category, source)
        new_events.append(event)
    st.session_state.events = new_events + st.session_state.events
    st.session_state.last_scan_at = now_iso()
    st.session_state.scan_count += 1
    return new_events


def get_live_event_impact(event: dict[str, Any]) -> dict[str, Any]:
    """Current counterfactual impact of an approved event, not a historical causal estimate."""
    category = event["category"]
    _, _, before, after = scenario_metrics(category, event["changes"], exclude_event_id=event["id"])
    return {
        "category": category, "event_id": event["id"],
        "before_demand": before["demand_units"], "after_demand": after["demand_units"],
        "delta_demand": after["demand_units"] - before["demand_units"],
        "before_profit": before["gross_profit_eur"], "after_profit": after["gross_profit_eur"],
        "delta_profit": after["gross_profit_eur"] - before["gross_profit_eur"],
        "before_status": before["status"], "after_status": after["status"],
        "before_stockout": before["stockout_risk_pct"], "after_stockout": after["stockout_risk_pct"],
        "before_waste": before["waste_cost_eur"], "after_waste": after["waste_cost_eur"],
    }

# -----------------------------------------------------------------------------
# Case-based decision support (not causal or autonomous learning)
# -----------------------------------------------------------------------------
def similarity_score(new_event: dict[str, Any], old_event: dict[str, Any]) -> float:
    score = 0.0
    if new_event.get("category") == old_event.get("category"):
        score += .35
    new_vars = set(new_event.get("variables", []))
    old_vars = set(old_event.get("variables", []))
    union = new_vars | old_vars
    if union:
        score += .40 * len(new_vars & old_vars) / len(union)
    if new_event.get("template") == old_event.get("template"):
        score += .25
    return score


def find_relevant_cases(event: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    cases = []
    for old in st.session_state.decision_history:
        if old.get("event_id") == event.get("id"):
            continue
        if old.get("decision") not in ("approved", "rejected", "deferred"):
            continue
        score = similarity_score(event, old)
        if score >= .22:
            cases.append({**old, "similarity": score})
    return sorted(cases, key=lambda x: x["similarity"], reverse=True)[:limit]


def record_decision(event: dict[str, Any], decision: str, note: str = "") -> None:
    event["status"] = decision
    event["decision_at"] = now_iso()
    event["decision_note"] = note.strip()
    if decision == "approved":
        if event["id"] not in st.session_state.approved_event_ids:
            st.session_state.approved_event_ids.append(event["id"])
    record = {
        "event_id": event["id"], "created_at": event.get("created_at"),
        "decision_at": event["decision_at"], "template": event["template"],
        "category": event["category"], "variables": event["variables"],
        "decision": decision, "note": note.strip(), "severity": event.get("severity", "Info"),
        "ai_recommendation": event.get("ai_recommendation", "approve"),
        "delta_demand_units_at_review": event.get("delta_demand_units", 0),
        "delta_gross_profit_eur_at_review": event.get("delta_gross_profit_eur", 0),
    }
    st.session_state.decision_history.insert(0, record)
    persist_decision(record)


def style_status_table(df: pd.DataFrame):
    """Apply readable theme-aware table text while preserving status highlighting."""
    theme_base = str(st.get_option("theme.base") or "light").lower()
    dark_mode = theme_base == "dark"
    text_color = "#ffffff" if dark_mode else "#111827"
    header_bg = "#262730" if dark_mode else "#e5e7eb"
    header_color = "#ffffff" if dark_mode else "#111827"
    border_color = "#555862" if dark_mode else "#9ca3af"

    def color_status(value: Any) -> str:
        if value == "Adequate":
            return "background-color: #d9ead3; color: #173b1a; font-weight: 700"
        if value == "Warning":
            return "background-color: #fff2cc; color: #5c4600; font-weight: 700"
        if value == "Critical":
            return "background-color: #f4cccc; color: #6b0000; font-weight: 700"
        return f"color: {text_color}"

    styler = df.style.set_properties(**{"color": text_color, "background-color": "#111318" if dark_mode else "#ffffff"}).set_table_styles([
        {"selector": "th", "props": [("background-color", header_bg), ("color", header_color), ("font-weight", "bold"), ("border", f"1px solid {border_color}")]},
        {"selector": "td", "props": [("color", text_color), ("border", f"1px solid {border_color}")]},
    ])
    if "Status" in df.columns:
        try:
            styler = styler.map(color_status, subset=["Status"])
        except Exception:
            styler = styler.applymap(color_status, subset=["Status"])
    return styler


def render_sound_control(pending_count: int) -> None:
    """Attempt sound automatically for pending alerts, with a silent first-interaction fallback.

    Browsers may block autoplay. If blocked, the first ordinary user interaction arms
    the alert sound; no visible sound-enable button is shown.
    """
    if int(pending_count) <= 0:
        return
    components.html(f"""
    <script>
    (() => {{
      const pending = {int(pending_count)};
      let hasPlayed = false;
      let armed = false;
      async function playAlert() {{
        if (hasPlayed || !pending) return;
        try {{
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          if (!AudioCtx) return;
          const ctx = new AudioCtx();
          if (ctx.state === 'suspended') await ctx.resume();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.value = 880;
          gain.gain.value = 0.07;
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.22);
          hasPlayed = true;
          setTimeout(() => ctx.close(), 350);
        }} catch (e) {{ /* Browser autoplay policy may require a user gesture. */ }}
      }}
      function armOnInteraction() {{
        if (armed || hasPlayed) return;
        armed = true;
        window.addEventListener('pointerdown', playAlert, {{ once: true }});
        window.addEventListener('keydown', playAlert, {{ once: true }});
      }}
      playAlert().then(() => {{ if (!hasPlayed) armOnInteraction(); }});
      setTimeout(() => {{ if (!hasPlayed) armOnInteraction(); }}, 250);
    }})();
    </script>
    """, height=0)


def fmt_eur(value: float) -> str:
    return f"€{value:,.2f}"


def fmt_num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"

# -----------------------------------------------------------------------------
# Header, controls, and alerts
# -----------------------------------------------------------------------------
if st.session_state.get("retail_brain_page", "📊 9-agent dashboard") != "📊 9-agent dashboard":
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.title(st.session_state.get("retail_brain_page", "Retail Brain").split(" ", 1)[-1])
        st.caption("Store operations workspace · Simulated decision support")
    with header_right:
        st.markdown("<div style='text-align:right;padding-top:1.1rem;color:var(--rb-muted);font-size:.82rem;font-weight:600'>● WORKSPACE ONLINE</div>", unsafe_allow_html=True)

PAGE_OPTIONS = [
    "📊 9-agent dashboard",
    "🚨 Decision center",
    "📈 Decisions effects",
    "🧠 Decision learning & history",
    "🧪 Trial & error lab",
    "💬 AI manager assistant",
    "🔬 Model explainability",
]

with st.sidebar:
    st.markdown("<div style='padding:8px 4px 16px;font-size:1.35rem;font-weight:800;color:var(--text-color)'>◈ Retail Brain</div>", unsafe_allow_html=True)
    if "retail_brain_page" not in st.session_state or st.session_state.retail_brain_page not in PAGE_OPTIONS:
        st.session_state.retail_brain_page = PAGE_OPTIONS[0]
    st.caption("WORKSPACE")
    for page_label in PAGE_OPTIONS:
        is_active = st.session_state.retail_brain_page == page_label
        if st.button(
            page_label,
            key=f"nav_{PAGE_OPTIONS.index(page_label)}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.retail_brain_page = page_label
            st.rerun()
    selected_page = st.session_state.retail_brain_page
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.divider()
    st.caption("STORE OPERATIONS WORKSPACE")
    st.write("**Simulation mode:** simulated operations · local SQLite decision audit")
    st.session_state.auto_monitor = st.toggle(
        "Auto-scan every 2 minutes",
        value=st.session_state.auto_monitor,
        help="Simulation only. The app session must remain open; SQLite decision audit persistence depends on the host filesystem.",
    )
    if st.session_state.auto_monitor:
        st.caption("Auto-scan is active for this session.")
    if st.button("🔎 Scan for operational scenarios", use_container_width=True, type="primary"):
        scan_scenarios(3)
        st.rerun()
    if st.button("➕ Generate one new alert", use_container_width=True):
        name = random.choice(list(SCENARIO_TEMPLATES))
        category = random.choice(list(CATEGORIES))
        event = create_scenario(name, category, "Manual alert")
        st.session_state.events.insert(0, event)
        st.rerun()

# A bounded, rerun-based monitor avoids a dependency on newer Streamlit fragment APIs.
if st.session_state.auto_monitor:
    try:
        last = st.session_state.last_auto_event_at
        elapsed = (datetime.now() - datetime.fromisoformat(last.replace("Z", "+00:00")).replace(tzinfo=None)).total_seconds() if last else 9999
    except Exception:
        elapsed = 9999
    if elapsed >= 120:
        name = random.choice(list(SCENARIO_TEMPLATES))
        category = random.choice(list(CATEGORIES))
        st.session_state.events.insert(0, create_scenario(name, category, "Auto-monitor"))
        st.session_state.last_auto_event_at = now_iso()

pending_events = [e for e in st.session_state.events if e.get("status") == "pending"]
critical_count = sum(e.get("severity") == "Critical" for e in pending_events)
if pending_events and selected_page != "📊 9-agent dashboard":
    if critical_count:
        st.error(f"🚨 **Operational alert:** {len(pending_events)} scenario(s) need manager review, including {critical_count} critical alert(s). Review decisions are simulated and require manager input.")
    else:
        st.warning(f"🔔 **Manager review required:** {len(pending_events)} scenario(s) are pending approval, rejection, or deferral.")
elif selected_page != "📊 9-agent dashboard":
    st.success("No pending simulated alerts. Use **Scan for operational scenarios** to generate a new review queue.")
render_sound_control(len(pending_events))


# -----------------------------------------------------------------------------
# Review dialog
# -----------------------------------------------------------------------------
@st.dialog("Review operational scenario", width="large")
def review_dialog(event_id: str) -> None:
    event = next((e for e in st.session_state.events if e["id"] == event_id), None)
    if not event:
        st.warning("This scenario is no longer available in the current session.")
        return
    st.subheader(f"{event['template']} — {event['category']}")
    st.caption(f"Created {event['created_at']} · Source: {event.get('source', 'Unknown')} · Severity: {event.get('severity', 'Info')}")
    st.write(event["description"])
    st.markdown(f"**Scenario changes:** {event['change_summary']}")
    _, _, before, after = scenario_metrics(event["category"], event["changes"])
    metrics = [
        ("Estimated daily demand", before["demand_units"], after["demand_units"], "units"),
        ("Estimated gross profit", before["gross_profit_eur"], after["gross_profit_eur"], "eur"),
        ("Stockout risk", before["stockout_risk_pct"], after["stockout_risk_pct"], "pct"),
        ("Waste cost proxy", before["waste_cost_eur"], after["waste_cost_eur"], "eur"),
    ]
    compare = []
    for label, b, a, kind in metrics:
        if kind == "eur":
            bs, ass, delta = fmt_eur(float(b)), fmt_eur(float(a)), fmt_eur(float(a)-float(b))
        elif kind == "pct":
            bs, ass, delta = f"{float(b):.1f}%", f"{float(a):.1f}%", f"{float(a)-float(b):+.1f} pp"
        else:
            bs, ass, delta = fmt_num(float(b)), fmt_num(float(a)), f"{float(a)-float(b):+.1f}"
        compare.append({"Metric": label, "Before": bs, "After if applied": ass, "Change": delta})
    st.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)
    st.caption("These are model-based projections against the current simulated profile, not measured outcomes or causal estimates.")
    with st.expander("Why does the model estimate this demand? (SHAP explanation)", expanded=True):
        st.write("The chart explains the fitted demand model for the scenario profile at the point of review.")
        render_shap_waterfall(apply_changes_to_profile(current_profile(event["category"]), event["changes"]), key=f"review_{event_id}")

    cases = find_relevant_cases(event)
    st.markdown("#### Relevant historical decisions")
    if cases:
        case_rows = []
        for case in cases:
            case_rows.append({
                "Similarity": f"{case['similarity']:.0%}", "Previous scenario": case.get("template", "—"),
                "Category": case.get("category", "—"), "Decision": case.get("decision", "—").title(),
                "Recorded note": case.get("note") or "No manager note recorded",
                "Projected profit delta at review": fmt_eur(float(case.get("delta_gross_profit_eur_at_review", 0))),
            })
        st.dataframe(pd.DataFrame(case_rows), use_container_width=True, hide_index=True)
        st.caption("Case matching uses category, overlapping changed variables, and scenario type. Similarity is a reference aid, not an ML-trained recommendation.")
    else:
        st.info("No sufficiently similar prior decisions are recorded in this session yet.")

    note = st.text_area("Manager rationale (optional)", key=f"note_{event_id}", placeholder="Why is this decision appropriate? What should be monitored?")
    a, b, c = st.columns(3)
    if a.button("✅ Approve scenario", use_container_width=True, type="primary", key=f"approve_{event_id}"):
        record_decision(event, "approved", note)
        st.success("Approved in the simulated session state. Dashboard profile metrics will recalculate.")
        st.rerun()
    if b.button("❌ Reject scenario", use_container_width=True, key=f"reject_{event_id}"):
        record_decision(event, "rejected", note)
        st.rerun()
    if c.button("⏸ Defer decision", use_container_width=True, key=f"defer_{event_id}"):
        # Deferral is recorded in history but returns the scenario to pending review.
        record_decision(event, "deferred", note)
        event["status"] = "pending"
        st.rerun()

# -----------------------------------------------------------------------------
# Main workspace page selected from the left navigation
# -----------------------------------------------------------------------------
if selected_page == "🧪 Trial & error lab":
    st.subheader("Trial & error lab — test decisions before acting")
    st.write("Change any operating parameter and compare the projected outcome with the current simulated profile. Nothing in this lab is approved or applied to the dashboard; it is a what-if sandbox.")
    st.warning("Exploratory simulation only. Results are model-based estimates, not validated causal effects or a guarantee of real-world outcomes.")

    sandbox_category = st.selectbox("Category to simulate", list(CATEGORIES), key="sandbox_category")
    baseline = current_profile(sandbox_category)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Commercial parameters")
        sim_price = st.slider("Own selling price (€)", 0.10, 15.00, float(np.clip(baseline["price"], .10, 15)), .05, key=f"sandbox_price_{sandbox_category}")
        sim_comp_price = st.slider("Competitor price (€)", 0.10, 15.00, float(np.clip(baseline["competitor_price"], .10, 15)), .05, key=f"sandbox_comp_price_{sandbox_category}")
        sim_promo = st.slider("Promotion intensity (%)", 0, 70, int(np.clip(baseline["promo_pct"], 0, 70)), 1, key=f"sandbox_promo_{sandbox_category}")
        sim_display = st.slider("Display quality (%)", 0, 100, int(np.clip(baseline["display_pct"], 0, 100)), 1, key=f"sandbox_display_{sandbox_category}")
        sim_traffic = st.slider("Store traffic index", 20, 250, int(np.clip(baseline["traffic_idx"], 20, 250)), 1, key=f"sandbox_traffic_{sandbox_category}")
        sim_temp = st.slider("Temperature (°C)", -10, 45, int(np.clip(baseline["temperature"], -10, 45)), 1, key=f"sandbox_temp_{sandbox_category}")
    with right:
        st.markdown("#### Supply, inventory and operating parameters")
        sim_stock = st.slider("Available stock (units)", 0, 2000, int(np.clip(baseline["stock_units"], 0, 2000)), 5, key=f"sandbox_stock_{sandbox_category}")
        sim_lead = st.slider("Supplier lead time (days)", 0, 21, int(np.clip(baseline["lead_time_days"], 0, 21)), 1, key=f"sandbox_lead_{sandbox_category}")
        sim_expiry = st.slider("Days until expiry / freshness horizon", 0, 365, int(np.clip(baseline["expiry_days"], 0, 365)), 1, key=f"sandbox_expiry_{sandbox_category}")
        sim_staff = st.slider("Staffing units", 0, 20, int(np.clip(baseline["staffing_units"], 0, 20)), 1, key=f"sandbox_staff_{sandbox_category}")
        sim_cost = st.slider("Unit purchase cost (€)", 0.05, 12.00, float(np.clip(baseline["unit_cost"], .05, 12)), .05, key=f"sandbox_cost_{sandbox_category}")
        sim_waste = st.slider("Baseline waste rate (%)", 0.0, 50.0, float(np.clip(baseline.get("waste_rate", .02)*100, 0, 50)), .5, key=f"sandbox_waste_{sandbox_category}")
        sim_compliance = st.checkbox("Simulate an active compliance issue", value=bool(baseline.get("compliance_issue", False)), key=f"sandbox_compliance_{sandbox_category}")

    simulated = dict(baseline)
    simulated.update({"price": sim_price, "competitor_price": sim_comp_price, "promo_pct": sim_promo,
                      "display_pct": sim_display, "traffic_idx": sim_traffic, "temperature": sim_temp,
                      "stock_units": sim_stock, "lead_time_days": sim_lead, "expiry_days": sim_expiry,
                      "staffing_units": sim_staff, "unit_cost": sim_cost, "waste_rate": sim_waste/100,
                      "compliance_issue": sim_compliance})
    before = calculate_metrics(sandbox_category, baseline)
    after = calculate_metrics(sandbox_category, simulated)
    delta_demand = after["demand_units"] - before["demand_units"]
    delta_profit = after["gross_profit_eur"] - before["gross_profit_eur"]
    delta_revenue = after["revenue_eur"] - before["revenue_eur"]
    delta_waste = after["waste_cost_eur"] - before["waste_cost_eur"]
    price_position_before = (baseline["price"] / max(baseline["competitor_price"], .01) - 1) * 100
    price_position_after = (simulated["price"] / max(simulated["competitor_price"], .01) - 1) * 100
    supply_gap_before = max(0, before["demand_units"] * baseline["lead_time_days"] - baseline["stock_units"])
    supply_gap_after = max(0, after["demand_units"] * simulated["lead_time_days"] - simulated["stock_units"])

    st.markdown("#### Projected outcome vs current profile")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated demand (units/day)", f"{after['demand_units']:.1f}", f"{delta_demand:+.1f} vs current")
    k2.metric("Gross profit (€/day)", fmt_eur(after["gross_profit_eur"]), f"{delta_profit:+.2f} € vs current")
    k3.metric("Revenue (€/day)", fmt_eur(after["revenue_eur"]), f"{delta_revenue:+.2f} € vs current")
    k4.metric("Stockout risk", f"{after['stockout_risk_pct']:.1f}%", f"{after['stockout_risk_pct']-before['stockout_risk_pct']:+.1f} pp vs current")

    compare = pd.DataFrame([
        {"Indicator": "Demand (units/day)", "Current profile": before["demand_units"], "Trial scenario": after["demand_units"], "Change": delta_demand},
        {"Indicator": "Sales fulfilled (units/day)", "Current profile": before["sales_units"], "Trial scenario": after["sales_units"], "Change": after["sales_units"]-before["sales_units"]},
        {"Indicator": "Lost sales exposure (units/day)", "Current profile": before["lost_sales_units"], "Trial scenario": after["lost_sales_units"], "Change": after["lost_sales_units"]-before["lost_sales_units"]},
        {"Indicator": "Gross profit (€/day)", "Current profile": before["gross_profit_eur"], "Trial scenario": after["gross_profit_eur"], "Change": delta_profit},
        {"Indicator": "Gross margin (%)", "Current profile": before["margin_pct"], "Trial scenario": after["margin_pct"], "Change": after["margin_pct"]-before["margin_pct"]},
        {"Indicator": "Stock cover (days)", "Current profile": before["stock_cover_days"], "Trial scenario": after["stock_cover_days"], "Change": after["stock_cover_days"]-before["stock_cover_days"]},
        {"Indicator": "Stockout risk (%)", "Current profile": before["stockout_risk_pct"], "Trial scenario": after["stockout_risk_pct"], "Change": after["stockout_risk_pct"]-before["stockout_risk_pct"]},
        {"Indicator": "Waste cost proxy (€/day)", "Current profile": before["waste_cost_eur"], "Trial scenario": after["waste_cost_eur"], "Change": delta_waste},
        {"Indicator": "Staffing pressure index", "Current profile": before["staffing_pressure"], "Trial scenario": after["staffing_pressure"], "Change": after["staffing_pressure"]-before["staffing_pressure"]},
    ])
    st.dataframe(compare.round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Agent feedback: what may happen")
    feedback = []
    if sim_price > baseline["price"] + .001:
        feedback.append(("Pricing & demand", f"You raised own price by {((sim_price/baseline['price'])-1)*100:+.1f}%. The demand estimator projects {delta_demand:+.1f} units/day. Higher unit margin may be offset by lower volume; projected gross profit changes by {delta_profit:+.2f} €/day."))
    elif sim_price < baseline["price"] - .001:
        feedback.append(("Pricing & demand", f"You reduced own price by {((sim_price/baseline['price'])-1)*100:+.1f}%. The model projects {delta_demand:+.1f} units/day of demand change. Check whether added volume compensates for lower margin per unit."))
    else:
        feedback.append(("Pricing & demand", f"Own price is unchanged. Overall demand changes by {delta_demand:+.1f} units/day due to the other inputs."))
    if price_position_after < price_position_before:
        feedback.append(("Competition / market position", f"Your price position improved by {(price_position_before-price_position_after):.1f} percentage points relative to the competitor price. This is a price-position proxy, not a measured market-share forecast."))
    elif price_position_after > price_position_before:
        feedback.append(("Competition / market position", f"Your price is less competitive relative to the competitor by {(price_position_after-price_position_before):.1f} percentage points. The prototype does not have competitor demand data, so market share itself cannot be reliably calculated."))
    feedback.append(("Inventory & availability", f"Estimated stockout risk moves from {before['stockout_risk_pct']:.1f}% to {after['stockout_risk_pct']:.1f}%. Approximate demand over supplier lead time not covered by stock changes from {supply_gap_before:.1f} to {supply_gap_after:.1f} units."))
    if sim_lead > baseline["lead_time_days"]:
        feedback.append(("Supply chain", f"Lead time increases by {sim_lead-baseline['lead_time_days']} day(s). That raises exposure to delayed replenishment; review supplier reliability and safety stock before acting."))
    elif sim_lead < baseline["lead_time_days"]:
        feedback.append(("Supply chain", f"Lead time decreases by {baseline['lead_time_days']-sim_lead} day(s), which can reduce replenishment exposure if the assumption is operationally achievable."))
    if delta_waste > .01:
        feedback.append(("Freshness & waste", f"The waste-cost proxy increases by {delta_waste:+.2f} €/day. Review expiry horizon, stock quantity, and promotion markdown options."))
    elif delta_waste < -.01:
        feedback.append(("Freshness & waste", f"The waste-cost proxy decreases by {delta_waste:.2f} €/day under these assumptions."))
    if after["staffing_pressure"] > before["staffing_pressure"] + .01:
        feedback.append(("Workforce", f"Staffing pressure increases from {before['staffing_pressure']:.2f} to {after['staffing_pressure']:.2f}. A demand uplift without sufficient coverage may increase execution risk."))
    elif after["staffing_pressure"] < before["staffing_pressure"] - .01:
        feedback.append(("Workforce", f"Staffing pressure decreases from {before['staffing_pressure']:.2f} to {after['staffing_pressure']:.2f} under the simulated staffing and demand inputs."))
    if after["compliance_risk"]:
        feedback.append(("Compliance", "A compliance risk flag is active. Treat this as a review trigger, not an automated compliance determination."))
    if after["status"] != before["status"]:
        feedback.append(("Overall risk", f"Status changes from {before['status']} to {after['status']}. Review the underlying risk indicators before deciding."))
    for title, message in feedback:
        st.markdown(f"**{title}:** {message}")
    st.caption("Market share is not directly modeled. Competitive feedback uses relative price position only. Supply-chain coverage and waste are simplified proxies. All projections are synthetic and intended for experimentation, not live operating decisions.")

    with st.expander("View all simulated inputs"):
        input_compare = pd.DataFrame([{"Parameter": k, "Current profile": baseline.get(k), "Trial scenario": simulated.get(k)} for k in ["price", "competitor_price", "promo_pct", "display_pct", "traffic_idx", "temperature", "stock_units", "lead_time_days", "expiry_days", "staffing_units", "unit_cost", "waste_rate", "compliance_issue"]])
        st.dataframe(input_compare, use_container_width=True, hide_index=True)

if selected_page == "📊 9-agent dashboard":
    df = metrics_frame()
    total_revenue = float(df["revenue_eur"].sum())
    total_profit = float(df["gross_profit_eur"].sum())
    total_sales = float(df["sales_units"].sum())
    mean_stockout = float(df["stockout_risk_pct"].mean())
    mean_margin = float(df["margin_pct"].mean())

    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
          <div>
            <div style="font-size:1.75rem;font-weight:800;letter-spacing:-.04em;color:var(--text-color)">👋 Good morning, Retail Manager</div>
            <div style="font-size:.94rem;color:var(--rb-muted);margin-top:.25rem">Here's what's happening across your store operations today.</div>
          </div>
          <div style="padding:.55rem .8rem;border:1px solid rgba(98,84,199,.18);border-radius:12px;background:var(--secondary-background-color);font-size:.82rem;font-weight:650">● Simulation workspace online</div>
        </div>
        """, unsafe_allow_html=True,
    )

    kpi_cols = st.columns(5)
    kpis = [
        ("Projected Revenue", fmt_eur(total_revenue), "Across monitored categories", "💜"),
        ("Estimated Units Sold", f"{total_sales:,.0f}", "Units/day · simulated", "🟠"),
        ("Projected Gross Profit", fmt_eur(total_profit), f"Average margin {mean_margin:.1f}%", "🟢"),
        ("Average Stockout Risk", f"{mean_stockout:.1f}%", "Across monitored categories", "🔴"),
        ("Pending Decisions", f"{len(pending_events)}", f"{critical_count} critical alert(s)", "🔵"),
    ]
    for col, (label, value, note, emoji) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
                <div style="min-height:118px;padding:1rem;border:1px solid rgba(120,120,150,.16);border-radius:15px;background:var(--secondary-background-color);box-shadow:0 4px 16px rgba(40,35,90,.045)">
                  <div style="display:flex;align-items:center;gap:.5rem;font-size:.82rem;color:var(--rb-muted);font-weight:650"><span style="font-size:1.15rem">{emoji}</span>{label}</div>
                  <div style="font-size:1.55rem;font-weight:800;letter-spacing:-.035em;color:var(--rb-purple);margin:.45rem 0 .25rem">{value}</div>
                  <div style="font-size:.75rem;color:var(--rb-muted)">{note}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

    # ROW 1: agents left, category performance middle, recommendations right.
    agents_col, category_col, insights_col = st.columns([1.05, 1.35, 1.15], gap="medium")

    agent_items = [
        ("🧠", "Demand & XAI", "Demand prediction and feature contributions"),
        ("📦", "Inventory safety", "Stock coverage and stockout risk"),
        ("🏷️", "Commercial intelligence", "Own-price vs competitor-price position"),
        ("🚚", "Replenishment", "Stock, lead time and supply coverage"),
        ("🌿", "Freshness & waste", "Expiry horizon and waste-cost proxy"),
        ("👥", "Workforce planning", "Demand relative to staffing capacity"),
        ("🎯", "Promotion analysis", "Promotion and display settings"),
        ("💶", "Profitability", "Revenue, margin and gross profit"),
        ("🛡️", "Compliance & tasks", "Compliance and operational risk flags"),
    ]
    if "dashboard_selected_agent" not in st.session_state:
        st.session_state.dashboard_selected_agent = "Demand & XAI"

    with agents_col:
        st.markdown("### 9 AI agents")
        st.caption("Select an agent to inspect its live simulated signals.")
        for i in range(0, len(agent_items), 3):
            agent_button_cols = st.columns(3, gap="small")
            for j, (icon, title, description) in enumerate(agent_items[i:i+3]):
                with agent_button_cols[j]:
                    is_selected = st.session_state.dashboard_selected_agent == title
                    if st.button(
                        icon, key=f"agent_icon_{i+j}", help=f"{title}: {description}",
                        use_container_width=True, type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.dashboard_selected_agent = title
                        st.rerun()
                    st.markdown(
                        f"<div style='text-align:center;font-size:.66rem;line-height:1.15;font-weight:700;margin:-.35rem 0 .5rem'>{title}</div>",
                        unsafe_allow_html=True,
                    )

        selected_agent = st.session_state.dashboard_selected_agent
        st.markdown(
            f"<div style='padding:.7rem;border:1px solid rgba(102,86,207,.25);border-radius:12px;background:var(--secondary-background-color)'><div style='font-weight:800;font-size:.86rem'>{selected_agent}</div><div style='font-size:.72rem;color:var(--rb-muted);margin-top:.2rem'>{next((x[2] for x in agent_items if x[1] == selected_agent), '')}</div></div>",
            unsafe_allow_html=True,
        )
        selected_row = None
        selected_profile = None
        if selected_agent == "Demand & XAI":
            selected_row = df.sort_values("demand_units", ascending=False).iloc[0]
            st.metric("Highest estimated demand", f"{selected_row['category']}", f"{float(selected_row['demand_units']):,.1f} units/day")
            st.caption("For model contributions, open Model Explainability.")
        elif selected_agent == "Inventory safety":
            selected_row = df.sort_values("stockout_risk_pct", ascending=False).iloc[0]
            st.metric("Highest stockout risk", f"{float(selected_row['stockout_risk_pct']):.1f}%", str(selected_row["category"]))
            st.metric("Stock cover", f"{float(selected_row['stock_cover_days']):.1f} days")
        elif selected_agent == "Commercial intelligence":
            selected_row = df.iloc[0]
            prof = current_profile(str(selected_row["category"]))
            st.metric(f"{selected_row['category']} price gap", fmt_eur(float(prof["price"])-float(prof["competitor_price"])), "Own price minus competitor price")
            st.caption("Positive = own price is higher than the competitor price.")
        elif selected_agent == "Replenishment":
            selected_row = df.sort_values("stock_cover_days", ascending=True).iloc[0]
            st.metric("Lowest stock cover", f"{float(selected_row['stock_cover_days']):.1f} days", str(selected_row["category"]))
            st.metric("Lost sales estimate", f"{float(selected_row['lost_sales_units']):,.1f} units/day")
        elif selected_agent == "Freshness & waste":
            selected_row = df.sort_values("waste_cost_eur", ascending=False).iloc[0]
            st.metric("Highest waste-cost proxy", fmt_eur(float(selected_row["waste_cost_eur"])), str(selected_row["category"]))
        elif selected_agent == "Workforce planning":
            selected_row = df.sort_values("staffing_pressure", ascending=False).iloc[0]
            st.metric("Highest staffing pressure", f"{float(selected_row['staffing_pressure']):.2f}", str(selected_row["category"]))
            st.caption("Pressure is a simulated index, not a staffing recommendation.")
        elif selected_agent == "Promotion analysis":
            promo_rows = []
            for cat in CATEGORIES:
                prof = current_profile(cat)
                promo_rows.append((cat, float(prof.get("promo_pct", 0)), float(prof.get("display_pct", 0))))
            active_promo = [x for x in promo_rows if x[1] > 0]
            st.metric("Categories with active promotion", f"{len(active_promo)}/{len(promo_rows)}")
            st.caption("Promo and display values are simulated profile inputs.")
        elif selected_agent == "Profitability":
            selected_row = df.sort_values("gross_profit_eur", ascending=False).iloc[0]
            st.metric("Top gross profit", fmt_eur(float(selected_row["gross_profit_eur"])), str(selected_row["category"]))
            st.metric("Margin", f"{float(selected_row['margin_pct']):.1f}%")
        else:
            selected_row = df[df["status"].astype(str).str.lower().isin(["critical", "warning"])].head(1)
            selected_row = selected_row.iloc[0] if not selected_row.empty else df.iloc[0]
            selected_profile = current_profile(str(selected_row["category"]))
            st.metric("Risk status", str(selected_row["status"]), str(selected_row["category"]))
            st.metric("Compliance flag", "Open" if selected_profile.get("compliance_issue", False) else "No flag", "Simulated input")

    with category_col:
        st.markdown("### Category performance")
        st.caption("Revenue, margin and operating status")
        for _, row in df.iterrows():
            category = str(row["category"])
            revenue = float(row["revenue_eur"])
            profit = float(row["gross_profit_eur"])
            margin = float(row["margin_pct"])
            status = str(row["status"])
            max_rev = max(float(df["revenue_eur"].max()), 1.0)
            bar_width = max(5, int(100 * revenue / max_rev))
            status_color = "#dc5656" if status.lower() == "critical" else "#ed9b2d" if status.lower() in ("warning", "high") else "#28a66a"
            bar_color = "#ff8a24" if category == "Birra" else "#6656cf" if category == "Prodotti da Forno" else "#28a66a" if category == "Ortofrutta" else "#5687f5"
            st.markdown(f"""
                <div style="padding:.68rem 0;border-bottom:1px solid rgba(120,120,150,.13)">
                  <div style="display:flex;justify-content:space-between;gap:.4rem;font-size:.82rem;font-weight:700"><span>{category}</span><span style="color:var(--rb-muted)">{margin:.1f}% margin</span></div>
                  <div style="font-size:.8rem;color:var(--rb-muted);margin:.18rem 0 .4rem">{fmt_eur(revenue)} revenue · {fmt_eur(profit)} profit/day</div>
                  <div style="height:6px;border-radius:99px;background:rgba(120,120,150,.14);overflow:hidden"><div style="height:6px;width:{bar_width}%;border-radius:99px;background:{bar_color}"></div></div>
                  <div style="font-size:.7rem;color:{status_color};margin-top:.3rem;font-weight:700">● {status}</div>
                </div>""", unsafe_allow_html=True)

    with insights_col:
        st.markdown("### Top insights & recommendations")
        st.caption("Prioritized signals from current simulated metrics")
        worst_stock = df.sort_values("stockout_risk_pct", ascending=False).iloc[0]
        best_profit = df.sort_values("gross_profit_eur", ascending=False).iloc[0]
        most_waste = df.sort_values("waste_cost_eur", ascending=False).iloc[0]
        insights = [
            ("⚠️", f"{worst_stock['category']} has the highest stockout risk ({float(worst_stock['stockout_risk_pct']):.1f}%).", "Review stock and lead-time assumptions."),
            ("↗️", f"{best_profit['category']} leads projected daily gross profit.", f"Estimated at {fmt_eur(float(best_profit['gross_profit_eur']))} per day."),
            ("♻️", f"{most_waste['category']} has the highest waste-cost proxy.", "Review freshness horizon and waste assumptions."),
        ]
        for emoji, headline, detail in insights:
            st.markdown(f"""
                <div style="display:flex;gap:.55rem;padding:.72rem .65rem;margin:.45rem 0;border:1px solid rgba(120,120,150,.14);border-radius:12px;background:var(--secondary-background-color)">
                  <div style="font-size:1.15rem">{emoji}</div><div><div style="font-size:.8rem;font-weight:700">{headline}</div><div style="font-size:.72rem;color:var(--rb-muted);margin-top:.25rem">{detail}</div></div>
                </div>""", unsafe_allow_html=True)

    # ROW 2: demand forecast left, sales overview middle, assistant right.
    forecast_col, sales_col, assistant_col = st.columns([1.05, 1.35, 1.15], gap="medium")
    with forecast_col:
        st.markdown("### Demand forecast")
        st.caption("Model estimate by category · not a time-series forecast")
        forecast_df = df[["category", "demand_units"]].rename(columns={"category": "Category", "demand_units": "Estimated units/day"})
        try:
            import altair as alt
            forecast = alt.Chart(forecast_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X("Category:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0, labelLimit=100)),
                y=alt.Y("Estimated units/day:Q", title="Units / day"),
                color=alt.Color("Category:N", scale=alt.Scale(
                    domain=["Birra", "Prodotti da Forno", "Ortofrutta", "Detersivi"],
                    range=["#ff8a24", "#6656cf", "#28a66a", "#5687f5"]
                ), legend=None),
                tooltip=["Category", alt.Tooltip("Estimated units/day:Q", format=",.1f")],
            ).properties(height=245)
            st.altair_chart(forecast, use_container_width=True)
        except Exception:
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    with sales_col:
        st.markdown("### Sales & demand overview")
        st.caption("Estimated demand compared with stock-constrained sales")
        trend_df = df[["category", "demand_units", "sales_units"]].melt(id_vars="category", var_name="Metric", value_name="Units per day")
        trend_df["Metric"] = trend_df["Metric"].map({"demand_units": "Estimated demand", "sales_units": "Fulfillable sales"})
        try:
            import altair as alt
            trend = alt.Chart(trend_df).mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5).encode(
                x=alt.X("category:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=120)),
                y=alt.Y("Units per day:Q", title="Units / day", scale=alt.Scale(zero=True)),
                color=alt.Color("Metric:N", title=None, scale=alt.Scale(domain=["Estimated demand", "Fulfillable sales"], range=["#ff8a24", "#6656cf"])),
                tooltip=["category", "Metric", alt.Tooltip("Units per day:Q", format=",.1f")],
            ).properties(height=245)
            st.altair_chart(trend, use_container_width=True)
        except Exception:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

    with assistant_col:
        st.markdown("### 💜 AI Manager Assistant")
        st.caption("Ask about a decision, scenario or what-if.")
        assistant_question = st.text_area(
            "Ask the Retail Brain assistant",
            placeholder="e.g. What if we increase Birra price by 10%?",
            label_visibility="collapsed",
            key="dashboard_assistant_question",
            height=110,
        )
        if st.button("Ask assistant →", type="primary", use_container_width=True, key="dashboard_assistant_submit"):
            if assistant_question.strip():
                st.session_state.assistant_prompt = assistant_question.strip()
                st.session_state.retail_brain_page = "💬 AI manager assistant"
                st.rerun()
            else:
                st.warning("Enter a question or scenario first.")

    st.caption("Retail Brain is a prototype using simulated data and model-based estimates. Recommendations are decision support only; no live store systems or operational controls are connected.")

if selected_page == "🚨 Decision center":
    st.subheader("Scenario queue and manager review")
    st.write("Scenarios are prospective what-if cases. Approving one applies its changes to the simulated category profile; rejecting it does not.")
    if not st.session_state.events:
        st.info("No scenarios generated yet. Use the sidebar scan button to create a queue.")
    else:
        event_rows = []
        for event in st.session_state.events:
            event_rows.append({
                "Scenario": event["template"], "Category": event["category"],
                "Severity": event["severity"], "Status": event["status"].title(),
                "Projected demand delta": round(float(event.get("delta_demand_units", 0)), 1),
                "Projected gross-profit delta (€)": round(float(event.get("delta_gross_profit_eur", 0)), 2),
                "Created": event["created_at"],
            })
        queue_df = pd.DataFrame(event_rows)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)
        pending = [e for e in st.session_state.events if e["status"] == "pending"]
        if pending:
            options = {f"{e['severity']} · {e['category']} · {e['template']} · {e['created_at']}": e["id"] for e in pending}
            selected_label = st.selectbox("Select a pending scenario to review", list(options.keys()))
            if st.button("Open manager review", type="primary"):
                review_dialog(options[selected_label])
        else:
            st.success("No pending scenarios to review.")

    st.markdown("#### Projected impact by category")
    impact_rows = []
    for category in CATEGORIES:
        profile = current_profile(category)
        m = calculate_metrics(category, profile)
        impact_rows.append({
            "Category": category, "Demand (units/day)": round(float(m["demand_units"]), 1),
            "Available stock": int(profile["stock_units"]), "Stockout risk (%)": round(float(m["stockout_risk_pct"]), 1),
            "Gross profit/day (€)": round(float(m["gross_profit_eur"]), 2),
            "Waste cost proxy (€)": round(float(m["waste_cost_eur"]), 2), "Status": m["status"],
        })
    impact_df = pd.DataFrame(impact_rows)
    st.dataframe(style_status_table(impact_df), use_container_width=True, hide_index=True)
    csv_data = queue_df.to_csv(index=False).encode("utf-8") if st.session_state.events else b"Scenario,Category,Status\n"
    st.download_button("⬇️ Download manager action queue (CSV)", data=csv_data,
                       file_name="retail_brain_manager_action_queue.csv", mime="text/csv")
    impact_csv = impact_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download projected category impact (CSV)", data=impact_csv,
                       file_name="retail_brain_projected_category_impact.csv", mime="text/csv")

if selected_page == "📈 Decisions effects":
    st.subheader("Decisions effects")
    decided_for_effects = [e for e in st.session_state.events if e.get("status") in ("approved", "rejected", "deferred")]
    if not decided_for_effects:
        st.info("No decisions have been recorded in this session yet. Review a scenario to approve, reject, or defer it and see its estimated effects here.")
    else:
        rows = []
        for event in decided_for_effects:
            impact = get_live_event_impact(event)
            rows.append({
                "Scenario": event["template"], "Category": event["category"],
                "Decision": event.get("status", "unknown").title(),
                "Demand before": round(float(impact["before_demand"]), 1),
                "Demand after": round(float(impact["after_demand"]), 1),
                "Demand change": round(float(impact["delta_demand"]), 1),
                "Gross profit before (€)": round(float(impact["before_profit"]), 2),
                "Gross profit after (€)": round(float(impact["after_profit"]), 2),
                "Gross profit change (€)": round(float(impact["delta_profit"]), 2),
                "Stockout risk before (%)": round(float(impact["before_stockout"]), 1),
                "Stockout risk after (%)": round(float(impact["after_stockout"]), 1),
                "Status before": impact["before_status"], "Status after": impact["after_status"],
            })
        effects_df = pd.DataFrame(rows)
        st.dataframe(effects_df, use_container_width=True, hide_index=True)
        st.caption("For approved decisions, the comparison estimates the change applied to the simulated profile. For rejected or deferred decisions, it shows a counterfactual estimate of the scenario's potential impact. These are current modelled comparisons, not observed real-world outcomes or causal estimates.")
        st.download_button("⬇️ Download decisions effects (CSV)", effects_df.to_csv(index=False).encode("utf-8"),
                           file_name="retail_brain_decisions_effects.csv", mime="text/csv")

    st.markdown("#### Current profile versus original baseline")
    baseline_rows = []
    for category in CATEGORIES:
        base_m = calculate_metrics(category, st.session_state.base_profiles[category])
        current_m = calculate_metrics(category, current_profile(category))
        baseline_rows.append({
            "Category": category, "Baseline demand": round(float(base_m["demand_units"]), 1),
            "Current demand": round(float(current_m["demand_units"]), 1),
            "Demand delta": round(float(current_m["demand_units"])-float(base_m["demand_units"]), 1),
            "Baseline gross profit (€)": round(float(base_m["gross_profit_eur"]), 2),
            "Current gross profit (€)": round(float(current_m["gross_profit_eur"]), 2),
            "Gross profit delta (€)": round(float(current_m["gross_profit_eur"])-float(base_m["gross_profit_eur"]), 2),
            "Current status": current_m["status"],
        })
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)

if selected_page == "🧠 Decision learning & history":
    st.subheader("Decision learning & history")
    st.markdown("#### AI retrospective assessment of previous decisions")
    st.caption("The assessment is generated automatically from the scenario's simulated before/after metrics and the recorded decision. It is not a verified real-world outcome; no manager rating is required.")
    decided_events = [e for e in st.session_state.events if e.get("status") in ("approved", "rejected", "deferred")]
    if decided_events:
        chosen = st.selectbox(
            "Select a previous decision for AI assessment",
            options=decided_events,
            format_func=lambda e: f"{e['category']} · {e['template']} · {e['status'].title()}",
            key="ai_assessment_event_select",
        )
        impact = get_live_event_impact(chosen)
        delta_profit = float(impact["delta_profit"])
        delta_demand = float(impact["delta_demand"])
        delta_stockout = float(impact["after_stockout"]) - float(impact["before_stockout"])
        delta_waste = float(impact["after_waste"]) - float(impact["before_waste"])
        decision = chosen.get("status", "deferred")
        positive = delta_profit > 0 and delta_stockout <= 0 and delta_waste <= 0
        negative = delta_profit < 0 or delta_stockout > 5 or delta_waste > max(1.0, float(impact["before_waste"]) * .15)
        if decision == "approved":
            if positive:
                assessment = "Potentially beneficial in simulation"
                rationale = "The current simulated comparison shows higher gross profit without an increase in stockout risk or waste cost. Monitor demand and availability before treating this as a successful real-world decision."
                severity = "Positive simulated signal"
            elif negative:
                assessment = "Potential downside detected in simulation"
                reasons = []
                if delta_profit < 0: reasons.append(f"gross profit changes by {delta_profit:+.2f} €/day")
                if delta_stockout > 5: reasons.append(f"stockout risk rises by {delta_stockout:+.1f} percentage points")
                if delta_waste > max(1.0, float(impact["before_waste"]) * .15): reasons.append(f"waste-cost proxy rises by {delta_waste:+.2f} €/day")
                rationale = "The model flags potential trade-offs: " + "; ".join(reasons) + ". Review the assumptions and compare with observed store data before drawing conclusions."
                severity = "Review recommended"
            else:
                assessment = "Mixed or limited simulated effect"
                rationale = "The simulated indicators do not show a clear overall benefit or material downside. The decision should be evaluated against the business objective and later operational evidence."
                severity = "Inconclusive simulated signal"
        elif decision == "rejected":
            assessment = "Rejected scenario — avoided impact estimated"
            rationale = f"Had the scenario's changes been applied, the current counterfactual estimates indicate gross profit {delta_profit:+.2f} €/day, demand {delta_demand:+.1f} units/day, stockout risk {delta_stockout:+.1f} percentage points, and waste cost {delta_waste:+.2f} €/day versus the comparison profile. This is a modelled counterfactual, not proof of what would have happened."
            severity = "Counterfactual only"
        else:
            assessment = "Deferred decision — exposure remains uncommitted"
            rationale = f"The scenario remains unapplied. Its current modelled change is gross profit {delta_profit:+.2f} €/day and demand {delta_demand:+.1f} units/day, with stockout risk changing by {delta_stockout:+.1f} percentage points. Gather the missing evidence or revisit assumptions before deciding."
            severity = "Awaiting decision"

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Estimated demand change", f"{delta_demand:+.1f} units/day")
        a2.metric("Estimated gross-profit change", f"€{delta_profit:+.2f}/day")
        a3.metric("Stockout-risk change", f"{delta_stockout:+.1f} pp")
        a4.metric("Waste-cost change", f"€{delta_waste:+.2f}/day")
        st.info(f"**AI assessment: {assessment}**\n\n**Signal:** {severity}\n\n{rationale}")
        if st.button("Save AI assessment to decision history", type="primary"):
            assessed_at = now_iso()
            chosen["ai_assessment"] = assessment
            chosen["ai_assessment_rationale"] = rationale
            chosen["ai_assessment_signal"] = severity
            chosen["ai_assessed_at"] = assessed_at
            chosen["ai_assessment_metrics"] = {
                "delta_demand_units": round(delta_demand, 2),
                "delta_gross_profit_eur": round(delta_profit, 2),
                "delta_stockout_risk_pp": round(delta_stockout, 2),
                "delta_waste_cost_eur": round(delta_waste, 2),
            }
            for record in st.session_state.decision_history:
                if record.get("event_id") == chosen["id"]:
                    record["ai_assessment"] = assessment
                    record["ai_assessment_rationale"] = rationale
                    record["ai_assessment_signal"] = severity
                    record["ai_assessed_at"] = assessed_at
                    record.update(chosen["ai_assessment_metrics"])
            st.success("AI assessment saved to this session's decision history.")
            st.rerun()
    else:
        st.caption("A scenario must have a recorded decision before the AI can assess it retrospectively.")

    st.markdown("### Decision learning and case library")
    if not st.session_state.decision_history:
        st.info("No decisions recorded yet. Review a scenario and record a decision to build a session-level reference set.")
    else:
        hist = pd.DataFrame(st.session_state.decision_history)
        m1, m2 = st.columns(2)
        m1.metric("Recorded decision records", len(hist))
        m2.metric("Unique scenarios with a recorded decision", hist["event_id"].nunique() if "event_id" in hist.columns else len(hist))
        st.markdown("#### Decision mix")
        mix = hist["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
        try:
            import altair as alt
            chart = alt.Chart(mix).mark_bar().encode(
                x=alt.X("Count:Q", title="Number of decisions"),
                y=alt.Y("Decision:N", sort="-x", title="Decision", axis=alt.Axis(labelAngle=0)),
                tooltip=["Decision", "Count"],
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.dataframe(mix, use_container_width=True, hide_index=True)
        st.markdown("#### Case library")
        cols = ["decision_at", "template", "category", "decision", "delta_demand_units_at_review", "delta_gross_profit_eur_at_review", "ai_assessment", "ai_assessment_signal", "ai_assessment_rationale", "note"]
        hist_view = hist[[c for c in cols if c in hist.columns]].copy()
        hist_view = hist_view.rename(columns={
            "decision_at": "Decision time", "template": "Scenario", "category": "Category", "decision": "Decision",
            "delta_demand_units_at_review": "Projected demand delta at review",
            "delta_gross_profit_eur_at_review": "Projected gross-profit delta (€)",
            "ai_assessment": "AI assessment", "ai_assessment_signal": "AI signal",
            "ai_assessment_rationale": "AI rationale", "note": "Manager decision rationale",
        })
        st.dataframe(hist_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export decision case library (CSV)", hist_view.to_csv(index=False).encode("utf-8"),
                           file_name="retail_brain_decision_case_library.csv", mime="text/csv")

    st.markdown("### Decision history and audit trail")
    if st.session_state.decision_history:
        audit_hist = pd.DataFrame(st.session_state.decision_history)
        st.dataframe(audit_hist, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download decision audit trail (CSV)", audit_hist.to_csv(index=False).encode("utf-8"),
                           file_name="retail_brain_decision_audit_trail.csv", mime="text/csv")
    else:
        st.info("No decisions have been recorded yet.")
    st.markdown("#### All scenarios in this session")
    if st.session_state.events:
        all_events = []
        for e in st.session_state.events:
            all_events.append({"Created": e["created_at"], "Scenario": e["template"], "Category": e["category"],
                               "Status": e["status"], "Severity": e["severity"], "Changes": e["change_summary"],
                               "Manager rationale": e.get("decision_note", ""), "AI assessment": e.get("ai_assessment", "Not assessed yet"),
                               "AI assessment rationale": e.get("ai_assessment_rationale", "")})
        all_events_df = pd.DataFrame(all_events)
        st.dataframe(all_events_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download full scenario log (CSV)", all_events_df.to_csv(index=False).encode("utf-8"),
                           file_name="retail_brain_scenario_log.csv", mime="text/csv")

if selected_page == "💬 AI manager assistant":
    st.subheader("AI manager assistant")
    st.write("Ask about the dashboard, current category performance, previous decisions, scenarios, and what-if experiments. The assistant works immediately without an API key or setup.")

    def _assistant_context() -> dict[str, Any]:
        category_summary = []
        for category in CATEGORIES:
            profile = current_profile(category)
            metrics = calculate_metrics(category, profile)
            category_summary.append({
                "category": category,
                "profile": {k: profile.get(k) for k in ["price", "competitor_price", "promo_pct", "display_pct", "traffic_idx", "temperature", "stock_units", "lead_time_days", "expiry_days", "staffing_units", "unit_cost", "waste_rate", "compliance_issue"]},
                "metrics": {k: round(float(metrics[k]), 3) if isinstance(metrics.get(k), (int, float, np.integer, np.floating)) else str(metrics.get(k)) for k in ["demand_units", "sales_units", "lost_sales_units", "revenue_eur", "gross_profit_eur", "margin_pct", "stockout_risk_pct", "waste_cost_eur", "stock_cover_days", "staffing_pressure", "status"] if k in metrics},
            })
        recent_events = [{k: e.get(k) for k in ["id", "category", "template", "status", "severity", "created_at", "changes", "change_summary", "decision_note", "ai_assessment", "ai_assessment_rationale"]} for e in st.session_state.events[:20]]
        recent_history = [{k: r.get(k) for k in ["category", "template", "status", "timestamp", "decision_at", "note", "assessment", "rationale", "ai_assessment", "ai_assessment_rationale"]} for r in st.session_state.decision_history[-20:]]
        return {"categories": category_summary, "recent_scenarios": recent_events, "recent_decision_history": recent_history,
                "scenario_count": len(st.session_state.events), "decision_history_count": len(st.session_state.decision_history),
                "data_note": "All figures are synthetic/session simulation outputs, not live retailer data."}

    def _local_manager_answer(question: str, context: dict[str, Any]) -> str:
        """Built-in, no-key assistant grounded in the app's current simulated session."""
        q = question.lower()
        cats = context["categories"]
        category = next((c for c in cats if c["category"].lower() in q), None)
        if category is None and any(term in q for term in ["worst", "risk", "critical", "stockout", "stock out"]):
            category = max(cats, key=lambda c: float(c["metrics"].get("stockout_risk_pct", 0) or 0))
        if category is None and any(term in q for term in ["best", "highest profit", "most profitable"]):
            category = max(cats, key=lambda c: float(c["metrics"].get("gross_profit_eur", 0) or 0))
        if any(term in q for term in ["previous decision", "decision history", "past decision", "what did we decide", "history"]):
            if not context["recent_decision_history"] and not context["recent_scenarios"]:
                return "There are no recorded decisions or scenarios in this session yet. Use **Decision center** to generate a scenario and record a manager decision; then ask me to review it here."
            lines = ["### Recent decision and scenario review", f"Recorded scenarios: **{context['scenario_count']}** · Decision-history records: **{context['decision_history_count']}**"]
            for e in context["recent_scenarios"][:8]:
                lines.append(f"- **{e.get('category','Category')} — {e.get('template','Scenario')}**: status **{e.get('status','unknown')}**, severity **{e.get('severity','not recorded')}**. {e.get('ai_assessment_rationale') or e.get('decision_note') or 'No assessment/rationale recorded.'}")
            lines.append("These are session records and simulated projections, not verified real-world outcomes.")
            return "\n".join(lines)
        if any(term in q for term in ["dashboard", "summarize", "summary", "overall", "all categories", "how are we doing"]):
            lines = ["### Current simulated operational snapshot", "| Category | Demand/day | Gross profit/day | Stockout risk | Waste cost/day | Status |", "|---|---:|---:|---:|---:|---|"]
            for c in cats:
                m = c["metrics"]
                lines.append(f"| {c['category']} | {float(m.get('demand_units',0)):.1f} | €{float(m.get('gross_profit_eur',0)):.2f} | {float(m.get('stockout_risk_pct',0)):.1f}% | €{float(m.get('waste_cost_eur',0)):.2f} | {m.get('status','n/a')} |")
            lines.append("\nFigures are synthetic estimates from the current app session.")
            return "\n".join(lines)
        if category:
            p, m = category["profile"], category["metrics"]
            lines = [f"### {category['category']}: current simulated position",
                     f"- Selling price: **€{float(p.get('price') or 0):.2f}** per unit; competitor price: **€{float(p.get('competitor_price') or 0):.2f}**.",
                     f"- Estimated demand: **{float(m.get('demand_units',0)):.1f} units/day**; estimated sales: **{float(m.get('sales_units',0)):.1f}**; lost sales: **{float(m.get('lost_sales_units',0)):.1f}**.",
                     f"- Revenue: **€{float(m.get('revenue_eur',0)):.2f}/day**; gross profit: **€{float(m.get('gross_profit_eur',0)):.2f}/day**; margin: **{float(m.get('margin_pct',0)):.1f}%**.",
                     f"- Stockout risk: **{float(m.get('stockout_risk_pct',0)):.1f}%**; stock cover: **{float(m.get('stock_cover_days',0)):.1f} days**; waste-cost proxy: **€{float(m.get('waste_cost_eur',0)):.2f}/day**."]
            if any(term in q for term in ["price", "increase", "raise", "higher", "decrease", "lower", "what if", "what-if", "if i"]):
                pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
                pct = float(pct_match.group(1)) if pct_match else 5.0
                if any(term in q for term in ["decrease", "lower", "reduce", "cut"]):
                    direction = -1
                else:
                    direction = 1
                trial_price = float(p.get("price") or 0) * (1 + direction * pct / 100)
                price_gap = trial_price - float(p.get("competitor_price") or 0)
                lines += [f"\n### What-if: {pct:g}% {'increase' if direction > 0 else 'decrease'} in selling price",
                          f"- Trial price: **€{trial_price:.2f}** per unit (before: €{float(p.get('price') or 0):.2f}).",
                          f"- Relative to the current competitor price, the trial price gap would be **€{price_gap:+.2f}/unit**."]
                if direction > 0:
                    lines.append("- **Gross profit trade-off:** unit margin may improve, but demand and sales volume may fall. Total gross profit rises only if the unit-margin gain outweighs lost volume.")
                    lines.append("- **Inventory/supply chain:** lower sales can reduce stockout pressure and slow replenishment needs; if demand is less price-sensitive than assumed, this may not happen.")
                else:
                    lines.append("- **Demand/competition trade-off:** a lower price may improve price competitiveness and stimulate demand, but it reduces margin per unit and may increase stockout/replenishment pressure.")
                lines.append("- The built-in assistant does not invent a precise new demand or profit number for this price-only experiment. Change the price in **Trial & error lab** to obtain the app's calculated scenario comparison.")
            else:
                lines.append(f"- Profile inputs include promotion {float(p.get('promo_pct') or 0):.1f}%, display {float(p.get('display_pct') or 0):.1f}%, stock {float(p.get('stock_units') or 0):.0f} units, supplier lead time {float(p.get('lead_time_days') or 0):.1f} days, and expiry horizon {float(p.get('expiry_days') or 0):.1f} days.")
                lines.append("Ask a follow-up such as ‘what if I increase the price by 5%?’ or use Trial & error lab for a numerical before/after calculation.")
            lines.append("\n**Caution:** these are synthetic simulation estimates, not live store data or verified causal effects.")
            return "\n".join(lines)
        if any(term in q for term in ["what if", "what-if", "increase price", "raise price", "pricing", "price increase"]):
            return ("For a useful what-if, specify a category and change, for example: **‘What if I increase Ortofrutta price by 5%?’** "
                    "I can explain the price gap and likely trade-offs now. For numerical recalculation of demand, profit, stockouts and waste, open **Trial & error lab**, select the category, change the relevant parameter, and compare the baseline with the trial.")
        if any(term in q for term in ["agent", "nine-agent", "9-agent", "what does"]):
            return ("The dashboard organizes the simulated workflow into nine operational roles: **demand forecasting, inventory risk, commercial intelligence, replenishment, freshness/waste, workforce planning, promotion/display, profitability, and compliance/task management**. "
                    "They are decision-support views in this prototype, not nine independent live AI systems. Ask about a category or metric to inspect its current simulated values.")
        return ("I can help interpret the current dashboard, category metrics, scenarios, decision history, pricing trade-offs, inventory, replenishment, waste, staffing, and competition. "
                "Try asking **‘Summarize the dashboard’**, **‘What is the stockout risk for Ortofrutta?’**, **‘Review previous decisions’**, or **‘What if I increase Birra price by 5%?’**\n\n"
                "This built-in assistant works without a key and uses the current session's synthetic data. For complex what-if calculations, use **Trial & error lab** so the app's estimator recalculates the numbers.")

    # Clear stale API-key prompts left in Streamlit session state by older app versions.
    _legacy_key_prompt_fragments = (
        "enter your selected provider's api key",
        "enter your selected provider’s api key",
        "configure it in streamlit secrets as openai_api_key",
        "to start the ai conversation",
    )
    if "manager_ai_messages" not in st.session_state:
        st.session_state.manager_ai_messages = [{"role": "assistant", "content": "Hello! I'm Retail Brain's built-in manager assistant. I'm ready to help—no API key or setup needed. Try ‘Summarize the dashboard’ or ask a category-specific what-if question."}]
    else:
        st.session_state.manager_ai_messages = [
            msg for msg in st.session_state.manager_ai_messages
            if not any(fragment in str(msg.get("content", "")).lower() for fragment in _legacy_key_prompt_fragments)
        ]
        if not st.session_state.manager_ai_messages:
            st.session_state.manager_ai_messages = [{"role": "assistant", "content": "Hello! I'm Retail Brain's built-in manager assistant. I'm ready to help—no API key or setup needed. Try ‘Summarize the dashboard’ or ask a category-specific what-if question."}]
    control_col, _ = st.columns([1, 4])
    with control_col:
        if st.button("🗑️ Clear chat", key="clear_manager_ai_chat"):
            st.session_state.manager_ai_messages = []
            st.rerun()

    for msg in st.session_state.manager_ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about a decision, scenario, or what-if…", key="manager_ai_chat_input")
    if user_prompt:
        st.session_state.manager_ai_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        try:
            assistant_reply = _local_manager_answer(user_prompt, _assistant_context())
        except Exception as exc:
            assistant_reply = ("I couldn't fully read the current session metrics. You can still ask general Retail Brain questions, "
                               f"or try again after the dashboard refreshes. Technical detail: {exc}")
        st.session_state.manager_ai_messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

    with st.expander("About this assistant"):
        st.markdown(
            "- Ready to use immediately: no API key, provider selection, or settings required.\n"
            "- It reads the current session's synthetic profiles, metrics, scenarios, and decision history.\n"
            "- It is a built-in context-aware assistant, not a connection to ChatGPT or Gemini. Its what-if explanations are guidance; use Trial & error lab for numerical recalculation.\n"
            "- All figures are simulated and should not be treated as verified real-world outcomes."
        )

if selected_page == "🔬 Model explainability":
    st.subheader("Model explainability and feature contributions")
    model, model_description = get_model()
    st.write(f"**Demand estimator:** {model_description}")
    st.caption("The model is trained on generated synthetic examples. Feature importance describes this prototype's fitted model and must not be interpreted as a validated real-world causal explanation.")
    X_train, y_train = make_training_data()
    if XGBOOST_AVAILABLE and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    else:
        # Transparent proxy importance ranking for fallback mode.
        proxy = {"price": .18, "competitor_price": .09, "promo_pct": .13, "display_pct": .08,
                 "temperature": .04, "traffic_idx": .17, "stock_units": .12, "lead_time_days": .04,
                 "expiry_days": .04, "staffing_units": .05, "unit_cost": .01, "category_factor": .05}
        importance = pd.DataFrame({"Feature": list(proxy), "Importance": list(proxy.values())}).sort_values("Importance", ascending=False)
    importance["Feature label"] = importance["Feature"].map(FEATURE_LABELS)
    try:
        import altair as alt
        importance_chart = alt.Chart(importance).mark_bar().encode(
            x=alt.X("Importance:Q", title="Relative importance"),
            y=alt.Y("Feature label:N", sort="-x", title="Feature", axis=alt.Axis(labelAngle=0)),
            tooltip=["Feature label", "Importance"],
        )
        st.altair_chart(importance_chart, use_container_width=True)
    except Exception:
        st.dataframe(importance[["Feature label", "Importance"]].round(4), use_container_width=True, hide_index=True)
    st.dataframe(importance[["Feature label", "Importance"]].round(4), use_container_width=True, hide_index=True)

    st.markdown("#### Local SHAP explanation")
    selected_explain_category = st.selectbox("Category for explanation", list(CATEGORIES), key="explain_category")
    explain_profile = current_profile(selected_explain_category)
    explanation, shap_error = shap_explanation(explain_profile)
    if explanation is not None:
        exp_values = explanation["values"]
        impact_df = pd.DataFrame({
            "Feature": FEATURES,
            "Feature label": [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "Current value": [float(explanation["row"].iloc[0][f]) for f in FEATURES],
            "SHAP contribution (units/day)": exp_values,
            "Absolute contribution": np.abs(exp_values),
        }).sort_values("Absolute contribution", ascending=False)
        st.metric("Estimated daily demand", f"{explanation['prediction']:.1f} units", help="Prediction from the currently fitted model.")
        st.caption(f"SHAP base value: {explanation['base_value']:.2f} units/day. Contributions sum approximately to the model prediction, subject to model/explainer numeric tolerance.")
        try:
            import matplotlib.pyplot as plt
            exp = shap.Explanation(values=explanation["values"], base_values=explanation["base_value"],
                                   data=explanation["row"].iloc[0].to_numpy(dtype=float),
                                   feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES])
            fig = plt.figure(figsize=(10, 5.8))
            shap.plots.waterfall(exp, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"SHAP values are available, but the waterfall chart could not be rendered: {exc}")
        st.dataframe(impact_df[["Feature label", "Current value", "SHAP contribution (units/day)", "Absolute contribution"]].round(3), use_container_width=True, hide_index=True)
        st.download_button("Download local SHAP contributions (CSV)", impact_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"retail_brain_shap_{selected_explain_category.lower().replace(' ', '_')}.csv", mime="text/csv")
    else:
        st.warning(shap_error or "Real SHAP explanation unavailable.")
        st.info("No heuristic proxy is shown as SHAP. Install compatible versions of xgboost and shap to enable model-specific local explanations.")

    st.markdown("#### Trust calibration — manager/AI agreement")
    if not st.session_state.decision_history:
        st.info("Record manager decisions to start measuring agreement with the AI recommendation.")
    else:
        trust_df = pd.DataFrame(st.session_state.decision_history)
        if "decision" in trust_df.columns:
            decided = trust_df[trust_df["decision"].isin(["approved", "rejected"])].copy()
            if decided.empty:
                st.info("Only deferred decisions are recorded so far. Agreement is calculated after approve/reject decisions.")
            else:
                # Explicitly a behavioral agreement proxy; not a validated psychometric trust scale.
                decided["AI recommendation"] = decided.get("ai_recommendation", pd.Series("approve", index=decided.index)).fillna("approve")
                decided["Agrees with AI"] = decided["decision"].astype(str).str.lower() == decided["AI recommendation"].astype(str).str.lower()
                agreement = float(decided["Agrees with AI"].mean() * 100)
                overrides = int((~decided["Agrees with AI"]).sum())
                t1, t2, t3 = st.columns(3)
                t1.metric("Decision agreement", f"{agreement:.1f}%")
                t2.metric("Approve decisions", int((decided["decision"] == "approved").sum()))
                t3.metric("Overrides vs recommendation", overrides)
                st.caption("Operational proxy only: approval is treated as agreement with the default AI recommendation and rejection as an override. This is not a validated Trust in AI (TAI) or Decision Confidence (DC) scale; interpret alongside manager rationale and scenario severity.")
                decided["Decision date"] = pd.to_datetime(decided["decision_at"], errors="coerce")
                by_severity = decided.groupby("severity", dropna=False)["Agrees with AI"].mean().mul(100).reset_index(name="Agreement rate (%)")
                st.markdown("**Agreement by scenario severity**")
                try:
                    import altair as alt
                    chart = alt.Chart(by_severity).mark_bar(cornerRadiusEnd=5).encode(
                        x=alt.X("Agreement rate (%):Q", scale=alt.Scale(domain=[0, 100]), title="Agreement rate (%)"),
                        y=alt.Y("severity:N", title="Scenario severity", sort="-x", axis=alt.Axis(labelAngle=0)),
                        tooltip=["severity", alt.Tooltip("Agreement rate (%):Q", format=".1f")])
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.dataframe(by_severity, use_container_width=True, hide_index=True)
                decided = decided.sort_values("Decision date")
                decided["Agreement rate (%)"] = decided["Agrees with AI"].expanding().mean().mul(100)
                st.markdown("**Cumulative agreement over recorded decisions**")
                try:
                    import altair as alt
                    chart = alt.Chart(decided).mark_line(point=True).encode(
                        x=alt.X("Decision date:T", title="Decision time"),
                        y=alt.Y("Agreement rate (%):Q", scale=alt.Scale(domain=[0, 100]), title="Cumulative agreement (%)"),
                        tooltip=["decision_at", "template", "decision", "Agreement rate (%)"])
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.dataframe(decided[["decision_at", "template", "decision", "Agreement rate (%)"]], use_container_width=True, hide_index=True)

# Footer
st.divider()
if st.session_state.get("persistence_warning"):
    st.warning("SQLite audit persistence is unavailable in this environment; decisions remain session-only. Details: " + str(st.session_state.persistence_warning)[:240])
st.caption("Prototype limitations: synthetic data only; no real retailer feeds, orders, pricing, staffing, or compliance actions. Decision history is written to a local SQLite audit file where filesystem persistence is supported; hosted ephemeral filesystems may reset. Forecasts and SHAP explanations describe the fitted synthetic-data model and are not causal evidence or validated real-world performance.")
