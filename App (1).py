from __future__ import annotations

"""Lidl Multi-Agent Store Operations MVP — synthetic-data thesis/demo prototype.

Run: streamlit run app.py
Required: streamlit, pandas, numpy
Optional: xgboost, shap (the app has a transparent fallback if unavailable)
No connection to Lidl systems; all scenarios and outcomes are simulated in this app session.
"""

import json
import random
import uuid
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

st.set_page_config(page_title="Lidl Store Operations MVP", page_icon="🛒", layout="wide")

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
    st.session_state.decision_history.insert(0, {
        "event_id": event["id"], "created_at": event.get("created_at"),
        "decision_at": event["decision_at"], "template": event["template"],
        "category": event["category"], "variables": event["variables"],
        "decision": decision, "note": note.strip(),
        "delta_demand_units_at_review": event.get("delta_demand_units", 0),
        "delta_gross_profit_eur_at_review": event.get("delta_gross_profit_eur", 0),
    })


def style_status_table(df: pd.DataFrame):
    def color_status(value: Any) -> str:
        if value == "Adequate":
            return "background-color: #d9ead3; color: #173b1a; font-weight: 700"
        if value == "Warning":
            return "background-color: #fff2cc; color: #5c4600; font-weight: 700"
        if value == "Critical":
            return "background-color: #f4cccc; color: #6b0000; font-weight: 700"
        return ""
    styler = df.style.set_properties(**{"color": "#111827"}).set_table_styles([
        {"selector": "th", "props": [("background-color", "#e5e7eb"), ("color", "#111827"), ("font-weight", "bold"), ("border", "1px solid #9ca3af")]},
        {"selector": "td", "props": [("border", "1px solid #d1d5db")]},
    ])
    if "Status" in df.columns:
        try:
            styler = styler.map(color_status, subset=["Status"])
        except Exception:
            styler = styler.applymap(color_status, subset=["Status"])
    return styler


def render_sound_control(pending_count: int) -> None:
    # Browsers generally require a user gesture before audio playback is allowed.
    components.html(f"""
    <div style="font-family:Arial,sans-serif;padding:4px 0;color:#111827">
      <button id="soundButton" style="border:1px solid #9ca3af;border-radius:8px;padding:8px 12px;background:#f3f4f6;cursor:pointer">
        🔔 Enable / test sound alert
      </button>
      <span style="margin-left:8px;font-size:12px;color:#4b5563">Pending alerts: {int(pending_count)}</span>
      <script>
      const btn = document.getElementById('soundButton');
      btn.addEventListener('click', async () => {{
        try {{
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          if (ctx.state === 'suspended') await ctx.resume();
          const osc = ctx.createOscillator(); const gain = ctx.createGain();
          osc.type = 'sine'; osc.frequency.value = 880; gain.gain.value = 0.08;
          osc.connect(gain); gain.connect(ctx.destination); osc.start();
          setTimeout(() => {{ osc.stop(); ctx.close(); }}, 220);
          btn.textContent = '✓ Sound enabled / tested';
        }} catch(e) {{ btn.textContent = 'Sound blocked by browser'; }}
      }});
      </script>
    </div>
    """, height=54)


def fmt_eur(value: float) -> str:
    return f"€{value:,.2f}"


def fmt_num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"

# -----------------------------------------------------------------------------
# Header, controls, and alerts
# -----------------------------------------------------------------------------
st.title("🛒 Lidl Multi-Agent Store Operations MVP")
st.caption("Synthetic-data prototype for managerial decision support · No live Lidl systems or real operational actions")

with st.sidebar:
    st.header("Control room")
    st.write("**Simulation mode:** session-only state")
    if st.button("🔎 Scan for operational scenarios", use_container_width=True, type="primary"):
        scan_scenarios(3)
        st.rerun()
    if st.button("➕ Generate one new alert", use_container_width=True):
        name = random.choice(list(SCENARIO_TEMPLATES))
        category = random.choice(list(CATEGORIES))
        event = create_scenario(name, category, "Manual alert")
        st.session_state.events.insert(0, event)
        st.rerun()
    st.divider()
    st.session_state.auto_monitor = st.toggle("Auto-scan every 2 minutes", value=st.session_state.auto_monitor,
                                              help="Simulation only. Requires the app session to remain open.")
    if st.session_state.auto_monitor:
        st.info("Auto-monitor is session-scoped. Streamlit refresh behavior depends on version and browser activity.")
    st.divider()
    st.write("**Scenario variables**")
    st.caption("Price · promotion · display · traffic · weather · stock · lead time · expiry · staffing · compliance")
    st.divider()
    st.write("**Alert legend**")
    st.markdown("🔴 Critical  ·  🟡 Warning  ·  🔵 Informational")

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
if pending_events:
    if critical_count:
        st.error(f"🚨 **Operational alert:** {len(pending_events)} scenario(s) need manager review, including {critical_count} critical alert(s). Review decisions are simulated and require manager input.")
    else:
        st.warning(f"🔔 **Manager review required:** {len(pending_events)} scenario(s) are pending approval, rejection, or deferral.")
else:
    st.success("No pending simulated alerts. Use **Scan for operational scenarios** to generate a new review queue.")
render_sound_control(len(pending_events))

# KPI cards
current_df = metrics_frame()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Categories monitored", len(CATEGORIES))
col2.metric("Pending review", len(pending_events))
col3.metric("Critical alerts", critical_count)
col4.metric("Projected daily revenue", fmt_eur(float(current_df["revenue_eur"].sum())))
col5.metric("Projected gross profit", fmt_eur(float(current_df["gross_profit_eur"].sum())))

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
# Main workspace tabs
# -----------------------------------------------------------------------------
tab_dashboard, tab_scenarios, tab_effects, tab_learning, tab_history, tab_model = st.tabs([
    "📊 9-agent dashboard", "🚨 Scenario centre", "📈 Approved scenario effects",
    "🧠 Decision learning", "🗂 Decision history", "🔬 Model explainability",
])

with tab_dashboard:
    st.subheader("Nine-agent operational dashboard")
    st.caption("Agent outputs are synthesized from category profiles and the demand model; they are not connected to production systems.")
    df = metrics_frame()
    display = df[["category", "demand_units", "sales_units", "lost_sales_units", "revenue_eur", "gross_profit_eur", "margin_pct", "stockout_risk_pct", "waste_cost_eur", "status"]].copy()
    display.columns = ["Category", "Demand (units/day)", "Sales (units/day)", "Lost sales (units/day)", "Revenue/day (€)", "Gross profit/day (€)", "Margin (%)", "Stockout risk (%)", "Waste cost proxy (€)", "Status"]
    for c in ["Demand (units/day)", "Sales (units/day)", "Lost sales (units/day)"]:
        display[c] = display[c].round(1)
    for c in ["Revenue/day (€)", "Gross profit/day (€)", "Waste cost proxy (€)"]:
        display[c] = display[c].round(2)
    display["Margin (%)"] = display["Margin (%)"].round(1)
    display["Stockout risk (%)"] = display["Stockout risk (%)"].round(1)
    st.dataframe(style_status_table(display), use_container_width=True, hide_index=True)

    agents = [
        ("1. Demand forecasting", "Predicts daily demand by category", "Estimated daily demand", "demand_units", "units/day"),
        ("2. Inventory risk", "Flags projected stockouts and low cover", "Stockout risk", "stockout_risk_pct", "%"),
        ("3. Commercial intelligence", "Compares own and competitor pricing", "Price gap", None, "€/unit"),
        ("4. Replenishment", "Highlights demand exceeding available stock", "Lost sales exposure", "lost_sales_units", "units/day"),
        ("5. Freshness & waste", "Estimates expiry-related waste pressure", "Waste cost proxy", "waste_cost_eur", "€/day"),
        ("6. Workforce planning", "Compares demand load with staffing capacity", "Staffing pressure", "staffing_pressure", "index"),
        ("7. Promotion & display", "Tracks promo/display inputs in the demand model", "Promotion intensity", None, "%"),
        ("8. Profitability", "Estimates gross profit and gross margin", "Gross profit", "gross_profit_eur", "€/day"),
        ("9. Compliance & task management", "Raises flags for compliance or critical operating risks", "Compliance / status", None, "flag"),
    ]
    cards = st.columns(3)
    for i, (name, desc, label, field, unit) in enumerate(agents):
        with cards[i % 3]:
            st.markdown(f"**{name}**")
            st.caption(desc)
            if field:
                value = float(df[field].mean())
                if unit == "€/day": value_text = fmt_eur(value)
                elif unit == "%": value_text = f"{value:.1f}%"
                elif unit == "units/day": value_text = f"{value:.1f}"
                else: value_text = f"{value:.2f}"
            elif i == 2:
                gaps = [float(current_profile(c)["competitor_price"])-float(current_profile(c)["price"]) for c in CATEGORIES]
                value_text = f"{np.mean(gaps):+.2f} €/unit"
            elif i == 6:
                value_text = f"{np.mean([current_profile(c)['promo_pct'] for c in CATEGORIES]):.1f}%"
            else:
                value_text = f"{int(df['compliance_risk'].sum())} risk flag(s)"
            st.metric(label, value_text)
            st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("#### Demand and available stock")
        chart_df = df.set_index("category")[["demand_units", "sales_units"]].rename(columns={"demand_units": "Estimated demand", "sales_units": "Fulfillable sales"})
        st.bar_chart(chart_df)
    with right:
        st.markdown("#### Gross profit by category")
        st.bar_chart(df.set_index("category")[["gross_profit_eur"]].rename(columns={"gross_profit_eur": "Gross profit (€ / day)"}))

    st.markdown("#### Category-level profile")
    selected_category = st.selectbox("Inspect a category", list(CATEGORIES.keys()), key="dashboard_category")
    profile = current_profile(selected_category)
    metric = calculate_metrics(selected_category, profile)
    with st.expander(f"{selected_category}: simulated operational inputs", expanded=True):
        p1, p2, p3 = st.columns(3)
        with p1:
            st.write(f"**Own price:** {fmt_eur(profile['price'])}")
            st.write(f"**Competitor price:** {fmt_eur(profile['competitor_price'])}")
            st.write(f"**Promotion:** {profile['promo_pct']:.1f}%")
            st.write(f"**Display quality:** {profile['display_pct']:.1f}%")
        with p2:
            st.write(f"**Stock:** {profile['stock_units']} units")
            st.write(f"**Lead time:** {profile['lead_time_days']} days")
            st.write(f"**Expiry:** {profile['expiry_days']} days")
            st.write(f"**Temperature:** {profile['temperature']:.1f} °C")
        with p3:
            st.write(f"**Traffic index:** {profile['traffic_idx']:.1f}")
            st.write(f"**Staffing capacity:** {profile['staffing_units']}")
            st.write(f"**Unit cost:** {fmt_eur(profile['unit_cost'])}")
            st.write(f"**Compliance flag:** {'Yes' if profile.get('compliance_issue') else 'No'}")
        st.write(f"Estimated daily demand: **{metric['demand_units']:.1f} units** · Status: **{metric['status']}**")

with tab_scenarios:
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
                       file_name="lidl_manager_action_queue.csv", mime="text/csv")
    impact_csv = impact_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download projected category impact (CSV)", data=impact_csv,
                       file_name="lidl_projected_category_impact.csv", mime="text/csv")

with tab_effects:
    st.subheader("Approved scenario effects")
    approved = [e for e in st.session_state.events if e.get("status") == "approved"]
    if not approved:
        st.info("No scenarios have been approved in this session. Approve a scenario from the Scenario centre to see recalculated before/after effects here.")
    else:
        rows = []
        for event in approved:
            impact = get_live_event_impact(event)
            rows.append({
                "Scenario": event["template"], "Category": event["category"],
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
        st.caption("Before/after values are recalculated against the current session state with the selected approved event excluded for the baseline. With multiple approved events, these are current counterfactual comparisons, not the original review-time values and not causal estimates.")
        st.download_button("⬇️ Download approved scenario effects (CSV)", effects_df.to_csv(index=False).encode("utf-8"),
                           file_name="lidl_approved_scenario_effects.csv", mime="text/csv")

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

with tab_learning:
    st.subheader("Decision learning: historical case reference")
    st.write("This module does not train an autonomous policy or claim causal learning. It retrieves similar manager-reviewed cases using category, scenario type, and overlap in changed variables.")
    if not st.session_state.decision_history:
        st.info("No decisions recorded yet. Review a scenario and record a decision to build a session-level reference set.")
    else:
        hist = pd.DataFrame(st.session_state.decision_history)
        st.metric("Recorded decision records", len(hist))
        st.metric("Unique scenarios with a recorded decision", hist["event_id"].nunique())
        st.markdown("#### Decision mix")
        st.bar_chart(hist["decision"].value_counts().rename_axis("Decision").to_frame("Count"))
        st.markdown("#### Case library")
        cols = ["decision_at", "template", "category", "decision", "delta_demand_units_at_review", "delta_gross_profit_eur_at_review", "note"]
        hist_view = hist[[c for c in cols if c in hist.columns]].copy()
        hist_view = hist_view.rename(columns={
            "decision_at": "Decision time", "template": "Scenario", "category": "Category", "decision": "Decision",
            "delta_demand_units_at_review": "Projected demand delta at review",
            "delta_gross_profit_eur_at_review": "Projected gross-profit delta (€)", "note": "Manager note",
        })
        st.dataframe(hist_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export decision case library (CSV)", hist_view.to_csv(index=False).encode("utf-8"),
                           file_name="lidl_decision_case_library.csv", mime="text/csv")

    st.markdown("#### Record a manager-observed outcome")
    decided_events = [e for e in st.session_state.events if e.get("status") in ("approved", "rejected", "deferred")]
    if decided_events:
        chosen = st.selectbox("Scenario", options=decided_events, format_func=lambda e: f"{e['category']} · {e['template']} · {e['status'].title()}", key="outcome_event_select")
        outcome_note = st.text_area("Observed outcome / follow-up note", key="outcome_note", placeholder="Example: next-day shelf check showed no stockout; estimate not independently verified.")
        outcome_label = st.selectbox("Outcome assessment", ["Not yet assessed", "Better than expected", "As expected", "Worse than expected", "Insufficient evidence"])
        if st.button("Save manager outcome note"):
            chosen["outcome_note"] = outcome_note.strip()
            chosen["outcome_assessment"] = outcome_label
            chosen["outcome_recorded_at"] = now_iso()
            # Attach outcome to matching history entries; the note remains manager-entered.
            for record in st.session_state.decision_history:
                if record.get("event_id") == chosen["id"]:
                    record["outcome_note"] = outcome_note.strip()
                    record["outcome_assessment"] = outcome_label
                    record["outcome_recorded_at"] = chosen["outcome_recorded_at"]
            st.success("Outcome note saved in the current session.")
            st.rerun()
    else:
        st.caption("A scenario must have a recorded decision before an outcome note can be attached.")

with tab_history:
    st.subheader("Decision audit trail")
    if st.session_state.decision_history:
        hist = pd.DataFrame(st.session_state.decision_history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download decision audit trail (CSV)", hist.to_csv(index=False).encode("utf-8"),
                           file_name="lidl_decision_audit_trail.csv", mime="text/csv")
    else:
        st.info("No decisions have been recorded yet.")
    st.markdown("#### All scenarios in this session")
    if st.session_state.events:
        all_events = []
        for e in st.session_state.events:
            all_events.append({"Created": e["created_at"], "Scenario": e["template"], "Category": e["category"],
                               "Status": e["status"], "Severity": e["severity"], "Changes": e["change_summary"],
                               "Manager rationale": e.get("decision_note", ""), "Outcome assessment": e.get("outcome_assessment", "")})
        all_events_df = pd.DataFrame(all_events)
        st.dataframe(all_events_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download full scenario log (CSV)", all_events_df.to_csv(index=False).encode("utf-8"),
                           file_name="lidl_scenario_log.csv", mime="text/csv")

with tab_model:
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
    st.bar_chart(importance.set_index("Feature label")[["Importance"]])
    st.dataframe(importance[["Feature label", "Importance"]].round(4), use_container_width=True, hide_index=True)

    st.markdown("#### Local feature contribution proxy")
    selected_explain_category = st.selectbox("Category for explanation", list(CATEGORIES), key="explain_category")
    explain_profile = current_profile(selected_explain_category)
    baseline_profile = st.session_state.base_profiles[selected_explain_category]
    contributions = []
    for feature in FEATURES:
        current_value = float(explain_profile[feature])
        baseline_value = float(baseline_profile[feature])
        delta = current_value - baseline_value
        # Explicit directional heuristic, clearly labeled as a proxy, not SHAP.
        direction = -1 if feature in ("price", "lead_time_days") else 1
        if feature == "competitor_price": direction = 1
        proxy_contribution = delta * direction
        contributions.append({"Feature": feature, "Feature label": FEATURE_LABELS[feature],
                              "Current value": round(current_value, 3), "Baseline value": round(baseline_value, 3),
                              "Input delta": round(delta, 3), "Directional contribution proxy": round(proxy_contribution, 3)})
    impact_df = pd.DataFrame(contributions)
    impact_df["Absolute contribution proxy"] = impact_df["Directional contribution proxy"].abs()
    impact_df = impact_df.sort_values("Absolute contribution proxy", ascending=False)
    st.dataframe(impact_df[["Feature label", "Current value", "Baseline value", "Input delta", "Directional contribution proxy", "Absolute contribution proxy"]],
                 use_container_width=True, hide_index=True)
    st.caption("SHAP is optional and installed if available, but this screen deliberately labels the local table as a directional heuristic. It is not a SHAP explanation. For rigorous SHAP values, validate the model, background data, and explainer configuration before use.")

# Footer
st.divider()
st.caption("Prototype limitations: synthetic data only; no real Lidl feeds, orders, pricing, staffing, or compliance actions. Decisions, manager notes, and scenarios are held in the current Streamlit session and are not persisted to a database. Forecasts are exploratory and require validation before operational use.")
