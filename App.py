from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import retail_brain_core as core

try:
    import altair as alt
    ALTAIR = True
except Exception:  # pragma: no cover
    alt = None
    ALTAIR = False

# -----------------------------------------------------------------------------
# Persistence: SQLite audit table next to the app (path can be overridden with the
# RETAIL_BRAIN_AUDIT_DB environment variable). It survives refreshes and restarts on
# the same filesystem. Hosts with ephemeral storage (for example Streamlit Community
# Cloud after a redeploy) can still reset it, so it is not a managed database.
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "retail_operations_data.csv"
AUDIT_DB_PATH = Path(os.environ.get("RETAIL_BRAIN_AUDIT_DB", str(APP_DIR / "retail_brain_audit.sqlite3")))


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(AUDIT_DB_PATH, timeout=5)
    conn.execute("CREATE TABLE IF NOT EXISTS decision_history (event_id TEXT, decision_at TEXT, payload TEXT NOT NULL)")
    return conn


def load_persisted_decisions() -> list[dict[str, Any]]:
    """Newest first. Never raises; problems are reported once in the footer."""
    try:
        conn = _db()
        try:
            rows = conn.execute("SELECT payload FROM decision_history ORDER BY rowid DESC").fetchall()
        finally:
            conn.close()
        return [json.loads(r[0]) for r in rows]
    except Exception as exc:
        st.session_state["persistence_warning"] = str(exc)
        return []


def persist_decision(record: dict[str, Any]) -> bool:
    try:
        conn = _db()
        try:
            conn.execute("INSERT INTO decision_history (event_id, decision_at, payload) VALUES (?, ?, ?)",
                         (str(record.get("event_id", "")), str(record.get("decision_at", "")), json.dumps(record, ensure_ascii=False, default=str)))
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:
        st.session_state["persistence_warning"] = str(exc)
        return False

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
/* readable text inside dialogs, and no stray chart tooltips */
[data-testid="stDialog"] [data-testid="stCaptionContainer"], [data-testid="stDialog"] [data-testid="stCaptionContainer"] p,
[role="dialog"] [data-testid="stCaptionContainer"], [role="dialog"] [data-testid="stCaptionContainer"] p,
[data-testid="stDialog"] small, [role="dialog"] small { color: var(--text-color) !important; opacity: 1 !important; }
[role="dialog"] button[kind="primary"] { background: #6254c7 !important; border-color: #6254c7 !important; color: #fff !important; border-radius: 10px; font-weight: 650; }
/* spacing between the mini tabs inside pages and dialogs (covers old and new Streamlit tab markup) */
[data-baseweb="tab-list"], [role="tablist"] { gap: .6rem !important; border-bottom: 1px solid rgba(120,120,150,.18); }
[data-testid="stTab"], button[role="tab"] { padding: .6rem 1.25rem !important; margin: 0 .2rem !important; font-weight: 600 !important; white-space: nowrap; border-radius: 10px 10px 0 0; }
[data-testid="stTab"][aria-selected="true"], button[role="tab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"] p, button[role="tab"][aria-selected="true"] p { color: #6254c7 !important; }
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"] { background-color: #6254c7 !important; height: 3px !important; }
#vg-tooltip-element:not(.visible) { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Scenario templates (expressed in the variables of the real data schema)
# Change grammar: <field>_pct = multiplicative %, <field>_delta = additive, <field>_set = overwrite
# -----------------------------------------------------------------------------
CHANGE_LABELS = {
    "price_pct": "Esselunga price", "competitor_price_pct": "Competitor prices", "traffic_pct": "Store traffic",
    "price_lidl_pct": "Lidl price", "price_conad_pct": "Conad price", "price_carrefour_pct": "Carrefour price",
    "temperature_delta": "Temperature", "delay_hrs_delta": "Delivery delay (h)",
    "staff_capacity_pct_delta": "Staff capacity (pts)", "compliance_pct_delta": "Shelf compliance (pts)",
    "unit_cost_pct": "Unit cost", "promo_active_set": "Promotion active", "cold_chain_set": "Cold-chain failure",
    "partner_active_set": "Partnership deal", "shelf_share_pl_delta": "Private-label shelf share (pts)",
    "pl_share_delta": "Private-label sales share (pts)", "event_set": "Calendar event",
}
SCENARIO_TEMPLATES = {
    "Competitor price reduction": {
        "description": "A competitor cuts price, which widens our price gap and may divert price-sensitive visitors.",
        "changes": {"competitor_price_pct": -12, "traffic_pct": -5},
        "variables": ["competitor_price", "traffic"], "dynamic": "competitor",
    },
    "Demand spike and delivery delay": {
        "description": "Traffic rises while a delayed delivery tightens availability and stretches staff.",
        "changes": {"traffic_pct": 18, "delay_hrs_delta": 3, "staff_capacity_pct_delta": -8},
        "variables": ["traffic", "delay_hrs", "staff_capacity_pct"],
    },
    "Promotion opportunity": {
        "description": "Switching on a campaign may lift demand, but it also draws on supply and staff.",
        "changes": {"promo_active_set": 1, "traffic_pct": 5},
        "variables": ["promo_active", "traffic"],
    },
    "Freshness and waste risk": {
        "description": "A cold-chain failure on a warm day raises write-offs on perishable lines.",
        "changes": {"cold_chain_set": 1, "temperature_delta": 4, "traffic_pct": -8},
        "variables": ["cold_chain", "temperature", "traffic"],
    },
    "Hot weather and traffic increase": {
        "description": "Warmer weather and more visitors can increase demand for selected categories.",
        "changes": {"temperature_delta": 7, "traffic_pct": 15},
        "variables": ["temperature", "traffic"],
    },
    "Cost and margin pressure": {
        "description": "Own price, competitor prices and unit cost move, squeezing estimated gross margin.",
        "changes": {"price_pct": 5, "competitor_price_pct": -4, "unit_cost_pct": 6},
        "variables": ["price", "competitor_price", "unit_cost"],
    },
    "Staffing and compliance alert": {
        "description": "Lower staffing during higher traffic creates an operational compliance risk.",
        "changes": {"staff_capacity_pct_delta": -15, "traffic_pct": 6, "compliance_pct_delta": -12},
        "variables": ["staff_capacity_pct", "traffic", "compliance_pct"],
    },
    "Partnership deal activated": {
        "description": "A partnership deal goes live and may bring extra visitors to the category.",
        "changes": {"partner_active_set": 1, "traffic_pct": 4},
        "variables": ["partner_active", "traffic"],
    },
    "Private-label expansion": {
        "description": "More shelf space for the private label shifts the sales mix toward lower-priced own-brand products.",
        "changes": {"shelf_share_pl_delta": 10, "pl_share_delta": 5},
        "variables": ["shelf_share_pl", "pl_share"],
    },
    "Seasonal event peak": {
        "description": "A calendar event lifts store traffic and demand.",
        "changes": {"traffic_pct": 20},
        "variables": ["event", "traffic"], "dynamic": "event",
    },
}

# -----------------------------------------------------------------------------
# Data, model and context (cached)
# -----------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@st.cache_data(show_spinner=False)
def _load_default_data():
    return core.load_operations_csv(DEFAULT_CSV)


def get_data():
    """Return (df, warnings, source_label). An uploaded file replaces the bundled CSV."""
    up = st.session_state.get("upload_data")
    if up is not None:
        return up["df"], up["warnings"], up["name"]
    df, warnings = _load_default_data()
    return df, warnings, "Bundled demo file (retail_operations_data.csv)"


@st.cache_resource(show_spinner="Training the demand model on history before the selected date…")
def _train_cached(fingerprint: str, asof_iso: str, _df: pd.DataFrame):
    return core.train_model(_df, pd.Timestamp(asof_iso))


def init_state() -> None:
    defaults = {
        "events": [], "decision_history": [], "approved_event_ids": [],
        "base_profiles": {}, "context_key": None,
        "last_scan_at": None, "last_auto_event_at": None, "auto_monitor": False,
        "scan_count": 0, "upload_data": None, "upload_error": None,
        "popup_enabled": False, "tasks": {}, "manual_tasks": [],
        "assumed_elasticity": core.DEFAULT_ASSUMED_ELASTICITY,
        "assumed_promo_uplift": core.DEFAULT_ASSUMED_PROMO_UPLIFT,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.get("audit_history_loaded"):
        st.session_state.decision_history = load_persisted_decisions()
        st.session_state.audit_history_loaded = True


init_state()
DF, DATA_WARNINGS, DATA_SOURCE = get_data()
FINGERPRINT = core.data_fingerprint(DF)
ALL_DATES = sorted(DF["Date"].dt.normalize().unique())
MIN_ASOF = ALL_DATES[min(119, len(ALL_DATES) - 1)]
MAX_ASOF = ALL_DATES[-1]

with st.sidebar:
    st.markdown("<div style='padding:8px 4px 16px;font-size:1.35rem;font-weight:800;color:var(--text-color)'>◈ Retail Brain</div>", unsafe_allow_html=True)
    nav_slot = st.container()
    with st.expander("📅 Data and as-of date", expanded=False):
        st.caption(f"Source: {DATA_SOURCE}")
        st.caption(f"{len(DF):,} rows · {DF['Category'].nunique()} categories · {pd.Timestamp(ALL_DATES[0]).date()} to {pd.Timestamp(MAX_ASOF).date()}")
        chosen_date = st.date_input("Decision day (as-of date)", value=pd.Timestamp(MAX_ASOF).date(),
                                    min_value=pd.Timestamp(MIN_ASOF).date(), max_value=pd.Timestamp(MAX_ASOF).date(),
                                    help="The model is trained only on days before this date, so nothing from the future leaks into the forecast.")
        up = st.file_uploader("Replace with your own CSV", type=["csv"], help="Must follow the same column schema as the bundled file.")
        if up is not None and (st.session_state.upload_data is None or st.session_state.upload_data["name"] != up.name):
            try:
                udf, uw = core.load_operations_csv(up)
                st.session_state.upload_data = {"df": udf, "warnings": uw, "name": up.name}
                st.session_state.upload_error = None
                st.rerun()
            except Exception as exc:
                st.session_state.upload_error = str(exc)
        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)
        if st.session_state.upload_data is not None and st.button("Use the bundled demo file again", use_container_width=True):
            st.session_state.upload_data = None
            st.rerun()
        for w in DATA_WARNINGS:
            st.warning(w)
    with st.expander("⚙️ Commercial-lever assumptions", expanded=False):
        st.session_state.assumed_elasticity = st.slider("Price elasticity vs competitors", -3.0, 0.0, float(st.session_state.assumed_elasticity), 0.1)
        st.session_state.assumed_promo_uplift = st.slider("Campaign demand uplift", 0.0, 0.5, float(st.session_state.assumed_promo_uplift), 0.01, format="%.2f")

_snap = [d for d in ALL_DATES if d <= np.datetime64(pd.Timestamp(chosen_date))]
ASOF = pd.Timestamp(_snap[-1] if _snap else MIN_ASOF)
MODEL = _train_cached(FINGERPRINT, ASOF.date().isoformat(), DF)
LEVERS = core.resolve_levers(MODEL.lever_est, st.session_state.assumed_elasticity, st.session_state.assumed_promo_uplift)

_ctx_key = (FINGERPRINT, ASOF.date().isoformat())
if st.session_state.context_key != _ctx_key:
    st.session_state.base_profiles = core.build_profiles(DF, MODEL, ASOF)
    if st.session_state.context_key is not None:
        st.session_state.events = []
        st.session_state.approved_event_ids = []
    st.session_state.context_key = _ctx_key
CATEGORIES = list(st.session_state.base_profiles.keys())


def current_profile(category: str, exclude_event_id: str | None = None) -> dict[str, Any]:
    """Apply approved scenario deltas to the as-of profile of a category."""
    profile = dict(st.session_state.base_profiles[category])
    for event in reversed(st.session_state.events):
        if event.get("status") != "approved" or event.get("category") != category:
            continue
        if exclude_event_id and event.get("id") == exclude_event_id:
            continue
        profile = core.apply_changes(profile, event.get("changes", {}))
    return profile


def calculate_metrics(category: str, profile: dict[str, Any], keep_log: bool = False) -> dict[str, Any]:
    return core.analyze(profile, MODEL, LEVERS, keep_log=keep_log)


def metrics_frame() -> pd.DataFrame:
    return pd.DataFrame([calculate_metrics(c, current_profile(c)) for c in CATEGORIES])


def scenario_metrics(category: str, changes: dict[str, Any], exclude_event_id: str | None = None):
    before_profile = current_profile(category, exclude_event_id=exclude_event_id)
    after_profile = core.apply_changes(before_profile, changes)
    return before_profile, after_profile, calculate_metrics(category, before_profile), calculate_metrics(category, after_profile)


def describe_changes(changes: dict[str, Any]) -> str:
    parts = []
    for key, value in changes.items():
        label = CHANGE_LABELS.get(key, key)
        if key.endswith("_pct"):
            parts.append(f"{label} {float(value):+g}%")
        elif key.endswith("_delta"):
            parts.append(f"{label} {float(value):+g}")
        elif key.endswith("_set") and key != "event_set":
            parts.append(f"{label}: {'yes' if value else 'no'}")
        else:
            parts.append(f"{label}: {value}")
    return " · ".join(parts)


def create_scenario(template_name: str, category: str, source: str = "Manual scan") -> dict[str, Any]:
    template = SCENARIO_TEMPLATES[template_name]
    changes = dict(template["changes"])
    description, variables = template["description"], list(template["variables"])
    if template.get("dynamic") == "competitor":
        comp = random.choice([n for n, _ in core.COMPETITORS])
        changes = {f"{core.COMP_KEYS[comp]}_pct": -15, "traffic_pct": -4}
        description = f"{comp} cuts its price by 15%, which widens the gap to Esselunga and may divert price-sensitive visitors."
        variables = [core.COMP_KEYS[comp], "traffic"]
    elif template.get("dynamic") == "event":
        events = [e for e in MODEL.events if str(e).lower() != "none"]
        ev = random.choice(events) if events else "None"
        changes = {"event_set": ev, "traffic_pct": 20}
        description = f"{ev} lifts store traffic and demand."
    if template_name == "Freshness and waste risk" and _is_non_perishable(category):
        changes.pop("cold_chain_set", None)
    if template_name == "Hot weather and traffic increase" and category == "Detersivi":
        changes["traffic_pct"] = 8
    before_profile, after_profile, before, after = scenario_metrics(category, changes)
    delta_demand = float(after["demand_units"]) - float(before["demand_units"])
    delta_profit = float(after["gross_profit_eur"]) - float(before["gross_profit_eur"])
    risk_rank = {"Adequate": 0, "Warning": 1, "Critical": 2}
    severity = "Critical" if after["status"] == "Critical" or risk_rank[after["status"]] > risk_rank[before["status"]] else ("Warning" if after["status"] == "Warning" else "Info")
    # The system's default recommendation: apply the change unless it lowers profit, raises stockout risk by
    # more than 5 points, or leaves the category Critical. Managers are measured against this (trust calibration).
    recommendation = "approve" if (delta_profit >= 0 and after["status"] != "Critical"
                                   and float(after["stockout_risk_pct"]) - float(before["stockout_risk_pct"]) <= 5) else "reject"
    return {
        "id": str(uuid.uuid4()), "created_at": now_iso(), "template": template_name,
        "description": description, "category": category, "changes": changes,
        "variables": variables, "status": "pending", "source": source,
        "severity": severity, "ai_recommendation": recommendation, "before": {k: v for k, v in before.items() if k not in ("agent_reports", "messages")},
        "after": {k: v for k, v in after.items() if k not in ("agent_reports", "messages")},
        "delta_demand_units": delta_demand, "delta_gross_profit_eur": delta_profit,
        "change_summary": describe_changes(changes),
    }


def _is_non_perishable(category: str) -> bool:
    return core.CATEGORY_ASSUMPTIONS.get(category, core.DEFAULT_ASSUMPTION)["shelf_life_days"] > 7


def available_templates() -> list[str]:
    has_events = any(str(e).lower() != "none" for e in MODEL.events)
    return [n for n, t in SCENARIO_TEMPLATES.items() if t.get("dynamic") != "event" or has_events]


def queue_popup(new_events: list[dict[str, Any]]) -> None:
    """When pop-up alerts are on, open the most severe new scenario automatically."""
    if st.session_state.get("popup_enabled") and new_events:
        rank = {"Critical": 2, "Warning": 1, "Info": 0}
        top = max(new_events, key=lambda e: rank.get(e.get("severity", "Info"), 0))
        st.session_state["popup_event_id"] = top["id"]


def scan_scenarios(count: int = 3, source: str = "Manual scan") -> list[dict[str, Any]]:
    pool = available_templates()
    names = random.sample(pool, k=min(count, len(pool)))
    new_events = [create_scenario(n, random.choice(CATEGORIES), source) for n in names]
    st.session_state.events = new_events + st.session_state.events
    st.session_state.last_scan_at = now_iso()
    st.session_state.scan_count += 1
    queue_popup(new_events)
    return new_events


def get_live_event_impact(event: dict[str, Any]) -> dict[str, Any]:
    """Current counterfactual impact of an event, not a historical causal estimate."""
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
# Explanation helpers (real SHAP "demand bridge")
# -----------------------------------------------------------------------------
def explain_profile(profile: dict[str, Any]) -> pd.DataFrame:
    return MODEL.explain(profile, LEVERS)


def render_bridge_chart(bridge: pd.DataFrame, key_prefix: str, top_n: int = 9) -> None:
    view = bridge[bridge["Type"] != "base"].sort_values("Absolute", ascending=False).head(top_n).copy()
    view["Effect"] = np.where(view["Contribution (units/day)"] >= 0, "Raises demand", "Lowers demand")
    view["Contribution"] = view["Contribution (units/day)"].round(2)
    if ALTAIR:
        chart = alt.Chart(view).mark_bar().encode(
            x=alt.X("Contribution:Q", title="Effect on forecast (units/day)"),
            y=alt.Y("Driver:N", sort=alt.EncodingSortField(field="Absolute", order="descending"), title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises demand", "Lowers demand"], range=["#6254c7", "#ff7a18"]), legend=alt.Legend(orient="bottom")),
            tooltip=["Driver", "Type", "Contribution"],
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(view[["Driver", "Contribution"]], use_container_width=True, hide_index=True)


def bridge_caption(bridge: pd.DataFrame) -> str:
    base = float(bridge.loc[bridge["Type"] == "base", "Contribution (units/day)"].iloc[0])
    return (f"Starts from a baseline of {base:.0f} units/day at typical traffic and average conditions, adds each driver, and reaches "
            f"the forecast of {bridge.attrs['forecast']:.0f} units/day. Context drivers are TreeSHAP values of the XGBoost rate model "
            "scaled by traffic; price and promotion rows are the lever adjustments applied on top.")


# -----------------------------------------------------------------------------
# Case-based decision support (not causal or autonomous learning)
# -----------------------------------------------------------------------------
def similarity_score(new_event: dict[str, Any], old_event: dict[str, Any]) -> float:
    score = 0.0
    if new_event.get("category") == old_event.get("category"):
        score += .35
    new_vars, old_vars = set(new_event.get("variables", [])), set(old_event.get("variables", []))
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


def compute_agreement_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Trust-calibration proxy: how often the manager's decision matches the system's recommendation.

    Each scenario carries a default recommendation (approve or reject, see create_scenario). Approving a
    scenario the system recommends approving, or rejecting one it recommends rejecting, counts as agreement.
    Deferrals are excluded. Records saved before recommendations existed are treated as 'approve'. This is a
    behavioural proxy for Trust in AI and Decision Confidence, not a validated psychometric scale.
    """
    decided = [h for h in history if h.get("decision") in ("approved", "rejected")]
    if not decided:
        return {"overall_rate": None, "n_decided": 0, "n_overrides": 0, "by_severity": pd.DataFrame(), "rolling": pd.DataFrame()}
    def agrees(h):
        rec = str(h.get("ai_recommendation") or "approve").lower()
        if h.get("modified"):
            return False
        return (h["decision"] == "approved") == (rec == "approve")
    flags = [agrees(h) for h in decided]
    rows = []
    for sev in ("Critical", "Warning", "Info"):
        idx = [i for i, h in enumerate(decided) if (h.get("severity") or "Info") == sev]
        if idx:
            rows.append({"Severity": sev, "Decisions": len(idx), "Agreement rate": sum(flags[i] for i in idx) / len(idx)})
    order = sorted(range(len(decided)), key=lambda i: decided[i].get("decision_at") or "")
    cum, rolling = 0, []
    for n, i in enumerate(order, start=1):
        cum += flags[i]
        rolling.append({"Decision #": n, "Cumulative agreement rate": cum / n})
    return {"overall_rate": sum(flags) / len(flags), "n_decided": len(decided), "n_overrides": len(flags) - sum(flags),
            "n_approved": sum(1 for h in decided if h["decision"] == "approved"),
            "by_severity": pd.DataFrame(rows), "rolling": pd.DataFrame(rolling)}


def record_decision(event: dict[str, Any], decision: str, note: str = "", modified: bool = False) -> None:
    event["status"] = decision
    event["decision_at"] = now_iso()
    event["decision_note"] = note.strip()
    if decision == "approved" and event["id"] not in st.session_state.approved_event_ids:
        st.session_state.approved_event_ids.append(event["id"])
    record = {
        "event_id": event["id"], "created_at": event.get("created_at"), "decision_at": event["decision_at"],
        "template": event["template"], "category": event["category"], "variables": event["variables"],
        "decision": decision, "note": note.strip(), "severity": event.get("severity", "Info"),
        "ai_recommendation": event.get("ai_recommendation", "approve"), "modified": bool(modified),
        "change_summary": event.get("change_summary", ""), "as_of": ASOF.date().isoformat(),
        "delta_demand_units_at_review": event.get("delta_demand_units", 0),
        "delta_gross_profit_eur_at_review": event.get("delta_gross_profit_eur", 0),
    }
    st.session_state.decision_history.insert(0, record)
    st.session_state["last_persist_ok"] = persist_decision(record)


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
# Header, navigation, controls, and alerts
# -----------------------------------------------------------------------------
if st.session_state.get("retail_brain_page", "📊 9-agent dashboard") != "📊 9-agent dashboard":
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.title(st.session_state.get("retail_brain_page", "Retail Brain").split(" ", 1)[-1])
        st.caption(f"Store operations workspace · Decision day {ASOF.date().isoformat()} · Model trained on {MODEL.n_train:,} category-days of history")
    with header_right:
        st.markdown("<div style='text-align:right;padding-top:1.1rem;color:var(--rb-muted);font-size:.82rem;font-weight:600'>● WORKSPACE ONLINE</div>", unsafe_allow_html=True)

PAGE_OPTIONS = [
    "📊 9-agent dashboard",
    "🚨 Decision center",
    "🏬 Store management",
    "📈 Decisions effects",
    "🧠 Decision learning & history",
    "🧪 Trial & error lab",
    "💬 AI manager assistant",
    "🔬 Model explainability",
    "✅ Data & validation",
]

with nav_slot:
    if "retail_brain_page" not in st.session_state or st.session_state.retail_brain_page not in PAGE_OPTIONS:
        st.session_state.retail_brain_page = PAGE_OPTIONS[0]
    st.caption("WORKSPACE")
    for page_label in PAGE_OPTIONS:
        is_active = st.session_state.retail_brain_page == page_label
        if st.button(page_label, key=f"nav_{PAGE_OPTIONS.index(page_label)}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.retail_brain_page = page_label
            st.rerun()
selected_page = st.session_state.retail_brain_page

with st.sidebar:
    st.divider()
    st.caption("ALERTS")
    st.session_state.auto_monitor = st.toggle("Auto-scan every 2 minutes", value=st.session_state.auto_monitor,
                                              help="Checks for new operational scenarios while the app stays open.")
    st.session_state.popup_enabled = st.toggle("Pop-up alerts", value=st.session_state.popup_enabled,
                                               help="Opens the review window automatically when a new scenario is raised.")
    if st.button("🔎 Scan for operational scenarios", use_container_width=True, type="primary"):
        scan_scenarios(3)
        st.rerun()
    if st.button("➕ Generate one new alert", use_container_width=True):
        ev = create_scenario(random.choice(available_templates()), random.choice(CATEGORIES), "Manual alert")
        st.session_state.events.insert(0, ev)
        queue_popup([ev])
        st.rerun()

if st.session_state.auto_monitor:
    def _seconds_since_last() -> float:
        try:
            last = st.session_state.last_auto_event_at
            return (datetime.now() - datetime.fromisoformat(last.replace("Z", "+00:00")).replace(tzinfo=None)).total_seconds() if last else 9999
        except Exception:
            return 9999

    if _seconds_since_last() >= 120:
        ev = create_scenario(random.choice(available_templates()), random.choice(CATEGORIES), "Auto-monitor")
        st.session_state.events.insert(0, ev)
        st.session_state.last_auto_event_at = now_iso()
        queue_popup([ev])

    @st.fragment(run_every=15)
    def _auto_scan_tick() -> None:
        if _seconds_since_last() >= 120:
            st.rerun()

    _auto_scan_tick()

pending_events = [e for e in st.session_state.events if e.get("status") == "pending"]
critical_count = sum(e.get("severity") == "Critical" for e in pending_events)
if pending_events and selected_page != "📊 9-agent dashboard":
    if critical_count:
        st.error(f"🚨 **{len(pending_events)} scenario(s) need review**, including {critical_count} critical. Open the Decision center.")
    else:
        st.warning(f"🔔 **{len(pending_events)} scenario(s)** are waiting for approval, rejection or deferral.")
try:
    render_sound_control(len(pending_events))
except Exception:
    pass


# -----------------------------------------------------------------------------
# Review dialog
# -----------------------------------------------------------------------------
@st.dialog("Review operational scenario", width="large")
def review_dialog(event_id: str) -> None:
    event = next((e for e in st.session_state.events if e["id"] == event_id), None)
    if not event:
        st.warning("This scenario is no longer available.")
        return
    st.subheader(f"{event['template']} · {event['category']}")
    st.write(f"{event['description']}")
    meta1, meta2, meta3 = st.columns(3)
    meta1.markdown(f"**Severity**<br>{event.get('severity', 'Info')}", unsafe_allow_html=True)
    meta2.markdown(f"**System recommendation**<br>{'Approve' if event.get('ai_recommendation', 'approve') == 'approve' else 'Reject'}", unsafe_allow_html=True)
    meta3.markdown(f"**Raised**<br>{event['created_at'][:16].replace('T', ' ')} · {event.get('source', '')}", unsafe_allow_html=True)
    st.markdown(f"**Scenario changes:** {event['change_summary']}")
    modify_key = f"modify_{event_id}"
    editing = bool(st.session_state.get(modify_key))

    if not editing:
        _, after_profile, before, after = scenario_metrics(event["category"], event["changes"])
        tab_impact, tab_why, tab_cases = st.tabs(["Impact", "Why this projection", "Similar decisions"])
        with tab_impact:
            metrics = [("Forecast demand", before["demand_units"], after["demand_units"], "units"),
                       ("Gross profit", before["gross_profit_eur"], after["gross_profit_eur"], "eur"),
                       ("Stockout risk", before["stockout_risk_pct"], after["stockout_risk_pct"], "pct"),
                       ("Waste cost proxy", before["waste_cost_eur"], after["waste_cost_eur"], "eur"),
                       ("Staffing pressure", before["staffing_pressure"], after["staffing_pressure"], "idx")]
            compare = []
            for label, bv, av, kind in metrics:
                if kind == "eur":
                    bs, ass, delta = fmt_eur(float(bv)), fmt_eur(float(av)), fmt_eur(float(av) - float(bv))
                elif kind == "pct":
                    bs, ass, delta = f"{float(bv):.1f}%", f"{float(av):.1f}%", f"{float(av)-float(bv):+.1f} pp"
                elif kind == "idx":
                    bs, ass, delta = f"{float(bv):.2f}", f"{float(av):.2f}", f"{float(av)-float(bv):+.2f}"
                else:
                    bs, ass, delta = fmt_num(float(bv)), fmt_num(float(av)), f"{float(av)-float(bv):+.1f}"
                compare.append({"Metric": label, "Before": bs, "After if applied": ass, "Change": delta})
            st.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)
            st.write(f"Demand after the change: **{after['demand_units']:.0f} units/day**, 80% interval {after['demand_low']:.0f} to {after['demand_high']:.0f}.")
        with tab_why:
            bridge = explain_profile(after_profile)
            render_bridge_chart(bridge, f"dlg_{event_id}")
            st.write(bridge_caption(bridge))
        with tab_cases:
            cases = find_relevant_cases(event)
            if cases:
                rows = [{"Similarity": f"{c['similarity']:.0%}", "Previous scenario": c.get("template", "—"), "Category": c.get("category", "—"),
                         "Decision": ("Modified and approved" if c.get("modified") else c.get("decision", "—").title()),
                         "Manager note": c.get("note") or "—",
                         "Profit delta at review": fmt_eur(float(c.get("delta_gross_profit_eur_at_review", 0)))} for c in cases]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write("No similar earlier decisions yet.")
        note = st.text_area("Manager rationale (optional)", key=f"note_{event_id}", placeholder="Why is this decision appropriate? What should be monitored?")
        a, b, c, d = st.columns(4)
        if a.button("✅ Approve", use_container_width=True, type="primary", key=f"approve_{event_id}"):
            record_decision(event, "approved", note)
            st.rerun()
        if b.button("❌ Reject", use_container_width=True, key=f"reject_{event_id}"):
            record_decision(event, "rejected", note)
            st.rerun()
        if c.button("⏸ Defer", use_container_width=True, key=f"defer_{event_id}"):
            record_decision(event, "deferred", note)
            event["status"] = "pending"
            st.rerun()
        if d.button("✏️ Modify", use_container_width=True, key=f"modify_btn_{event_id}", help="Edit the suggested values before deciding"):
            st.session_state[modify_key] = True
            st.rerun(scope="fragment")
    else:
        st.markdown("#### Edit the suggested values")
        mod: dict[str, Any] = {}
        cols = st.columns(2)
        for i, (key, val) in enumerate(event["changes"].items()):
            label, wk = CHANGE_LABELS.get(key, key), f"mod_{event_id}_{key}"
            with cols[i % 2]:
                if key == "event_set":
                    opts = list(MODEL.events)
                    mod[key] = st.selectbox(label, opts, index=opts.index(val) if val in opts else 0, key=wk)
                elif key.endswith("_set"):
                    mod[key] = int(st.checkbox(label, value=bool(val), key=wk))
                elif key.endswith("_pct"):
                    mod[key] = st.number_input(f"{label} (%)", value=float(val), step=1.0, key=wk)
                else:
                    mod[key] = st.number_input(label, value=float(val), step=0.5, key=wk)
        _, mod_profile, before, after = scenario_metrics(event["category"], mod)
        st.dataframe(pd.DataFrame([
            {"Metric": "Forecast demand (units/day)", "Before": round(before["demand_units"], 1), "With your values": round(after["demand_units"], 1)},
            {"Metric": "Gross profit (€/day)", "Before": round(before["gross_profit_eur"], 2), "With your values": round(after["gross_profit_eur"], 2)},
            {"Metric": "Stockout risk (%)", "Before": round(before["stockout_risk_pct"], 1), "With your values": round(after["stockout_risk_pct"], 1)},
            {"Metric": "Status", "Before": before["status"], "With your values": after["status"]}]),
            use_container_width=True, hide_index=True)
        note = st.text_area("Manager rationale (optional)", key=f"note_mod_{event_id}")
        a, b = st.columns(2)
        if a.button("✅ Approve with my values", use_container_width=True, type="primary", key=f"approve_mod_{event_id}"):
            event["original_changes"] = dict(event["changes"])
            event["changes"] = mod
            event["change_summary"] = describe_changes(mod)
            event["delta_demand_units"] = float(after["demand_units"] - before["demand_units"])
            event["delta_gross_profit_eur"] = float(after["gross_profit_eur"] - before["gross_profit_eur"])
            event["modified"] = True
            record_decision(event, "approved", (note + " [values modified by manager]").strip(), modified=True)
            st.session_state.pop(modify_key, None)
            st.rerun()
        if b.button("↩ Back", use_container_width=True, key=f"cancel_mod_{event_id}"):
            st.session_state.pop(modify_key, None)
            st.rerun(scope="fragment")


_popup_id = st.session_state.pop("popup_event_id", None)
if _popup_id and st.session_state.popup_enabled and any(e["id"] == _popup_id and e["status"] == "pending" for e in st.session_state.events):
    review_dialog(_popup_id)


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
STATUS_ICON = {"ok": "🟢", "watch": "🟡", "alert": "🔴"}

if selected_page == "🧪 Trial & error lab":
    sandbox_category = st.selectbox("Category to simulate", CATEGORIES, key="sandbox_category")
    baseline = current_profile(sandbox_category)
    tag = f"{sandbox_category}_{ASOF.date().isoformat()}"
    comp_names = [n for n, _ in core.COMPETITORS]
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("#### Inputs")
        t_price, t_demand, t_supply = st.tabs(["Prices", "Demand drivers", "Supply & staff"])
        with t_price:
            sim_price = st.slider(f"{core.OWN_NAME} price (€)", 0.10, float(max(5.0, baseline["price"] * 2)), float(baseline["price"]), 0.01, key=f"lab_price_{tag}")
            sim_comp = {}
            for n in comp_names:
                k = core.COMP_KEYS[n]
                sim_comp[k] = st.slider(f"{n} price (€)", 0.10, float(max(5.0, baseline[k] * 2)), float(baseline[k]), 0.01, key=f"lab_{k}_{tag}")
            sim_pl_price = st.slider("Private-label price (€)", 0.10, float(max(5.0, baseline["private_label_price"] * 2)), float(baseline["private_label_price"]), 0.01, key=f"lab_plp_{tag}")
            sim_promo = st.checkbox("Campaign active", value=bool(baseline["promo_active"]), key=f"lab_promo_{tag}")
        with t_demand:
            sim_traffic = st.slider("Store traffic (visitors/day)", 50, int(max(3000, baseline["traffic"] * 2)), int(baseline["traffic"]), 10, key=f"lab_traffic_{tag}")
            sim_temp = st.slider("Temperature (°C)", -10, 45, int(round(np.clip(baseline["temperature"], -10, 45))), 1, key=f"lab_temp_{tag}")
            sim_weather = st.selectbox("Weather", list(MODEL.weathers), index=list(MODEL.weathers).index(baseline["weather"]) if baseline["weather"] in MODEL.weathers else 0, key=f"lab_wx_{tag}")
            sim_event = st.selectbox("Calendar event", list(MODEL.events), index=list(MODEL.events).index(baseline["event"]) if baseline["event"] in MODEL.events else 0, key=f"lab_ev_{tag}")
            sim_partner = st.checkbox("Partnership deal active", value=bool(baseline["partner_active"]), key=f"lab_partner_{tag}")
            sim_pl_share = st.slider("Private-label sales share (%)", 0, 100, int(round(baseline["pl_share"])), 1, key=f"lab_plshare_{tag}")
            sim_shelf = st.slider("Private-label shelf share (%)", 0, 100, int(round(baseline["shelf_share_pl"])), 1, key=f"lab_shelf_{tag}")
        with t_supply:
            sim_delay = st.slider("Delivery delay (hours)", 0.0, 24.0, float(np.clip(baseline["delay_hrs"], 0, 24)), 0.5, key=f"lab_delay_{tag}")
            sim_staff = st.slider("Staff capacity (%)", 20, 100, int(np.clip(round(baseline["staff_capacity_pct"]), 20, 100)), 1, key=f"lab_staff_{tag}")
            sim_compliance = st.slider("Shelf compliance (%)", 0, 100, int(np.clip(round(baseline["compliance_pct"]), 0, 100)), 1, key=f"lab_compliance_{tag}")
            sim_cost = st.slider("Unit cost (€)", 0.05, float(max(5.0, baseline["unit_cost"] * 2)), float(baseline["unit_cost"]), 0.01, key=f"lab_cost_{tag}")
            sim_cold = st.checkbox("Cold-chain failure", value=bool(baseline["cold_chain"]), key=f"lab_cold_{tag}")
    simulated = dict(baseline)
    simulated.update({"price": sim_price, **sim_comp, "private_label_price": sim_pl_price, "promo_active": int(sim_promo),
                      "traffic": float(sim_traffic), "temperature": float(sim_temp), "weather": sim_weather, "event": sim_event,
                      "partner_active": int(sim_partner), "pl_share": float(sim_pl_share), "shelf_share_pl": float(sim_shelf),
                      "delay_hrs": float(sim_delay), "staff_capacity_pct": float(sim_staff), "compliance_pct": float(sim_compliance),
                      "unit_cost": float(sim_cost), "cold_chain": int(sim_cold)})
    simulated = core.apply_changes(simulated, {})
    before = calculate_metrics(sandbox_category, baseline)
    after = calculate_metrics(sandbox_category, simulated)
    dd, dp = after["demand_units"] - before["demand_units"], after["gross_profit_eur"] - before["gross_profit_eur"]
    with right:
        st.markdown("#### Projected outcome")
        k1, k2 = st.columns(2)
        k1.metric("Forecast demand (units/day)", f"{after['demand_units']:.1f}", f"{dd:+.1f} vs current")
        k2.metric("Gross profit (€/day)", fmt_eur(after["gross_profit_eur"]), f"{dp:+.2f} € vs current")
        k3, k4 = st.columns(2)
        k3.metric("Revenue (€/day)", fmt_eur(after["revenue_eur"]), f"{after['revenue_eur']-before['revenue_eur']:+.2f} € vs current")
        k4.metric("Stockout risk", f"{after['stockout_risk_pct']:.1f}%", f"{after['stockout_risk_pct']-before['stockout_risk_pct']:+.1f} pp vs current", delta_color="inverse")
        st.write(f"80% prediction interval for demand: **{after['demand_low']:.0f} to {after['demand_high']:.0f} units/day**.")
        tab_cmp, tab_why, tab_fb = st.tabs(["Comparison", "Why the forecast moves", "Agent feedback"])
        with tab_cmp:
            rows = [("Forecast demand (units/day)", "demand_units"), ("Sales fulfilled (units/day)", "sales_units"), ("Lost sales (units/day)", "lost_sales_units"),
                    ("Gross profit (€/day)", "gross_profit_eur"), ("Gross margin (%)", "margin_pct"), ("Safety buffer (units)", "safety_stock_units"),
                    ("Stockout risk (%)", "stockout_risk_pct"), ("Waste cost proxy (€/day)", "waste_cost_eur"), ("Staffing pressure index", "staffing_pressure"),
                    ("Compliance flags", "compliance_flags")]
            st.dataframe(pd.DataFrame([{"Indicator": n, "Current": float(before[k]), "Trial": float(after[k]), "Change": float(after[k]) - float(before[k])} for n, k in rows]).round(2),
                         use_container_width=True, hide_index=True)
        with tab_why:
            br_before, br_after = explain_profile(baseline), explain_profile(simulated)
            cmp_ = br_before[["Driver", "Contribution (units/day)"]].merge(br_after[["Driver", "Contribution (units/day)"]], on="Driver", suffixes=(" current", " trial"))
            cmp_["Change (units/day)"] = cmp_["Contribution (units/day) trial"] - cmp_["Contribution (units/day) current"]
            cmp_ = cmp_.reindex(cmp_["Change (units/day)"].abs().sort_values(ascending=False).index)
            st.dataframe(cmp_.round(2), use_container_width=True, hide_index=True)
        with tab_fb:
            fb = []
            if sim_price != baseline["price"]:
                fb.append(("Pricing", f"{core.OWN_NAME} price moves {((sim_price/baseline['price'])-1)*100:+.1f}%. With an elasticity of {LEVERS['elasticity']:.2f} ({LEVERS['elasticity_source']}), demand changes {dd:+.1f} units/day and gross profit {dp:+.2f} €/day."))
            else:
                fb.append(("Pricing", f"Price is unchanged. Demand changes {dd:+.1f} units/day because of the other inputs."))
            cb, ca = core.competitor_prices(baseline), core.competitor_prices(simulated)
            cheap_b, cheap_a = min(cb, key=cb.get), min(ca, key=ca.get)
            fb.append(("Competition", f"Cheapest competitor: {cheap_b} at €{cb[cheap_b]:.2f} now, {cheap_a} at €{ca[cheap_a]:.2f} in the trial. {core.OWN_NAME} sits {(sim_price/ca[cheap_a]-1)*100:+.1f}% against the cheapest and {(sim_price/simulated['competitor_price']-1)*100:+.1f}% against the competitor average."))
            if sim_partner != bool(baseline["partner_active"]) or sim_shelf != round(baseline["shelf_share_pl"]) or sim_pl_share != round(baseline["pl_share"]):
                fb.append(("Private label & partnerships", f"Private-label shelf share {sim_shelf}%, sales share {sim_pl_share}%, partnership deal {'on' if sim_partner else 'off'}. The model's response to these inputs is small in this dataset."))
            fb.append(("Inventory", f"A delay of {sim_delay:.1f} h gives an expected fill rate of {MODEL.fill_rate(sim_delay)*100:.0f}%. Stockout risk moves from {before['stockout_risk_pct']:.1f}% to {after['stockout_risk_pct']:.1f}%."))
            if after["waste_cost_eur"] - before["waste_cost_eur"] > 0.01:
                fb.append(("Freshness & waste", f"Waste cost proxy rises by €{after['waste_cost_eur']-before['waste_cost_eur']:.2f}/day."))
            if abs(after["staffing_pressure"] - before["staffing_pressure"]) > 0.01:
                fb.append(("Workforce", f"Staffing pressure moves from {before['staffing_pressure']:.2f} to {after['staffing_pressure']:.2f}."))
            if after["compliance_risk"]:
                fb.append(("Compliance", f"{after['compliance_flags']} active flag(s): {after['agent_reports']['compliance_tasks']['detail']}."))
            if after["status"] != before["status"]:
                fb.append(("Overall risk", f"Status changes from {before['status']} to {after['status']}."))
            for t, m in fb:
                st.markdown(f"**{t}:** {m}")

if selected_page == "📊 9-agent dashboard":
    all_metrics = {c: calculate_metrics(c, current_profile(c), keep_log=True) for c in CATEGORIES}
    df = pd.DataFrame([{k: v for k, v in m.items() if k not in ("agent_reports", "messages")} for m in all_metrics.values()])
    total_revenue, total_profit = float(df["revenue_eur"].sum()), float(df["gross_profit_eur"].sum())
    total_sales, mean_stockout, mean_margin = float(df["sales_units"].sum()), float(df["stockout_risk_pct"].mean()), float(df["margin_pct"].mean())
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
          <div>
            <div style="font-size:1.75rem;font-weight:800;letter-spacing:-.04em;color:var(--text-color)">👋 {greeting}, Retail Manager</div>
            <div style="font-size:.94rem;color:var(--rb-muted);margin-top:.25rem">Forecasts for {ASOF.strftime('%d %b %Y')} from a model trained on {MODEL.n_train:,} category-days of earlier history.</div>
          </div>
          <div style="padding:.55rem .8rem;border:1px solid rgba(98,84,199,.18);border-radius:12px;background:var(--secondary-background-color);font-size:.82rem;font-weight:650">● Workspace online</div>
        </div>
        """, unsafe_allow_html=True)
    if pending_events:
        st.markdown(f"<div style='padding:.55rem .8rem;margin-bottom:.8rem;border-radius:12px;border:1px solid rgba(220,86,86,.35);font-size:.85rem;font-weight:650'>🔔 {len(pending_events)} scenario(s) await manager review ({critical_count} critical). Open the Decision center.</div>", unsafe_allow_html=True)

    kpi_cols = st.columns(5)
    kpis = [("Forecast Revenue", fmt_eur(total_revenue), "Across monitored categories", "💜"),
            ("Expected Units Sold", f"{total_sales:,.0f}", "Units/day after supply limits", "🟠"),
            ("Forecast Gross Profit", fmt_eur(total_profit), f"Average margin {mean_margin:.1f}%", "🟢"),
            ("Average Stockout Risk", f"{mean_stockout:.1f}%", "Unfulfilled demand from delivery delay", "🔴"),
            ("Pending Decisions", f"{len(pending_events)}", f"{critical_count} critical alert(s)", "🔵")]
    for col, (label, value, note, emoji) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
                <div style="min-height:118px;padding:1rem;border:1px solid rgba(120,120,150,.16);border-radius:15px;background:var(--secondary-background-color);box-shadow:0 4px 16px rgba(40,35,90,.045)">
                  <div style="display:flex;align-items:center;gap:.5rem;font-size:.82rem;color:var(--rb-muted);font-weight:650"><span style="font-size:1.15rem">{emoji}</span>{label}</div>
                  <div style="font-size:1.55rem;font-weight:800;letter-spacing:-.035em;color:var(--rb-purple);margin:.45rem 0 .25rem">{value}</div>
                  <div style="font-size:.75rem;color:var(--rb-muted)">{note}</div>
                </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

    agents_col, category_col, insights_col = st.columns([1.05, 1.35, 1.15], gap="medium")
    agent_items = [
        ("🧠", "Demand & XAI", "Demand forecast with 80% interval", "demand_forecast"),
        ("📦", "Inventory safety", "Expected unfulfilled demand from delivery delay", "inventory_risk"),
        ("🏷️", "Commercial intelligence", "Own price vs competitors and private label", "commercial_intelligence"),
        ("🚚", "Replenishment", "Safety buffer sized from the demand interval", "replenishment"),
        ("🌿", "Freshness & waste", "Waste cost from shelf life, heat and cold chain", "freshness_waste"),
        ("👥", "Workforce planning", "Traffic load against staff capacity", "workforce_planning"),
        ("🎯", "Promotion analysis", "Campaign state and price position", "promotion_display"),
        ("💶", "Profitability", "Revenue, margin and gross profit", "profitability"),
        ("🛡️", "Compliance & tasks", "Shelf, cold-chain, queue and staffing flags", "compliance_tasks"),
    ]
    if "dashboard_selected_agent" not in st.session_state or st.session_state.dashboard_selected_agent not in [a[1] for a in agent_items] + [None]:
        st.session_state.dashboard_selected_agent = "Demand & XAI"

    def _fmt_agent(rep: dict[str, Any]) -> str:
        v = rep["value"]
        if rep["unit"] == "%":
            return f"{float(v):.1f}%"
        if rep["unit"] == "EUR/day":
            return fmt_eur(float(v))
        if rep["unit"] in ("units/day", "units"):
            return f"{float(v):,.1f} {rep['unit']}"
        if isinstance(v, (int, float, np.floating, np.integer)):
            return f"{float(v):,.2f}"
        return str(v)

    with agents_col:
        st.markdown("### 9 AI agents")
        st.caption("Click an agent to open it, click again to close.")
        for i in range(0, len(agent_items), 3):
            btn_cols = st.columns(3, gap="small")
            for j, (icon, title, description, _name) in enumerate(agent_items[i:i + 3]):
                with btn_cols[j]:
                    is_selected = st.session_state.dashboard_selected_agent == title
                    if st.button(icon, key=f"agent_icon_{i+j}", help=f"{title}: {description}", use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state.dashboard_selected_agent = None if is_selected else title
                        st.rerun()
                    st.markdown(f"<div style='text-align:center;font-size:.66rem;line-height:1.15;font-weight:700;margin:-.35rem 0 .5rem'>{title}</div>", unsafe_allow_html=True)
        sel_title = st.session_state.dashboard_selected_agent
        if sel_title is None:
            overview = []
            for icon, title, description, name in agent_items:
                sts = [all_metrics[c]["agent_reports"][name]["status"] for c in CATEGORIES]
                worst = "alert" if "alert" in sts else "watch" if "watch" in sts else "ok"
                overview.append({"Agent": f"{icon} {title}", "Status": STATUS_ICON[worst], "Categories to check": sum(s_ != "ok" for s_ in sts)})
            st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)
        else:
            sel = next(a for a in agent_items if a[1] == sel_title)
            st.markdown(f"<div style='padding:.7rem;border:1px solid rgba(102,86,207,.25);border-radius:12px;background:var(--secondary-background-color)'><div style='font-weight:800;font-size:.86rem'>{sel[1]}</div><div style='font-size:.72rem;color:var(--rb-muted);margin-top:.2rem'>{sel[2]}</div></div>", unsafe_allow_html=True)
            agent_rows = []
            for c in CATEGORIES:
                rep = all_metrics[c]["agent_reports"][sel[3]]
                agent_rows.append({"Category": c, "": STATUS_ICON.get(rep["status"], ""), rep["label"]: _fmt_agent(rep)})
            st.dataframe(pd.DataFrame(agent_rows), use_container_width=True, hide_index=True)
            with st.expander("What the agent saw"):
                for c in CATEGORIES:
                    st.markdown(f"**{c}:** {all_metrics[c]['agent_reports'][sel[3]]['detail']}")

    with category_col:
        st.markdown("### Category performance")
        st.caption("Revenue, margin and operating status")
        max_rev = max(float(df["revenue_eur"].max()), 1.0)
        colors = {"Birra": "#ff8a24", "Prodotti da Forno": "#6656cf", "Ortofrutta": "#28a66a", "Detersivi": "#5687f5"}
        for _, row in df.iterrows():
            category, revenue, profit, margin, status = str(row["category"]), float(row["revenue_eur"]), float(row["gross_profit_eur"]), float(row["margin_pct"]), str(row["status"])
            bar_width = max(5, int(100 * revenue / max_rev))
            status_color = "#dc5656" if status.lower() == "critical" else "#ed9b2d" if status.lower() == "warning" else "#28a66a"
            st.markdown(f"""
                <div style="padding:.68rem 0;border-bottom:1px solid rgba(120,120,150,.13)">
                  <div style="display:flex;justify-content:space-between;gap:.4rem;font-size:.82rem;font-weight:700"><span>{category}</span><span style="color:var(--rb-muted)">{margin:.1f}% margin</span></div>
                  <div style="font-size:.8rem;color:var(--rb-muted);margin:.18rem 0 .4rem">{fmt_eur(revenue)} revenue · {fmt_eur(profit)} profit/day</div>
                  <div style="height:6px;border-radius:99px;background:rgba(120,120,150,.14);overflow:hidden"><div style="height:6px;width:{bar_width}%;border-radius:99px;background:{colors.get(category, '#6254c7')}"></div></div>
                  <div style="font-size:.7rem;color:{status_color};margin-top:.3rem;font-weight:700">● {status}</div>
                </div>""", unsafe_allow_html=True)

    with insights_col:
        st.markdown("### Top insights & recommendations")
        st.caption("Prioritized signals from today's forecasts and agent outputs")
        worst_stock = df.sort_values("stockout_risk_pct", ascending=False).iloc[0]
        best_profit = df.sort_values("gross_profit_eur", ascending=False).iloc[0]
        most_waste = df.sort_values("waste_cost_eur", ascending=False).iloc[0]
        top_gap = df.sort_values("price_gap_pct", ascending=False).iloc[0]
        top_staff = df.sort_values("staffing_pressure", ascending=False).iloc[0]
        wp = current_profile(str(worst_stock["category"]))
        insights = [
            ("⚠️" if float(worst_stock["stockout_risk_pct"]) >= 8 else "✅",
             (f"{worst_stock['category']} has the highest stockout risk ({float(worst_stock['stockout_risk_pct']):.1f}%)." if float(worst_stock["stockout_risk_pct"]) >= 8
              else f"Stockout risk is low in every category (highest: {worst_stock['category']}, {float(worst_stock['stockout_risk_pct']):.1f}%)."), f"Delivery delay of {wp['delay_hrs']:.1f} h implies a fill rate of {MODEL.fill_rate(wp['delay_hrs'])*100:.0f}% based on history."),
            ("↗️", f"{best_profit['category']} leads forecast daily gross profit.", f"Estimated at {fmt_eur(float(best_profit['gross_profit_eur']))} per day."),
            ("♻️", f"{most_waste['category']} has the highest waste-cost proxy ({fmt_eur(float(most_waste['waste_cost_eur']))}).", "Waste rates are category assumptions; replace them with store data."),
            ("🏷️", f"{top_gap['category']} is priced {float(top_gap['price_gap_pct']):+.1f}% against competitors.", "Price effects use the elasticity set in the sidebar."),
        ]
        if float(top_staff["staffing_pressure"]) >= 0.5:
            insights.append(("👥", f"{top_staff['category']} shows staffing pressure of {float(top_staff['staffing_pressure']):.2f}.", "Traffic is high relative to staff capacity."))
        for emoji, headline, detail in insights:
            st.markdown(f"""
                <div style="display:flex;gap:.55rem;padding:.72rem .65rem;margin:.45rem 0;border:1px solid rgba(120,120,150,.14);border-radius:12px;background:var(--secondary-background-color)">
                  <div style="font-size:1.15rem">{emoji}</div><div><div style="font-size:.8rem;font-weight:700">{headline}</div><div style="font-size:.72rem;color:var(--rb-muted);margin-top:.25rem">{detail}</div></div>
                </div>""", unsafe_allow_html=True)

    forecast_col, sales_col, assistant_col = st.columns([1.05, 1.35, 1.15], gap="medium")
    with forecast_col:
        st.markdown("### Demand forecast")
        st.caption("Today's forecast with 80% interval, by category")
        fdf = df[["category", "demand_units", "demand_low", "demand_high"]].rename(columns={"category": "Category", "demand_units": "Forecast", "demand_low": "Low", "demand_high": "High"})
        if ALTAIR:
            base = alt.Chart(fdf).encode(x=alt.X("Category:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0, labelLimit=100)))
            bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                y=alt.Y("Forecast:Q", title="Units / day"),
                color=alt.Color("Category:N", scale=alt.Scale(domain=["Birra", "Prodotti da Forno", "Ortofrutta", "Detersivi"], range=["#ff8a24", "#6656cf", "#28a66a", "#5687f5"]), legend=None),
                tooltip=["Category", alt.Tooltip("Forecast:Q", format=",.1f"), alt.Tooltip("Low:Q", format=",.0f"), alt.Tooltip("High:Q", format=",.0f")])
            err = base.mark_rule(color="#22243a", size=2).encode(y="Low:Q", y2="High:Q")
            st.altair_chart((bars + err).properties(height=245), use_container_width=True)
        else:
            st.dataframe(fdf, use_container_width=True, hide_index=True)
    with sales_col:
        st.markdown("### Sales & demand overview")
        st.caption("Forecast demand compared with sales after supply limits")
        trend_df = df[["category", "demand_units", "sales_units"]].melt(id_vars="category", var_name="Metric", value_name="Units per day")
        trend_df["Metric"] = trend_df["Metric"].map({"demand_units": "Forecast demand", "sales_units": "Expected sales"})
        if ALTAIR:
            st.altair_chart(alt.Chart(trend_df).mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5).encode(
                x=alt.X("category:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=120)),
                y=alt.Y("Units per day:Q", title="Units / day", scale=alt.Scale(zero=True)),
                color=alt.Color("Metric:N", title=None, scale=alt.Scale(domain=["Forecast demand", "Expected sales"], range=["#ff8a24", "#6656cf"])),
                tooltip=["category", "Metric", alt.Tooltip("Units per day:Q", format=",.1f")]).properties(height=245), use_container_width=True)
        else:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)
    with assistant_col:
        st.markdown("### 💜 AI Manager Assistant")
        st.caption("Ask about a decision, scenario or what-if.")
        assistant_question = st.text_area("Ask the Retail Brain assistant", placeholder="e.g. What if we increase Birra price by 10%?",
                                          label_visibility="collapsed", key="dashboard_assistant_question", height=110)
        if st.button("Ask assistant →", type="primary", use_container_width=True, key="dashboard_assistant_submit"):
            if assistant_question.strip():
                st.session_state.assistant_prompt = assistant_question.strip()
                st.session_state.retail_brain_page = "💬 AI manager assistant"
                st.rerun()
            else:
                st.warning("Enter a question or scenario first.")

    actual = DF[DF["Date"].dt.normalize() == ASOF].set_index("Category")["Demand_QTY"].to_dict()
    with st.expander("Detailed table: forecast, interval and recorded demand"):
        display = df[["category", "demand_units", "demand_low", "demand_high", "sales_units", "lost_sales_units", "revenue_eur", "gross_profit_eur", "margin_pct", "stockout_risk_pct", "waste_cost_eur", "status"]].copy()
        display.insert(4, "Recorded demand", display["category"].map(actual))
        display.columns = ["Category", "Forecast (units/day)", "Low (80%)", "High (80%)", "Recorded demand", "Sales (units/day)", "Lost sales (units/day)", "Revenue/day (€)", "Gross profit/day (€)", "Margin (%)", "Stockout risk (%)", "Waste cost proxy (€)", "Status"]
        for c in display.columns[1:12]:
            display[c] = pd.to_numeric(display[c], errors="coerce").round(2 if "€" in c else 1)
        st.dataframe(style_status_table(display), use_container_width=True, hide_index=True)
    with st.expander("Category inputs and agent message log (event bus)"):
        cat_for_cards = st.selectbox("Category", CATEGORIES, key="dashboard_category")
        profile = current_profile(cat_for_cards)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.write(f"**{core.OWN_NAME}:** {fmt_eur(profile['price'])}")
            for _n, _p in core.competitor_prices(profile).items():
                st.write(f"**{_n}:** {fmt_eur(_p)}")
            st.write(f"**Private label:** {fmt_eur(profile['private_label_price'])}")
        with p2:
            st.write(f"**Traffic:** {profile['traffic']:,.0f} visitors")
            st.write(f"**Temperature:** {profile['temperature']:.1f} °C")
            st.write(f"**Weather / event:** {profile['weather']} / {profile['event']}")
            st.write(f"**Delivery delay:** {profile['delay_hrs']:.1f} h")
            st.write(f"**Campaign / partnership:** {'On' if profile['promo_active'] else 'Off'} / {'On' if profile['partner_active'] else 'Off'}")
        with p3:
            st.write(f"**Staff capacity:** {profile['staff_capacity_pct']:.0f}%")
            st.write(f"**Shelf compliance:** {profile['compliance_pct']:.0f}%")
            st.write(f"**Checkout queue:** {profile['queue_len']:.1f}")
            st.write(f"**Cold-chain failure:** {'Yes' if profile['cold_chain'] else 'No'}")
            st.write(f"**Private label:** {profile['pl_share']:.0f}% of sales, {profile['shelf_share_pl']:.0f}% of shelf")
        st.caption(f"Assumed: unit cost {fmt_eur(profile['unit_cost'])}, shelf life {profile['shelf_life_days']} days, base waste rate {profile['base_waste_rate']*100:.1f}%.")
        st.markdown("**Event-bus messages for this category**")
        st.caption("Each agent subscribes to topics on the event bus and publishes its result for others to pick up.")
        st.dataframe(pd.DataFrame(all_metrics[cat_for_cards]["messages"]), use_container_width=True, hide_index=True)

if selected_page == "🚨 Decision center":
    st.subheader("Scenario queue")
    queue_df = pd.DataFrame()
    if not st.session_state.events:
        st.info("No scenarios generated yet. Use the sidebar scan button to create a queue.")
    else:
        queue_df = pd.DataFrame([{
            "Scenario": e["template"], "Category": e["category"], "Severity": e["severity"], "Status": e["status"].title(),
            "Projected demand delta": round(float(e.get("delta_demand_units", 0)), 1),
            "Projected gross-profit delta (€)": round(float(e.get("delta_gross_profit_eur", 0)), 2), "Created": e["created_at"]}
            for e in st.session_state.events])
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
        m = calculate_metrics(category, current_profile(category))
        impact_rows.append({"Category": category, "Forecast (units/day)": round(float(m["demand_units"]), 1),
                            "Safety buffer (units)": round(float(m["safety_stock_units"]), 1),
                            "Stockout risk (%)": round(float(m["stockout_risk_pct"]), 1),
                            "Gross profit/day (€)": round(float(m["gross_profit_eur"]), 2),
                            "Waste cost proxy (€)": round(float(m["waste_cost_eur"]), 2), "Status": m["status"]})
    impact_df = pd.DataFrame(impact_rows)
    st.dataframe(style_status_table(impact_df), use_container_width=True, hide_index=True)
    csv_data = queue_df.to_csv(index=False).encode("utf-8") if not queue_df.empty else b"Scenario,Category,Status\n"
    st.download_button("⬇️ Download manager action queue (CSV)", data=csv_data, file_name="retail_brain_manager_action_queue.csv", mime="text/csv")
    st.download_button("⬇️ Download projected category impact (CSV)", data=impact_df.to_csv(index=False).encode("utf-8"),
                       file_name="retail_brain_projected_category_impact.csv", mime="text/csv")


if selected_page == "📈 Decisions effects":
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
        st.caption("Approved decisions show the change applied to the profile; rejected and deferred ones show what applying them would have done.")
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
    tab_a, tab_b, tab_c = st.tabs(["Assessment", "Case library", "Audit trail"])
    with tab_a:
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
                    assessment = "Likely beneficial"
                    rationale = "The current comparison shows higher gross profit without an increase in stockout risk or waste cost. Keep monitoring demand and availability."
                    severity = "Positive signal"
                elif negative:
                    assessment = "Potential downside"
                    reasons = []
                    if delta_profit < 0: reasons.append(f"gross profit changes by {delta_profit:+.2f} €/day")
                    if delta_stockout > 5: reasons.append(f"stockout risk rises by {delta_stockout:+.1f} percentage points")
                    if delta_waste > max(1.0, float(impact["before_waste"]) * .15): reasons.append(f"waste-cost proxy rises by {delta_waste:+.2f} €/day")
                    rationale = "The model flags potential trade-offs: " + "; ".join(reasons) + ". Review the assumptions against observed store data."
                    severity = "Review recommended"
                else:
                    assessment = "Mixed or limited effect"
                    rationale = "The indicators do not show a clear overall benefit or material downside. The decision should be evaluated against the business objective and later operational evidence."
                    severity = "Inconclusive signal"
            elif decision == "rejected":
                assessment = "Rejected scenario — avoided impact estimated"
                rationale = f"Had the scenario's changes been applied, the current counterfactual estimates indicate gross profit {delta_profit:+.2f} €/day, demand {delta_demand:+.1f} units/day, stockout risk {delta_stockout:+.1f} percentage points, and waste cost {delta_waste:+.2f} €/day versus the comparison profile."
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
                st.success("AI assessment saved to the decision history.")
                st.rerun()
        else:
            st.write("Record a decision on a scenario first; it can then be assessed here.")
    with tab_b:

        if not st.session_state.decision_history:
            st.info("No decisions recorded yet. Review a scenario and record a decision to build a reference set.")
        else:
            hist = pd.DataFrame(st.session_state.decision_history)
            m1, m2, m3 = st.columns(3)
            m1.metric("Recorded decision records", len(hist))
            m2.metric("Unique scenarios with a decision", hist["event_id"].nunique() if "event_id" in hist.columns else len(hist))
            m3.metric("Saved to audit file", "Yes" if st.session_state.get("last_persist_ok", not st.session_state.get("persistence_warning")) else "No")
            st.markdown("#### Decision mix")
            mix = hist["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
            if ALTAIR:
                st.altair_chart(alt.Chart(mix).mark_bar(color="#6254c7").encode(
                    x=alt.X("Count:Q", title="Number of decisions"), y=alt.Y("Decision:N", sort="-x", title=None, axis=alt.Axis(labelAngle=0)),
                    tooltip=["Decision", "Count"]), use_container_width=True)
            else:
                st.dataframe(mix, use_container_width=True, hide_index=True)
            st.markdown("#### Case library")
            cols = ["decision_at", "template", "category", "decision", "severity", "delta_demand_units_at_review", "delta_gross_profit_eur_at_review", "ai_assessment", "ai_assessment_signal", "ai_assessment_rationale", "note"]
            hist_view = hist[[c for c in cols if c in hist.columns]].rename(columns={
                "decision_at": "Decision time", "template": "Scenario", "category": "Category", "decision": "Decision", "severity": "Severity",
                "delta_demand_units_at_review": "Projected demand delta at review", "delta_gross_profit_eur_at_review": "Projected gross-profit delta (€)",
                "ai_assessment": "AI assessment", "ai_assessment_signal": "AI signal", "ai_assessment_rationale": "AI rationale", "note": "Manager decision rationale"})
            st.dataframe(hist_view, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Export decision case library (CSV)", hist_view.to_csv(index=False).encode("utf-8"),
                               file_name="retail_brain_decision_case_library.csv", mime="text/csv")

    with tab_c:
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


AGENT_ACTIONS = {
    "inventory_risk": "Chase the supplier or expedite the delivery",
    "replenishment": "Place the order including the safety buffer",
    "freshness_waste": "Check the cold chain and consider a markdown",
    "workforce_planning": "Add cashier or shelf cover for the peak",
    "compliance_tasks": "Fix the open shelf, queue or cold-chain issues",
    "commercial_intelligence": "Review the price against the cheapest competitor",
    "promotion_display": "Decide whether a campaign is justified",
    "profitability": "Review margin and unit cost",
    "demand_forecast": "Check the forecast against expected traffic",
}


def build_task_table(all_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Turn agent flags, pending scenarios and upcoming events into one working list."""
    auto: dict[str, dict[str, Any]] = {}
    for cat, m in all_metrics.items():
        for name, rep in m["agent_reports"].items():
            if rep["status"] in ("alert", "watch"):
                key = f"{ASOF.date()}|{cat}|{name}"
                auto[key] = {"Priority": "High" if rep["status"] == "alert" else "Medium", "Category": cat,
                             "Task": AGENT_ACTIONS.get(name, rep["label"]), "Why": f"{rep['label']}: {rep['detail']}", "Source": rep["title"].split(". ", 1)[-1]}
    if pending_events:
        auto[f"{ASOF.date()}|all|pending"] = {"Priority": "High" if critical_count else "Medium", "Category": "All",
                                              "Task": "Review pending scenarios", "Why": f"{len(pending_events)} scenario(s) waiting in the Decision center", "Source": "Decision center"}
    up = core.upcoming_events(DF, ASOF)
    for _, r in up.iterrows():
        if r["Starts in (days)"] <= 30:
            auto[f"{ASOF.date()}|all|{r['Event']}"] = {"Priority": "Medium", "Category": "All", "Task": f"Prepare for {r['Event']}",
                                                       "Why": f"Starts in {r['Starts in (days)']} days ({r['From']} to {r['To']}). Check stock, staffing and promotions.", "Source": "Calendar"}
    store = st.session_state.tasks
    for key, row in auto.items():
        if key not in store:
            store[key] = {**row, "Done": False}
    for key in list(store):
        if key.startswith(str(ASOF.date())) and key not in auto and not store[key]["Done"] and not key.startswith("manual|"):
            store.pop(key)
    rows = [{"key": k, **v} for k, v in store.items()]
    for i, t in enumerate(st.session_state.manual_tasks):
        rows.append({"key": f"manual|{i}", "Priority": t["Priority"], "Category": t["Category"], "Task": t["Task"], "Why": "Added by manager", "Source": "Manual", "Done": t["Done"]})
    df_ = pd.DataFrame(rows)
    if df_.empty:
        return df_
    order = {"High": 0, "Medium": 1, "Low": 2}
    df_["_o"] = df_["Priority"].map(order)
    return df_.sort_values(["_o", "key"]).drop(columns="_o").reset_index(drop=True)


def line_chart(data: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None, height: int = 230, fmt: str = ",.2f"):
    if not ALTAIR:
        st.dataframe(data, use_container_width=True, hide_index=True)
        return
    enc = dict(x=alt.X(f"{x}:T", title=None), y=alt.Y(f"{y}:Q", title=title or y, scale=alt.Scale(zero=False)))
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=None, legend=alt.Legend(orient="bottom"))
    st.altair_chart(alt.Chart(data).mark_line(strokeWidth=2).encode(**enc).properties(height=height), use_container_width=True)


if selected_page == "🏬 Store management":
    all_metrics_m = {c: calculate_metrics(c, current_profile(c)) for c in CATEGORIES}
    tab_tasks, tab_price, tab_pl, tab_cust, tab_cal, tab_ops = st.tabs(
        ["Tasks", "Prices & competitors", "Private label & partnerships", "Customers", "Calendar & staffing", "Operations"])

    with tab_tasks:
        tasks_df = build_task_table(all_metrics_m)
        open_n = int((~tasks_df["Done"]).sum()) if not tasks_df.empty else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Open tasks", open_n)
        c2.metric("High priority", int(((tasks_df["Priority"] == "High") & (~tasks_df["Done"])).sum()) if not tasks_df.empty else 0)
        c3.metric("Completed", int(tasks_df["Done"].sum()) if not tasks_df.empty else 0)
        if tasks_df.empty:
            st.write("No open tasks. Every agent reports a normal situation.")
        else:
            edited = st.data_editor(
                tasks_df[["Done", "Priority", "Category", "Task", "Why", "Source"]], hide_index=True, use_container_width=True,
                disabled=["Priority", "Category", "Task", "Why", "Source"], key=f"task_editor_{abs(hash(tuple(tasks_df['key']))) % 10**8}",
                column_config={"Done": st.column_config.CheckboxColumn("Done", width="small"), "Why": st.column_config.TextColumn(width="large")})
            for i, row in edited.reset_index(drop=True).iterrows():
                key = tasks_df.loc[i, "key"]
                if key.startswith("manual|"):
                    st.session_state.manual_tasks[int(key.split("|")[1])]["Done"] = bool(row["Done"])
                elif key in st.session_state.tasks:
                    st.session_state.tasks[key]["Done"] = bool(row["Done"])
        with st.expander("Add a task"):
            ta, tb, tc = st.columns([3, 1.3, 1.3])
            new_task = ta.text_input("Task", key="new_task_text", label_visibility="collapsed", placeholder="e.g. Confirm Saturday rota with the team")
            new_cat = tb.selectbox("Category", ["All"] + CATEGORIES, key="new_task_cat", label_visibility="collapsed")
            new_pri = tc.selectbox("Priority", ["Medium", "High", "Low"], key="new_task_pri", label_visibility="collapsed")
            if st.button("Add task", key="add_task_btn") and new_task.strip():
                st.session_state.manual_tasks.append({"Task": new_task.strip(), "Category": new_cat, "Priority": new_pri, "Done": False})
                st.rerun()

    with tab_price:
        board = core.competitor_board(DF, ASOF)
        cheapest_counts = board["Cheapest competitor"].value_counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("Cheapest competitor", cheapest_counts.index[0], f"in {int(cheapest_counts.iloc[0])} of {len(board)} categories")
        m2.metric(f"{core.OWN_NAME} vs cheapest", f"{board[f'{core.OWN_NAME} vs cheapest (%)'].mean():+.1f}%", help="Average across categories")
        m3.metric(f"{core.OWN_NAME} vs average", f"{board[f'{core.OWN_NAME} vs average (%)'].mean():+.1f}%", help="Average across categories")
        st.dataframe(board, use_container_width=True, hide_index=True)
        pc = st.selectbox("Price history for", CATEGORIES, key="price_hist_cat")
        line_chart(core.price_history(DF, ASOF, pc, 90), "Date", "Price (EUR)", "Retailer", "Shelf price (EUR)")
        st.caption("Last 90 days. Each line is one retailer's shelf price. A gap between Esselunga and the lowest line is the room competitors have to undercut you.")

    with tab_pl:
        pls = core.private_label_summary(DF, ASOF, 30)
        st.dataframe(pls, use_container_width=True, hide_index=True)
        st.caption("Shares are 30-day averages; the chart shows the last 120 days. The last column compares demand per visitor on partnership days with all other days; it is descriptive and based on few deal days.")
        pc2 = st.selectbox("Category", CATEGORIES, key="pl_cat")
        line_chart(core.private_label_series(DF, ASOF, pc2, 120), "Date", "Value", "Measure", "Share (%)")
        pdays = core.partnership_days(DF, ASOF, pc2)
        st.markdown(f"**Partnership deal days for {pc2}** ({len(pdays)} in the history)")
        if pdays.empty:
            st.write("No partnership deal days recorded for this category.")
        else:
            st.dataframe(pdays.assign(Date=pdays["Date"].dt.date), use_container_width=True, hide_index=True)

    with tab_cust:
        cs = core.customer_summary(DF, ASOF, 30)
        st.dataframe(cs, use_container_width=True, hide_index=True)
        cc = st.selectbox("Category", CATEGORIES, key="cust_cat")
        ser = core.customer_series(DF, ASOF, cc, 90)
        x1, x2 = st.columns(2)
        with x1:
            if "Customer_Satisfaction_Score" in ser.columns:
                line_chart(ser, "Date", "Customer_Satisfaction_Score", None, "Satisfaction (1-5)")
        with x2:
            line_chart(ser, "Date", "Checkout_Queue_Avg_Length", None, "Checkout queue (avg length)")
        drv = core.satisfaction_drivers(DF, ASOF)
        if not drv.empty:
            st.markdown("**What moves customer satisfaction**")
            st.dataframe(drv, use_container_width=True, hide_index=True)
            top = drv.iloc[0]
            st.caption(f"Strongest link in the history: {top['Factor'].lower()} (correlation {top['Correlation with satisfaction']:+.2f}). Correlation shows association, not cause.")

    with tab_cal:
        up = core.upcoming_events(DF, ASOF)
        st.markdown("**Upcoming events**")
        if up.empty:
            st.write("No calendar events after the decision day in the data.")
        else:
            st.dataframe(up, use_container_width=True, hide_index=True)
        ev = core.event_effects(DF, ASOF)
        if not ev.empty:
            st.markdown("**What past events did to traffic and demand**")
            st.dataframe(ev, use_container_width=True, hide_index=True)
        wk = core.weekday_profile(DF, ASOF)
        st.markdown("**Weekday pattern and rota planning**")
        if ALTAIR:
            st.altair_chart(alt.Chart(wk).mark_bar(color="#6254c7", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Weekday:N", sort=core.WEEKDAYS, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Traffic index:Q", scale=alt.Scale(zero=False)), tooltip=["Weekday", "Traffic index", "Avg staff capacity (%)"]).properties(height=220), use_container_width=True)
        spread = float(wk["Traffic index"].max() - wk["Traffic index"].min())
        low = wk.sort_values("Traffic index").head(2)["Weekday"].tolist()
        if spread < 5:
            st.caption(f"Traffic is almost the same on every weekday (spread of {spread:.0f} index points), so the history gives no reason to prefer one day off over another. Plan days off around events, not weekdays.")
        else:
            st.caption(f"{low[0]} and {low[1]} have the lowest traffic, which makes them the natural days for lighter rotas or days off.")
        st.dataframe(wk, use_container_width=True, hide_index=True)

    with tab_ops:
        ccs = core.cold_chain_summary(DF, ASOF, 90)
        st.dataframe(ccs, use_container_width=True, hide_index=True)
        oc = st.selectbox("Category", CATEGORIES, key="ops_cat")
        ops = core.operations_series(DF, ASOF, 90)
        ops = ops[ops["Category"] == oc]
        o1, o2, o3 = st.columns(3)
        with o1:
            line_chart(ops, "Date", "Logistics_Delivery_Delay_Hrs", None, "Delivery delay (hours)", 200)
        with o2:
            line_chart(ops, "Date", "Shelf_Sorting_Compliance_Pct", None, "Shelf compliance (%)", 200)
        with o3:
            if "Utility_Cost_Today_EUR" in ops.columns:
                line_chart(ops, "Date", "Utility_Cost_Today_EUR", None, "Utility cost (EUR/day)", 200)



# -----------------------------------------------------------------------------
# AI manager assistant: built-in engine, or Claude / ChatGPT grounded in the app's own numbers
# -----------------------------------------------------------------------------
LLM_ENGINES = {"Built-in assistant": None, "Claude (Anthropic)": "anthropic", "ChatGPT (OpenAI)": "openai"}
LLM_DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o"}
LLM_SECRET_NAME = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
LLM_SYSTEM_PROMPT = (
    "You are the Retail Brain assistant for the manager of an Esselunga store. Answer using ONLY the figures in the CONTEXT JSON. "
    "If something is not in the context, say you do not have it; never invent numbers, prices, competitors or dates. "
    "When `what_if_result` is present, report those numbers exactly and do not recompute them. "
    "If a price effect relies on an assumed elasticity (see model.levers), say so in one short clause. "
    "Competitors are Lidl, Conad and Carrefour; name them rather than saying 'competitors' when the context has per-competitor prices. "
    "Be concise (under 180 words), practical and plain. Use short bullets for lists. Suggest the relevant app page when helpful."
)


def _secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name)
        return str(v) if v else None
    except Exception:
        return None


def call_llm(provider: str, api_key: str, model: str, system: str, history: list[dict[str, str]], question: str) -> str:
    """Single request to the provider's chat API. Raises RuntimeError with a readable message on failure."""
    import requests
    msgs = [{"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")][-8:]
    msgs.append({"role": "user", "content": question})
    try:
        if provider == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                              json={"model": model, "max_tokens": 700, "system": system, "messages": msgs}, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"Claude API error {r.status_code}: {r.json().get('error', {}).get('message', r.text[:200])}")
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
                          json={"model": model, "max_tokens": 700, "messages": [{"role": "system", "content": system}] + msgs}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI API error {r.status_code}: {r.json().get('error', {}).get('message', r.text[:200])}")
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the {provider} API: {exc}") from exc


if selected_page == "💬 AI manager assistant":
    def _parse_what_if(q: str) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        down = ("decrease", "lower", "reduce", "cut", "drop", "fall", "discount")
        def sign(m: re.Match) -> float:
            seg = q[max(0, m.start() - 30):m.end() + 6]
            return -1.0 if any(t in seg for t in down) else 1.0
        named = False
        for n, _ in core.COMPETITORS:
            m = re.search(rf"{n.lower()}\D{{0,30}}?(\d+(?:\.\d+)?)\s*%", q)
            if m:
                changes[f"{core.COMP_KEYS[n]}_pct"] = sign(m) * float(m.group(1))
                named = True
        for word, key in (("competitor", "competitor_price_pct"), ("traffic", "traffic_pct"), ("visitors", "traffic_pct")):
            m = re.search(rf"{word}\D{{0,30}}?(\d+(?:\.\d+)?)\s*%", q)
            if m:
                changes[key] = sign(m) * float(m.group(1))
        if not named and "competitor_price_pct" not in changes:
            m = re.search(r"price\D{0,30}?(\d+(?:\.\d+)?)\s*%", q)
            if m:
                changes["price_pct"] = sign(m) * float(m.group(1))
        m = re.search(r"delay\D{0,20}?(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours)", q)
        if m:
            changes["delay_hrs_delta"] = float(m.group(1))
        if "promotion" in q or "campaign" in q:
            changes["promo_active_set"] = 0 if any(t in q for t in ["stop", "end", "remove", "without"]) else 1
        if "partnership" in q:
            changes["partner_active_set"] = 0 if any(t in q for t in ["stop", "end", "remove", "without"]) else 1
        return changes

    def _assistant_context() -> dict[str, Any]:
        cats = []
        for category in CATEGORIES:
            profile = current_profile(category)
            cats.append({"category": category, "profile": profile, "metrics": calculate_metrics(category, profile)})
        return {"categories": cats, "recent_scenarios": [{k: e.get(k) for k in ["category", "template", "status", "severity", "created_at", "change_summary", "decision_note", "ai_assessment_rationale"]} for e in st.session_state.events[:20]],
                "recent_decision_history": st.session_state.decision_history[:20],
                "scenario_count": len(st.session_state.events), "decision_history_count": len(st.session_state.decision_history)}

    def _what_if(category: dict[str, Any], ch: dict[str, Any]) -> dict[str, Any]:
        after_p = core.apply_changes(category["profile"], ch)
        after = calculate_metrics(category["category"], after_p)
        b = category["metrics"]
        return {"category": category["category"], "changes": describe_changes(ch), "before": b, "after": after,
                "elasticity_note": f"{LEVERS['elasticity']:.2f} ({LEVERS['elasticity_source']})" if any("price" in k for k in ch) else ""}

    def _find_category(q: str, cats: list[dict[str, Any]]):
        return next((c for c in cats if c["category"].lower() in q.lower()), None)

    def _llm_context_json(question: str, context: dict[str, Any]) -> str:
        cats_out = []
        for c in context["categories"]:
            p, m = c["profile"], c["metrics"]
            top = explain_profile(p)
            top = top[~top["Type"].isin(["base"])].sort_values("Absolute", ascending=False).head(3)
            cats_out.append({
                "category": c["category"],
                "forecast_units_per_day": round(m["demand_units"], 1), "interval_80": [round(m["demand_low"]), round(m["demand_high"])],
                "sales_units": round(m["sales_units"], 1), "lost_units": round(m["lost_sales_units"], 1),
                "revenue_eur": round(m["revenue_eur"], 2), "gross_profit_eur": round(m["gross_profit_eur"], 2), "margin_pct": round(m["margin_pct"], 1),
                "stockout_risk_pct": round(m["stockout_risk_pct"], 1), "safety_buffer_units": round(m["safety_stock_units"]),
                "waste_cost_eur": round(m["waste_cost_eur"], 2), "staffing_pressure": round(m["staffing_pressure"], 2),
                "compliance_flags": m["compliance_flags"], "status": m["status"],
                "prices_eur": {core.OWN_NAME: round(p["price"], 2), **{n: round(v, 2) for n, v in core.competitor_prices(p).items()}, "Private label": round(p["private_label_price"], 2)},
                "price_gap_vs_competitor_average_pct": round(m["price_gap_pct"], 1),
                "traffic": round(p["traffic"]), "temperature_c": round(p["temperature"], 1), "weather": p["weather"], "event": p["event"],
                "campaign_active": bool(p["promo_active"]), "partnership_active": bool(p["partner_active"]),
                "private_label_sales_share_pct": round(p["pl_share"]), "private_label_shelf_share_pct": round(p["shelf_share_pl"]),
                "delivery_delay_h": p["delay_hrs"], "staff_capacity_pct": round(p["staff_capacity_pct"]), "shelf_compliance_pct": round(p["compliance_pct"]),
                "checkout_queue": round(p["queue_len"], 1), "satisfaction_1_to_5": p.get("satisfaction"),
                "top_forecast_drivers_units": {r["Driver"]: round(r["Contribution (units/day)"], 1) for _, r in top.iterrows()},
            })
        v = MODEL.validation
        ctx = {
            "decision_day": ASOF.date().isoformat(), "categories": cats_out,
            "model": {"type": MODEL.kind, "mape_pct": round(v["overall"]["Retail Brain model"]["MAPE %"], 1),
                      "baseline_mape_pct": round(v["overall"]["Baseline: category avg rate x traffic"]["MAPE %"], 1),
                      "interval_coverage": round(v["coverage"], 2),
                      "levers": {"price_elasticity": f"{LEVERS['elasticity']:.2f} ({LEVERS['elasticity_source']})", "campaign_uplift": f"{LEVERS['promo_uplift']*100:+.0f}% ({LEVERS['promo_source']})"}},
            "pending_scenarios": [{k: e.get(k) for k in ["category", "template", "severity", "change_summary", "ai_recommendation"]} for e in st.session_state.events if e.get("status") == "pending"][:6],
            "recent_decisions": [{k: h.get(k) for k in ["category", "template", "decision", "note", "decision_at"]} for h in st.session_state.decision_history[:5]],
            "upcoming_events": core.upcoming_events(DF, ASOF).to_dict("records"),
        }
        cat = _find_category(question, context["categories"])
        ch = _parse_what_if(question.lower())
        if cat is not None and ch:
            wi = _what_if(cat, ch)
            ctx["what_if_result"] = {"category": wi["category"], "changes": wi["changes"], "elasticity_used": wi["elasticity_note"],
                                     "demand_before": round(wi["before"]["demand_units"], 1), "demand_after": round(wi["after"]["demand_units"], 1),
                                     "interval_80_after": [round(wi["after"]["demand_low"]), round(wi["after"]["demand_high"])],
                                     "gross_profit_before": round(wi["before"]["gross_profit_eur"], 2), "gross_profit_after": round(wi["after"]["gross_profit_eur"], 2),
                                     "stockout_risk_before_pct": round(wi["before"]["stockout_risk_pct"], 1), "stockout_risk_after_pct": round(wi["after"]["stockout_risk_pct"], 1),
                                     "status_before": wi["before"]["status"], "status_after": wi["after"]["status"]}
        return json.dumps(ctx, default=str)

    def _local_manager_answer(question: str, context: dict[str, Any]) -> str:
        q = question.lower()
        cats = context["categories"]
        category = _find_category(q, cats)
        if any(t in q for t in ["accuracy", "accurate", "validate", "validation", "reliable", "how good", "error"]):
            v = MODEL.validation["overall"]
            mm, bb = v["Retail Brain model"], v["Baseline: category avg rate x traffic"]
            return (f"### Forecast accuracy on a held-out period\nOn {MODEL.validation['test_days']} unseen days the model's MAPE is **{mm['MAPE %']:.1f}%** (MAE {mm['MAE (units)']:.1f} units). "
                    f"A baseline of category-average demand per visitor times traffic scores **{bb['MAPE %']:.1f}%**. The 80% interval covered **{MODEL.validation['coverage']*100:.0f}%** of outcomes. "
                    "Store traffic carries most of the signal. See **Data & validation** for the full table.")
        if any(t in q for t in ["why", "driver", "explain", "because"]):
            c = category or max(cats, key=lambda x: x["metrics"]["demand_units"])
            br = explain_profile(c["profile"])
            top = br[~br["Type"].isin(["base"])].sort_values("Absolute", ascending=False).head(4)
            lines = [f"### Why the {c['category']} forecast is {c['metrics']['demand_units']:.0f} units/day"]
            lines += [f"- **{r['Driver']}**: {r['Contribution (units/day)']:+.1f} units/day" for _, r in top.iterrows()]
            return "\n".join(lines)
        if any(t in q for t in ["previous decision", "decision history", "past decision", "what did we decide", "history"]):
            if not context["recent_decision_history"] and not context["recent_scenarios"]:
                return "No decisions or scenarios are recorded yet. Use the **Decision center** to review one."
            lines = ["### Recent decisions and scenarios", f"Scenarios this session: **{context['scenario_count']}** · Decision records: **{context['decision_history_count']}**"]
            lines += [f"- **{e.get('category')} · {e.get('template')}**: {e.get('status')}, severity {e.get('severity')}. {e.get('ai_assessment_rationale') or e.get('decision_note') or ''}" for e in context["recent_scenarios"][:8]]
            return "\n".join(lines)
        ch = _parse_what_if(q)
        if category is not None and (ch or any(t in q for t in ["what if", "what-if", "if i", "if we"])):
            ch = ch or {"price_pct": 5.0}
            wi = _what_if(category, ch)
            b, a = wi["before"], wi["after"]
            note = f"\n\nPrice response uses an elasticity of **{wi['elasticity_note']}**." if wi["elasticity_note"] else ""
            return (f"### What-if for {category['category']}: {wi['changes']}\n"
                    f"- Demand: **{b['demand_units']:.1f} → {a['demand_units']:.1f}** units/day (80% interval {a['demand_low']:.0f} to {a['demand_high']:.0f})\n"
                    f"- Gross profit: **€{b['gross_profit_eur']:.2f} → €{a['gross_profit_eur']:.2f}** per day\n"
                    f"- Stockout risk: **{b['stockout_risk_pct']:.1f}% → {a['stockout_risk_pct']:.1f}%**; waste proxy €{b['waste_cost_eur']:.2f} → €{a['waste_cost_eur']:.2f}\n"
                    f"- Status: **{b['status']} → {a['status']}**{note}")
        if any(t in q for t in ["competitor", "lidl", "conad", "carrefour", "cheapest"]):
            board = core.competitor_board(DF, ASOF)
            lines = [f"### Prices on {ASOF.date().isoformat()}", "| Category | Esselunga | Lidl | Conad | Carrefour | Private label | Cheapest |", "|---|---:|---:|---:|---:|---:|---|"]
            lines += [f"| {r['Category']} | €{r['Esselunga']:.2f} | €{r['Lidl']:.2f} | €{r['Conad']:.2f} | €{r['Carrefour']:.2f} | €{r['Private label']:.2f} | {r['Cheapest competitor']} |" for _, r in board.iterrows()]
            return "\n".join(lines)
        if any(t in q for t in ["private label", "partnership", "shelf share"]):
            pls = core.private_label_summary(DF, ASOF, 30)
            lines = ["### Private label and partnerships (last 30 days)"]
            lines += [f"- **{r['Category']}**: {r['Private-label sales share (%)']}% of sales on {r['Private-label shelf share (%)']}% of shelf; private label is {r['Private label cheaper by (%)']}% cheaper; {r['Partnership days (last period)']} partnership day(s)." for _, r in pls.iterrows()]
            return "\n".join(lines)
        if any(t in q for t in ["task", "to do", "todo", "what should i do"]):
            tdf = build_task_table({c["category"]: c["metrics"] for c in cats})
            if tdf.empty:
                return "No open tasks right now."
            lines = ["### Open tasks"] + [f"- **{r['Priority']}** · {r['Category']}: {r['Task']}" for _, r in tdf[~tdf["Done"]].head(8).iterrows()]
            return "\n".join(lines)
        if any(t in q for t in ["dashboard", "summarize", "summary", "overall", "all categories", "how are we doing"]):
            lines = [f"### Snapshot for {ASOF.date().isoformat()}", "| Category | Forecast/day | Gross profit/day | Stockout risk | Waste/day | Status |", "|---|---:|---:|---:|---:|---|"]
            lines += [f"| {c['category']} | {c['metrics']['demand_units']:.1f} | €{c['metrics']['gross_profit_eur']:.2f} | {c['metrics']['stockout_risk_pct']:.1f}% | €{c['metrics']['waste_cost_eur']:.2f} | {c['metrics']['status']} |" for c in cats]
            return "\n".join(lines)
        if category is None and any(t in q for t in ["worst", "risk", "critical", "stockout"]):
            category = max(cats, key=lambda c: c["metrics"]["stockout_risk_pct"])
        if category is None and any(t in q for t in ["best", "most profitable", "highest profit"]):
            category = max(cats, key=lambda c: c["metrics"]["gross_profit_eur"])
        if category:
            p, m = category["profile"], category["metrics"]
            comps = ", ".join(f"{n} €{v:.2f}" for n, v in core.competitor_prices(p).items())
            return (f"### {category['category']} on {ASOF.date().isoformat()}\n"
                    f"- {core.OWN_NAME} **€{p['price']:.2f}** against {comps} ({m['price_gap_pct']:+.1f}% vs the competitor average).\n"
                    f"- Forecast **{m['demand_units']:.1f}** units/day (80% interval {m['demand_low']:.0f} to {m['demand_high']:.0f}); sales **{m['sales_units']:.1f}**; lost **{m['lost_sales_units']:.1f}**.\n"
                    f"- Revenue **€{m['revenue_eur']:.2f}**, gross profit **€{m['gross_profit_eur']:.2f}**, margin **{m['margin_pct']:.1f}%**.\n"
                    f"- Stockout risk **{m['stockout_risk_pct']:.1f}%**, safety buffer **{m['safety_stock_units']:.0f}** units, waste proxy **€{m['waste_cost_eur']:.2f}**, staffing pressure **{m['staffing_pressure']:.2f}**, compliance flags **{m['compliance_flags']}**.")
        return ("I can summarize the dashboard, describe a category, compare competitor prices, report private-label and partnership figures, list tasks, explain a forecast, "
                "report accuracy, review decisions, or compute a what-if such as **‘What if Lidl cuts Birra price by 10%?’** or **‘What if Ortofrutta delivery delay is 4 hours?’**.")

    welcome = "Hello! Ask me about the dashboard, a category, competitor prices, private label, tasks, a forecast, or a what-if."
    if "manager_ai_messages" not in st.session_state:
        st.session_state.manager_ai_messages = [{"role": "assistant", "content": welcome}]
    st.session_state.setdefault("assistant_engine", "Built-in assistant") if False else None
    if "assistant_engine" not in st.session_state:
        st.session_state["assistant_engine"] = "Built-in assistant"
    top_a, top_b, top_c = st.columns([2, 1, 1])
    with top_a:
        st.selectbox("Answer engine", list(LLM_ENGINES), key="assistant_engine",
                     help="Claude and ChatGPT answer from the same figures the app shows. Without a key the built-in engine is used.")
    provider = LLM_ENGINES[st.session_state["assistant_engine"]]
    api_key, model_name = None, None
    if provider:
        with st.expander("Connection", expanded=not (_secret(LLM_SECRET_NAME[provider]) or st.session_state.get(f"key_{provider}"))):
            model_name = st.text_input("Model", value=st.session_state.get(f"model_{provider}", LLM_DEFAULT_MODEL[provider]), key=f"model_{provider}")
            secret = _secret(LLM_SECRET_NAME[provider])
            if secret:
                st.write(f"✅ API key found in the app secrets ({LLM_SECRET_NAME[provider]}).")
                api_key = secret
            else:
                api_key = st.text_input(f"{'Anthropic' if provider == 'anthropic' else 'OpenAI'} API key", type="password", key=f"key_{provider}",
                                        help="Kept in this browser session only. On Streamlit Cloud, store it under App settings → Secrets instead.") or None
            st.caption("Only the aggregated figures shown in the app are sent to the provider, never raw CSV rows.")
    with top_c:
        if st.button("🗑️ Clear chat", key="clear_manager_ai_chat", use_container_width=True):
            st.session_state.manager_ai_messages = [{"role": "assistant", "content": welcome}]
            st.rerun()
    for msg in st.session_state.manager_ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    handoff_prompt = st.session_state.pop("assistant_prompt", None)
    user_prompt = st.chat_input("Ask about a decision, scenario, price or what-if…", key="manager_ai_chat_input") or handoff_prompt
    if user_prompt:
        history = list(st.session_state.manager_ai_messages)
        st.session_state.manager_ai_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        reply, footer = "", ""
        ctx_obj = _assistant_context()
        try:
            if provider and api_key:
                with st.spinner(f"Asking {st.session_state['assistant_engine'].split(' (')[0]}…"):
                    reply = call_llm(provider, api_key, model_name or LLM_DEFAULT_MODEL[provider], LLM_SYSTEM_PROMPT + "\n\nCONTEXT:\n" + _llm_context_json(user_prompt, ctx_obj), history, user_prompt)
            else:
                if provider:
                    footer = "\n\n*No API key set, so the built-in engine answered.*"
                reply = _local_manager_answer(user_prompt, ctx_obj)
        except Exception as exc:
            reply = _local_manager_answer(user_prompt, ctx_obj)
            footer = f"\n\n*The {st.session_state['assistant_engine']} request failed ({str(exc)[:160]}), so the built-in engine answered.*"
        st.session_state.manager_ai_messages.append({"role": "assistant", "content": reply + footer})
        with st.chat_message("assistant"):
            st.markdown(reply + footer)

if selected_page == "🔬 Model explainability":
    tab_glob, tab_loc, tab_trust = st.tabs(["Overall drivers", "One category", "Trust calibration"])
    with tab_glob:
        kind_text = "XGBoost rate model with TreeSHAP explanations" if MODEL.kind == "xgboost" else "category-average baseline (XGBoost unavailable)"
        st.write(f"**Demand estimator:** {kind_text}. It predicts demand per visitor from context, multiplies by store traffic and applies the price and promotion levers.")
        try:
            gs = core.global_shap(MODEL, DF, 120)
        except Exception:
            gs = pd.DataFrame()
        if gs.empty:
            st.info("Global SHAP values need the shap and xgboost packages.")
        else:
            if ALTAIR:
                st.altair_chart(alt.Chart(gs).mark_bar(color="#6254c7").encode(
                    x=alt.X("Mean |effect| (units/day):Q", title="Average effect on the forecast (units/day)"),
                    y=alt.Y("Feature group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260))).properties(height=300), use_container_width=True)
            st.caption("Average absolute SHAP effect over the last 120 days, scaled by traffic. Store traffic is handled directly in the model, so it appears in the single-category view.")
    with tab_loc:
        cat = st.selectbox("Category", CATEGORIES, key="explain_category")
        bridge = explain_profile(current_profile(cat))
        render_bridge_chart(bridge, "explain", top_n=12)
        st.caption(bridge_caption(bridge))
        show = bridge[["Driver", "Type", "Contribution (units/day)"]].copy()
        show["Contribution (units/day)"] = show["Contribution (units/day)"].round(2)
        total = pd.DataFrame([{"Driver": "Forecast (sum of all rows)", "Type": "total", "Contribution (units/day)": round(float(bridge.attrs["sum_of_parts"]), 2)}])
        st.dataframe(pd.concat([show, total], ignore_index=True), use_container_width=True, hide_index=True)
    with tab_trust:
        st.caption("Agreement means the manager's decision matches the system's recommendation (approve unless the change lowers profit, raises stockout risk by more than 5 points, or leaves the category Critical). "
                   "Modified and deferred decisions are not counted as agreement.")
        agreement = compute_agreement_stats(st.session_state.decision_history)
        if agreement["n_decided"] == 0:
            st.info("Approve or reject a scenario to start measuring agreement.")
        else:
            t1, t2, t3 = st.columns(3)
            t1.metric("Decision agreement", f"{agreement['overall_rate']*100:.1f}%")
            t2.metric("Approved decisions", agreement["n_approved"])
            t3.metric("Overrides of the recommendation", agreement["n_overrides"])
            if not agreement["by_severity"].empty and ALTAIR:
                st.markdown("**Agreement by scenario severity**")
                st.altair_chart(alt.Chart(agreement["by_severity"]).mark_bar(cornerRadiusEnd=5, color="#6254c7").encode(
                    x=alt.X("Agreement rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Severity:N", sort=["Critical", "Warning", "Info"], title=None)).properties(height=140), use_container_width=True)
            if len(agreement["rolling"]) >= 2 and ALTAIR:
                st.markdown("**Cumulative agreement over recorded decisions**")
                st.altair_chart(alt.Chart(agreement["rolling"]).mark_line(point=True, color="#6254c7").encode(
                    x=alt.X("Decision #:Q", title="Decision sequence"),
                    y=alt.Y("Cumulative agreement rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]))).properties(height=200), use_container_width=True)

if selected_page == "✅ Data & validation":
    v = MODEL.validation
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows used", f"{len(DF):,}")
    d2.metric("Days of history", f"{DF['Date'].nunique():,}")
    d3.metric("Trained on", f"{MODEL.n_train:,}")
    d4.metric("Held-out test days", v["test_days"])
    st.caption(f"Source: {DATA_SOURCE}. The file's AI_* output columns are not used as inputs. Training uses only days before {ASOF.date().isoformat()}; the last 15% of those days are held out for testing.")
    t_acc, t_int, t_drv, t_lev, t_del, t_asm = st.tabs(["Accuracy", "Intervals", "Drivers", "Price & promotion", "Delivery", "Assumptions"])
    with t_acc:
        overall = pd.DataFrame(v["overall"]).T.reset_index().rename(columns={"index": "Method"})
        st.dataframe(overall.round(2), use_container_width=True, hide_index=True)
        mm, bb = v["overall"]["Retail Brain model"], v["overall"]["Baseline: category avg rate x traffic"]
        if mm["MAE (units)"] <= bb["MAE (units)"] * 0.98:
            st.success(f"The model beats the traffic baseline (MAE {mm['MAE (units)']:.1f} vs {bb['MAE (units)']:.1f} units).")
        elif mm["MAE (units)"] <= bb["MAE (units)"] * 1.05:
            st.info(f"The model is level with the 'category average per visitor × traffic' baseline (MAE {mm['MAE (units)']:.1f} vs {bb['MAE (units)']:.1f} units). Store traffic carries almost all the signal in this data.")
        else:
            st.warning(f"The simple baseline is more accurate than the model (MAE {bb['MAE (units)']:.1f} vs {mm['MAE (units)']:.1f} units).")
        if not v["per_category"].empty:
            st.dataframe(v["per_category"].round(2), use_container_width=True, hide_index=True)
        st.caption(f"XGBoost settings {v.get('xgb_params', {})} were chosen on a separate calibration window, never on the test days.")
    with t_int:
        cov = v["coverage"]
        c1, c2 = st.columns(2)
        c1.metric("Outcomes inside the 80% interval", f"{cov*100:.0f}%", f"{(cov - v['target_coverage'])*100:+.0f} pp vs 80% target")
        c2.metric("Interval half-width", f"±{v['interval_q']*100:.0f}% of forecast")
        series = v["series"]
        vc = st.selectbox("Category", sorted(series["Category"].unique()), key="val_cat")
        sc = series[series["Category"] == vc]
        if ALTAIR and not sc.empty:
            base = alt.Chart(sc).encode(x=alt.X("Date:T", title=None))
            band = base.mark_area(opacity=.18, color="#6254c7").encode(y="lo:Q", y2="hi:Q")
            fl = base.mark_line(color="#6254c7").encode(y=alt.Y("forecast:Q", title="Units per day"))
            al = base.mark_line(color="#ff7a18", strokeDash=[4, 3]).encode(y="actual:Q")
            st.altair_chart((band + fl + al).properties(height=260), use_container_width=True)
            st.caption("Purple: forecast with its 80% band. Orange dashed: recorded demand. Held-out period only.")
    with t_drv:
        imp = v["importance"]
        if imp is not None and not imp.empty:
            st.dataframe(imp.round(3), use_container_width=True, hide_index=True)
            st.caption("Increase in error when a feature group is shuffled on the test days. Values near or below zero mean the model does not rely on that group on unseen days.")
    with t_lev:
        le = MODEL.lever_est
        lever_rows = pd.DataFrame([
            {"Lever": "Price elasticity vs competitors", "Estimate": f"{le['elasticity_est']:.2f}", "Std. error": f"{le['elasticity_se']:.2f}",
             "t-statistic": f"{le['elasticity_t']:.1f}", "Detectable": "Yes" if le["elasticity_detectable"] else "No", "Used in app": f"{LEVERS['elasticity']:.2f} ({LEVERS['elasticity_source']})"},
            {"Lever": "Campaign uplift on demand", "Estimate": f"{le['promo_uplift_est']*100:+.1f}%", "Std. error": f"{le['promo_uplift_se']:.2f}",
             "t-statistic": f"{le['promo_t']:.1f}", "Detectable": "Yes" if le["promo_detectable"] else "No", "Used in app": f"{LEVERS['promo_uplift']*100:+.1f}% ({LEVERS['promo_source']})"}])
        st.dataframe(lever_rows, use_container_width=True, hide_index=True)
        st.caption(f"Esselunga's price moves only about ±{le['rel_price_std']*100:.1f}% against competitors in this file and there are {le['promo_days']} campaign days. "
                   "Where an effect is not detectable the app uses the editable assumption in the sidebar.")
    with t_del:
        fc = pd.DataFrame({"Delivery delay (hours)": MODEL.fill_x, "Expected fill rate": MODEL.fill_y}).drop_duplicates()
        if ALTAIR:
            st.altair_chart(alt.Chart(fc).mark_line(color="#6254c7", interpolate="step-after").encode(
                x="Delivery delay (hours):Q", y=alt.Y("Expected fill rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1.02]))).properties(height=260), use_container_width=True)
        st.caption("Share of demand actually sold as a function of delivery delay, learned from history. The inventory agent uses it to estimate unfulfilled demand.")
    with t_asm:
        asm = pd.DataFrame([{"Category": c, "Shelf life (days)": core.CATEGORY_ASSUMPTIONS.get(c, core.DEFAULT_ASSUMPTION)["shelf_life_days"],
                             "Base waste rate": f"{core.CATEGORY_ASSUMPTIONS.get(c, core.DEFAULT_ASSUMPTION)['base_waste_rate']*100:.1f}%",
                             "Gross margin used": f"{MODEL.margin.get(c, core.DEFAULT_MARGIN)*100:.0f}%",
                             "Unit cost (€)": round(MODEL.ref_price.get(c, 0) * (1 - MODEL.margin.get(c, core.DEFAULT_MARGIN)), 2)} for c in MODEL.categories])
        st.dataframe(asm, use_container_width=True, hide_index=True)
        st.caption("The file has no stock, expiry, waste or unit-cost columns. These values are editable constants in retail_brain_core.py.")

if st.session_state.get("persistence_warning"):
    st.divider()
    st.warning("Decision history could not be saved to disk here, so decisions stay in this session only. Details: " + str(st.session_state.persistence_warning)[:200])
