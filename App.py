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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Scenario templates (expressed in the variables of the real data schema)
# Change grammar: <field>_pct = multiplicative %, <field>_delta = additive, <field>_set = overwrite
# -----------------------------------------------------------------------------
CHANGE_LABELS = {
    "price_pct": "Own price", "competitor_price_pct": "Competitor price", "traffic_pct": "Store traffic",
    "temperature_delta": "Temperature", "delay_hrs_delta": "Delivery delay (h)",
    "staff_capacity_pct_delta": "Staff capacity (pts)", "compliance_pct_delta": "Shelf compliance (pts)",
    "unit_cost_pct": "Unit cost", "promo_active_set": "Promotion active", "cold_chain_set": "Cold-chain failure",
}
SCENARIO_TEMPLATES = {
    "Competitor price reduction": {
        "description": "Competitors cut price, which widens our price gap and may divert price-sensitive visitors.",
        "changes": {"competitor_price_pct": -12, "traffic_pct": -5},
        "variables": ["competitor_price", "traffic"],
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
        "description": "Own price, competitor price and unit cost move, squeezing estimated gross margin.",
        "changes": {"price_pct": 5, "competitor_price_pct": -4, "unit_cost_pct": 6},
        "variables": ["price", "competitor_price", "unit_cost"],
    },
    "Staffing and compliance alert": {
        "description": "Lower staffing during higher traffic creates an operational compliance risk.",
        "changes": {"staff_capacity_pct_delta": -15, "traffic_pct": 6, "compliance_pct_delta": -12},
        "variables": ["staff_capacity_pct", "traffic", "compliance_pct"],
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
        st.caption("Used only where the data cannot identify the effect (see Data and validation).")
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
        elif key.endswith("_set"):
            parts.append(f"{label}: {'yes' if value else 'no'}")
        else:
            parts.append(f"{label}: {value}")
    return " · ".join(parts)


def create_scenario(template_name: str, category: str, source: str = "Manual scan") -> dict[str, Any]:
    template = SCENARIO_TEMPLATES[template_name]
    changes = dict(template["changes"])
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
        "description": template["description"], "category": category, "changes": changes,
        "variables": list(template["variables"]), "status": "pending", "source": source,
        "severity": severity, "ai_recommendation": recommendation, "before": {k: v for k, v in before.items() if k not in ("agent_reports", "messages")},
        "after": {k: v for k, v in after.items() if k not in ("agent_reports", "messages")},
        "delta_demand_units": delta_demand, "delta_gross_profit_eur": delta_profit,
        "change_summary": describe_changes(changes),
    }


def _is_non_perishable(category: str) -> bool:
    return core.CATEGORY_ASSUMPTIONS.get(category, core.DEFAULT_ASSUMPTION)["shelf_life_days"] > 7


def scan_scenarios(count: int = 3, source: str = "Manual scan") -> list[dict[str, Any]]:
    names = random.sample(list(SCENARIO_TEMPLATES.keys()), k=min(count, len(SCENARIO_TEMPLATES)))
    new_events = [create_scenario(n, random.choice(CATEGORIES), source) for n in names]
    st.session_state.events = new_events + st.session_state.events
    st.session_state.last_scan_at = now_iso()
    st.session_state.scan_count += 1
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


def record_decision(event: dict[str, Any], decision: str, note: str = "") -> None:
    event["status"] = decision
    event["decision_at"] = now_iso()
    event["decision_note"] = note.strip()
    if decision == "approved" and event["id"] not in st.session_state.approved_event_ids:
        st.session_state.approved_event_ids.append(event["id"])
    record = {
        "event_id": event["id"], "created_at": event.get("created_at"), "decision_at": event["decision_at"],
        "template": event["template"], "category": event["category"], "variables": event["variables"],
        "decision": decision, "note": note.strip(), "severity": event.get("severity", "Info"),
        "ai_recommendation": event.get("ai_recommendation", "approve"), "as_of": ASOF.date().isoformat(),
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
    st.caption("STORE OPERATIONS WORKSPACE")
    st.write("**Data mode:** store history · local SQLite decision audit")
    st.session_state.auto_monitor = st.toggle(
        "Auto-scan every 2 minutes", value=st.session_state.auto_monitor,
        help="Generates scenarios on the current day's profile. Requires the app session to remain open.")
    if st.session_state.auto_monitor:
        st.caption("Auto-scan is active for this session.")
    if st.button("🔎 Scan for operational scenarios", use_container_width=True, type="primary"):
        scan_scenarios(3)
        st.rerun()
    if st.button("➕ Generate one new alert", use_container_width=True):
        st.session_state.events.insert(0, create_scenario(random.choice(list(SCENARIO_TEMPLATES)), random.choice(CATEGORIES), "Manual alert"))
        st.rerun()

if st.session_state.auto_monitor:
    try:
        last = st.session_state.last_auto_event_at
        elapsed = (datetime.now() - datetime.fromisoformat(last.replace("Z", "+00:00")).replace(tzinfo=None)).total_seconds() if last else 9999
    except Exception:
        elapsed = 9999
    if elapsed >= 120:
        st.session_state.events.insert(0, create_scenario(random.choice(list(SCENARIO_TEMPLATES)), random.choice(CATEGORIES), "Auto-monitor"))
        st.session_state.last_auto_event_at = now_iso()

pending_events = [e for e in st.session_state.events if e.get("status") == "pending"]
critical_count = sum(e.get("severity") == "Critical" for e in pending_events)
if pending_events and selected_page != "📊 9-agent dashboard":
    if critical_count:
        st.error(f"🚨 **Operational alert:** {len(pending_events)} scenario(s) need manager review, including {critical_count} critical alert(s). Decisions require manager input.")
    else:
        st.warning(f"🔔 **Manager review required:** {len(pending_events)} scenario(s) are pending approval, rejection, or deferral.")
elif selected_page != "📊 9-agent dashboard":
    st.success("No pending alerts. Use **Scan for operational scenarios** to generate a new review queue.")
render_sound_control(len(pending_events))
if LEVERS["elasticity_source"] == "assumed":
    st.caption("ℹ️ Price effects use an **assumed** elasticity because the store history cannot identify one. See **Data & validation**.")


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
    _, after_profile, before, after = scenario_metrics(event["category"], event["changes"])
    metrics = [
        ("Forecast demand", before["demand_units"], after["demand_units"], "units"),
        ("Gross profit", before["gross_profit_eur"], after["gross_profit_eur"], "eur"),
        ("Stockout risk", before["stockout_risk_pct"], after["stockout_risk_pct"], "pct"),
        ("Waste cost proxy", before["waste_cost_eur"], after["waste_cost_eur"], "eur"),
        ("Staffing pressure", before["staffing_pressure"], after["staffing_pressure"], "idx"),
    ]
    compare = []
    for label, b, a, kind in metrics:
        if kind == "eur":
            bs, ass, delta = fmt_eur(float(b)), fmt_eur(float(a)), fmt_eur(float(a) - float(b))
        elif kind == "pct":
            bs, ass, delta = f"{float(b):.1f}%", f"{float(a):.1f}%", f"{float(a)-float(b):+.1f} pp"
        elif kind == "idx":
            bs, ass, delta = f"{float(b):.2f}", f"{float(a):.2f}", f"{float(a)-float(b):+.2f}"
        else:
            bs, ass, delta = fmt_num(float(b)), fmt_num(float(a)), f"{float(a)-float(b):+.1f}"
        compare.append({"Metric": label, "Before": bs, "After if applied": ass, "Change": delta})
    st.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)
    st.caption(f"Demand after the change: {after['demand_units']:.0f} units/day, 80% interval {after['demand_low']:.0f} to {after['demand_high']:.0f}. "
               "Projections are model-based, not measured outcomes or causal estimates.")

    st.markdown("#### Why this projection")
    bridge = explain_profile(after_profile)
    render_bridge_chart(bridge, f"dlg_{event_id}")
    st.caption(bridge_caption(bridge))

    cases = find_relevant_cases(event)
    st.markdown("#### Relevant historical decisions")
    if cases:
        rows = [{"Similarity": f"{c['similarity']:.0%}", "Previous scenario": c.get("template", "—"), "Category": c.get("category", "—"),
                 "Decision": c.get("decision", "—").title(), "Recorded note": c.get("note") or "No manager note recorded",
                 "Projected profit delta at review": fmt_eur(float(c.get("delta_gross_profit_eur_at_review", 0)))} for c in cases]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Case matching uses category, overlapping changed variables, and scenario type. It is a reference aid, not an ML-trained recommendation.")
    else:
        st.info("No sufficiently similar prior decisions are recorded yet.")

    note = st.text_area("Manager rationale (optional)", key=f"note_{event_id}", placeholder="Why is this decision appropriate? What should be monitored?")
    a, b, c = st.columns(3)
    if a.button("✅ Approve scenario", use_container_width=True, type="primary", key=f"approve_{event_id}"):
        record_decision(event, "approved", note)
        st.rerun()
    if b.button("❌ Reject scenario", use_container_width=True, key=f"reject_{event_id}"):
        record_decision(event, "rejected", note)
        st.rerun()
    if c.button("⏸ Defer decision", use_container_width=True, key=f"defer_{event_id}"):
        record_decision(event, "deferred", note)
        event["status"] = "pending"
        st.rerun()


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
STATUS_ICON = {"ok": "🟢", "watch": "🟡", "alert": "🔴"}

if selected_page == "🧪 Trial & error lab":
    st.subheader("Trial & error lab: test decisions before acting")
    st.write("Change any operating parameter and compare the projected outcome with the current day's profile. Nothing here is approved or applied; it is a what-if sandbox.")
    st.warning("Exploratory simulation. Context effects (traffic, weather, staffing) come from the trained model. Price and promotion effects follow the lever settings shown on **Data & validation**.")
    sandbox_category = st.selectbox("Category to simulate", CATEGORIES, key="sandbox_category")
    baseline = current_profile(sandbox_category)
    tag = f"{sandbox_category}_{ASOF.date().isoformat()}"
    B = core._BOUNDS
    left, right = st.columns(2)
    with left:
        st.markdown("#### Commercial and demand drivers")
        sim_price = st.slider("Own selling price (€)", 0.10, float(max(5.0, baseline["price"] * 2)), float(baseline["price"]), 0.01, key=f"lab_price_{tag}")
        sim_comp = st.slider("Competitor price, mean (€)", 0.10, float(max(5.0, baseline["competitor_price"] * 2)), float(baseline["competitor_price"]), 0.01, key=f"lab_comp_{tag}")
        sim_promo = st.checkbox("Promotion / campaign active", value=bool(baseline["promo_active"]), key=f"lab_promo_{tag}")
        sim_traffic = st.slider("Store traffic (visitors/day)", 50, int(max(3000, baseline["traffic"] * 2)), int(baseline["traffic"]), 10, key=f"lab_traffic_{tag}")
        sim_temp = st.slider("Temperature (°C)", -10, 45, int(round(np.clip(baseline["temperature"], -10, 45))), 1, key=f"lab_temp_{tag}")
    with right:
        st.markdown("#### Supply, staffing and cost")
        sim_delay = st.slider("Delivery delay (hours)", 0.0, 24.0, float(np.clip(baseline["delay_hrs"], 0, 24)), 0.5, key=f"lab_delay_{tag}")
        sim_staff = st.slider("Staff capacity (%)", 20, 100, int(np.clip(round(baseline["staff_capacity_pct"]), 20, 100)), 1, key=f"lab_staff_{tag}")
        sim_comp_pct = st.slider("Shelf compliance (%)", 0, 100, int(np.clip(round(baseline["compliance_pct"]), 0, 100)), 1, key=f"lab_compliance_{tag}")
        sim_cost = st.slider("Unit cost (€), assumption derived from margin", 0.05, float(max(5.0, baseline["unit_cost"] * 2)), float(baseline["unit_cost"]), 0.01, key=f"lab_cost_{tag}")
        sim_cold = st.checkbox("Cold-chain failure", value=bool(baseline["cold_chain"]), key=f"lab_cold_{tag}")
    simulated = dict(baseline)
    simulated.update({"price": sim_price, "competitor_price": sim_comp, "promo_active": int(sim_promo), "traffic": float(sim_traffic),
                      "temperature": float(sim_temp), "delay_hrs": float(sim_delay), "staff_capacity_pct": float(sim_staff),
                      "compliance_pct": float(sim_comp_pct), "unit_cost": float(sim_cost), "cold_chain": int(sim_cold)})
    before = calculate_metrics(sandbox_category, baseline)
    after = calculate_metrics(sandbox_category, simulated)
    dd, dp = after["demand_units"] - before["demand_units"], after["gross_profit_eur"] - before["gross_profit_eur"]
    st.markdown("#### Projected outcome vs current profile")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecast demand (units/day)", f"{after['demand_units']:.1f}", f"{dd:+.1f} vs current")
    k2.metric("Gross profit (€/day)", fmt_eur(after["gross_profit_eur"]), f"{dp:+.2f} € vs current")
    k3.metric("Revenue (€/day)", fmt_eur(after["revenue_eur"]), f"{after['revenue_eur']-before['revenue_eur']:+.2f} € vs current")
    k4.metric("Stockout risk", f"{after['stockout_risk_pct']:.1f}%", f"{after['stockout_risk_pct']-before['stockout_risk_pct']:+.1f} pp vs current")
    st.caption(f"80% prediction interval for demand in the trial: {after['demand_low']:.0f} to {after['demand_high']:.0f} units/day.")
    rows = [("Forecast demand (units/day)", "demand_units"), ("Sales fulfilled (units/day)", "sales_units"), ("Lost sales (units/day)", "lost_sales_units"),
            ("Gross profit (€/day)", "gross_profit_eur"), ("Gross margin (%)", "margin_pct"), ("Safety buffer (units)", "safety_stock_units"),
            ("Stockout risk (%)", "stockout_risk_pct"), ("Waste cost proxy (€/day)", "waste_cost_eur"), ("Staffing pressure index", "staffing_pressure"),
            ("Compliance flags", "compliance_flags")]
    st.dataframe(pd.DataFrame([{"Indicator": n, "Current profile": float(before[k]), "Trial scenario": float(after[k]), "Change": float(after[k]) - float(before[k])} for n, k in rows]).round(2),
                 use_container_width=True, hide_index=True)

    st.markdown("#### Why the forecast moves")
    br_before, br_after = explain_profile(baseline), explain_profile(simulated)
    cmp_ = br_before[["Driver", "Contribution (units/day)"]].merge(br_after[["Driver", "Contribution (units/day)"]], on="Driver", suffixes=(" current", " trial"))
    cmp_["Change (units/day)"] = cmp_["Contribution (units/day) trial"] - cmp_["Contribution (units/day) current"]
    cmp_ = cmp_.reindex(cmp_["Change (units/day)"].abs().sort_values(ascending=False).index)
    st.dataframe(cmp_.round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Agent feedback: what may happen")
    fb = []
    if sim_price != baseline["price"]:
        fb.append(("Pricing & demand", f"Own price moves {((sim_price/baseline['price'])-1)*100:+.1f}%. Using a price elasticity of {LEVERS['elasticity']:.2f} ({LEVERS['elasticity_source']}), demand changes {dd:+.1f} units/day and gross profit {dp:+.2f} €/day."))
    else:
        fb.append(("Pricing & demand", f"Own price is unchanged. Demand changes {dd:+.1f} units/day because of the other inputs."))
    gap_b = (baseline["price"] / baseline["competitor_price"] - 1) * 100
    gap_a = (simulated["price"] / simulated["competitor_price"] - 1) * 100
    if abs(gap_a - gap_b) > 0.05:
        fb.append(("Competition", f"Price position vs competitors moves from {gap_b:+.1f}% to {gap_a:+.1f}%. This is a price-position proxy, not a market-share forecast."))
    fb.append(("Inventory & availability", f"Delay of {sim_delay:.1f} h gives an expected fill rate of {MODEL.fill_rate(sim_delay)*100:.0f}% (learned from history). Stockout risk moves from {before['stockout_risk_pct']:.1f}% to {after['stockout_risk_pct']:.1f}%."))
    if after["waste_cost_eur"] - before["waste_cost_eur"] > 0.01:
        fb.append(("Freshness & waste", f"Waste cost proxy rises by {after['waste_cost_eur']-before['waste_cost_eur']:.2f} €/day. Waste rates are assumptions per category."))
    if abs(after["staffing_pressure"] - before["staffing_pressure"]) > 0.01:
        fb.append(("Workforce", f"Staffing pressure moves from {before['staffing_pressure']:.2f} to {after['staffing_pressure']:.2f}."))
    if after["compliance_risk"]:
        fb.append(("Compliance", f"{after['compliance_flags']} active flag(s): {after['agent_reports']['compliance_tasks']['detail']}. Treat as a review trigger, not a compliance determination."))
    if after["status"] != before["status"]:
        fb.append(("Overall risk", f"Status changes from {before['status']} to {after['status']}."))
    for t, m in fb:
        st.markdown(f"**{t}:** {m}")
    with st.expander("View all simulated inputs"):
        st.dataframe(pd.DataFrame([{"Parameter": core.PROFILE_LABELS[k], "Current profile": baseline.get(k), "Trial scenario": simulated.get(k)} for k in core.PROFILE_LABELS]),
                     use_container_width=True, hide_index=True)

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
    if "dashboard_selected_agent" not in st.session_state or st.session_state.dashboard_selected_agent not in [a[1] for a in agent_items]:
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
        st.caption("Select an agent to see its live output for every category.")
        for i in range(0, len(agent_items), 3):
            btn_cols = st.columns(3, gap="small")
            for j, (icon, title, description, _name) in enumerate(agent_items[i:i + 3]):
                with btn_cols[j]:
                    is_selected = st.session_state.dashboard_selected_agent == title
                    if st.button(icon, key=f"agent_icon_{i+j}", help=f"{title}: {description}", use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state.dashboard_selected_agent = title
                        st.rerun()
                    st.markdown(f"<div style='text-align:center;font-size:.66rem;line-height:1.15;font-weight:700;margin:-.35rem 0 .5rem'>{title}</div>", unsafe_allow_html=True)
        sel = next(a for a in agent_items if a[1] == st.session_state.dashboard_selected_agent)
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
        st.caption("Recorded demand is the figure in the data for the decision day. The model never saw it: training stops the day before.")
    with st.expander("Category inputs and agent message log (event bus)"):
        cat_for_cards = st.selectbox("Category", CATEGORIES, key="dashboard_category")
        profile = current_profile(cat_for_cards)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.write(f"**Own price:** {fmt_eur(profile['price'])}")
            st.write(f"**Competitor mean:** {fmt_eur(profile['competitor_price'])}")
            st.write(f"**Private label:** {fmt_eur(profile['private_label_price'])}")
            st.write(f"**Promotion active:** {'Yes' if profile['promo_active'] else 'No'}")
        with p2:
            st.write(f"**Traffic:** {profile['traffic']:,.0f} visitors")
            st.write(f"**Temperature:** {profile['temperature']:.1f} °C")
            st.write(f"**Weather / event:** {profile['weather']} / {profile['event']}")
            st.write(f"**Delivery delay:** {profile['delay_hrs']:.1f} h")
        with p3:
            st.write(f"**Staff capacity:** {profile['staff_capacity_pct']:.0f}%")
            st.write(f"**Shelf compliance:** {profile['compliance_pct']:.0f}%")
            st.write(f"**Checkout queue:** {profile['queue_len']:.1f}")
            st.write(f"**Cold-chain failure:** {'Yes' if profile['cold_chain'] else 'No'}")
        st.caption(f"Unit cost ({fmt_eur(profile['unit_cost'])}), shelf life ({profile['shelf_life_days']} days) and base waste rate ({profile['base_waste_rate']*100:.1f}%) are assumptions, not columns in the data.")
        st.markdown("**Event-bus messages for this category**")
        st.caption("Agents never call each other. Each subscribes to topics on an in-process bus and publishes results for others to pick up. This separates responsibilities and makes the reasoning auditable; it runs in one process and is not a distributed system.")
        st.dataframe(pd.DataFrame(all_metrics[cat_for_cards]["messages"]), use_container_width=True, hide_index=True)
    st.caption("Retail Brain is a decision-support prototype running on a dummy operations file and model-based estimates. No live store systems or operational controls are connected.")

if selected_page == "🚨 Decision center":
    st.subheader("Scenario queue and manager review")
    st.write("Scenarios are prospective what-if cases on today's profile. Approving one applies its changes to the category profile; rejecting it does not.")
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
        st.info("No decisions recorded yet. Review a scenario and record a decision to build a reference set.")
    else:
        hist = pd.DataFrame(st.session_state.decision_history)
        m1, m2, m3 = st.columns(3)
        m1.metric("Recorded decision records", len(hist))
        m2.metric("Unique scenarios with a decision", hist["event_id"].nunique() if "event_id" in hist.columns else len(hist))
        m3.metric("Saved to audit file", "Yes" if st.session_state.get("last_persist_ok", not st.session_state.get("persistence_warning")) else "No")
        st.caption("Decisions are written to a local SQLite audit file and reloaded when the app restarts. On hosts with ephemeral storage the file can be wiped on redeploy, so it is not a managed database.")
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
    st.write("Ask about the dashboard, category performance, why a forecast looks the way it does, model accuracy, previous decisions, or what-if changes. It works immediately, with no API key, and calculates what-ifs with the same model as the rest of the app.")

    def _assistant_context() -> dict[str, Any]:
        cats = []
        for category in CATEGORIES:
            profile = current_profile(category)
            m = calculate_metrics(category, profile)
            cats.append({"category": category, "profile": profile, "metrics": m})
        recent_events = [{k: e.get(k) for k in ["id", "category", "template", "status", "severity", "created_at", "change_summary", "decision_note", "ai_assessment_rationale"]} for e in st.session_state.events[:20]]
        return {"categories": cats, "recent_scenarios": recent_events, "recent_decision_history": st.session_state.decision_history[:20],
                "scenario_count": len(st.session_state.events), "decision_history_count": len(st.session_state.decision_history)}

    def _parse_what_if(q: str) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        neg = any(t in q for t in ["decrease", "lower", "reduce", "cut", "drop", "fall"])
        for word, key in (("competitor", "competitor_price_pct"), ("price", "price_pct"), ("traffic", "traffic_pct"), ("visitors", "traffic_pct")):
            m = re.search(rf"{word}\D{{0,30}}?(\d+(?:\.\d+)?)\s*%", q)
            if m:
                v = float(m.group(1))
                if word == "competitor":
                    changes[key] = -v if neg else v
                else:
                    seg = q[max(0, m.start() - 25):m.end() + 5]
                    changes[key] = -v if any(t in seg for t in ["decrease", "lower", "reduce", "cut", "drop", "fall"]) else v
        m = re.search(r"delay\D{0,20}?(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours)", q)
        if m:
            changes["delay_hrs_delta"] = float(m.group(1))
        if "promotion" in q or "campaign" in q:
            changes["promo_active_set"] = 0 if any(t in q for t in ["stop", "end", "remove", "without"]) else 1
        return changes

    def _local_manager_answer(question: str, context: dict[str, Any]) -> str:
        q = question.lower()
        cats = context["categories"]
        category = next((c for c in cats if c["category"].lower() in q), None)
        if any(t in q for t in ["accuracy", "accurate", "validate", "validation", "reliable", "trust the", "how good", "error"]):
            v = MODEL.validation["overall"]
            mm, bb = v["Retail Brain model"], v["Baseline: category avg rate x traffic"]
            return (f"### Forecast accuracy on a held-out period\nOn {MODEL.validation['test_days']} unseen days the model's MAPE is **{mm['MAPE %']:.1f}%** (MAE {mm['MAE (units)']:.1f} units). "
                    f"A simple baseline of category-average demand per visitor times traffic scores **{bb['MAPE %']:.1f}%**. "
                    f"The 80% interval covered **{MODEL.validation['coverage']*100:.0f}%** of outcomes. "
                    "Most of the predictive power comes from store traffic; weather, staffing and calendar effects add little in this dataset. See **Data & validation** for the full table.")
        if any(t in q for t in ["why", "driver", "explain", "because"]) and (category or len(cats)):
            c = category or max(cats, key=lambda x: x["metrics"]["demand_units"])
            br = explain_profile(c["profile"])
            top = br[~br["Type"].isin(["base"])].sort_values("Absolute", ascending=False).head(4)
            lines = [f"### Why the {c['category']} forecast is {c['metrics']['demand_units']:.0f} units/day"]
            for _, r in top.iterrows():
                lines.append(f"- **{r['Driver']}**: {r['Contribution (units/day)']:+.1f} units/day")
            lines.append("\nContext drivers are TreeSHAP values of the XGBoost model scaled by traffic. Open **Model explainability** for the full bridge.")
            return "\n".join(lines)
        if any(t in q for t in ["previous decision", "decision history", "past decision", "what did we decide", "history"]):
            if not context["recent_decision_history"] and not context["recent_scenarios"]:
                return "No decisions or scenarios are recorded yet. Use **Decision center** to review one, then ask me again."
            lines = ["### Recent decisions and scenarios", f"Scenarios this session: **{context['scenario_count']}** · Decision records (including saved history): **{context['decision_history_count']}**"]
            for e in context["recent_scenarios"][:8]:
                lines.append(f"- **{e.get('category')} — {e.get('template')}**: {e.get('status')}, severity {e.get('severity')}. {e.get('ai_assessment_rationale') or e.get('decision_note') or 'No rationale recorded.'}")
            return "\n".join(lines)
        if category is not None and any(t in q for t in ["what if", "what-if", "if i", "if we", "increase", "raise", "decrease", "lower", "reduce", "cut", "delay", "campaign", "promotion"]):
            ch = _parse_what_if(q)
            if not ch:
                ch = {"price_pct": 5.0}
            after_p = core.apply_changes(category["profile"], ch)
            after = calculate_metrics(category["category"], after_p)
            b = category["metrics"]
            note = ""
            if any(k in ch for k in ("price_pct", "competitor_price_pct")):
                note = f"\n\nThe price response uses an elasticity of **{LEVERS['elasticity']:.2f}** ({LEVERS['elasticity_source']}); the store history cannot identify it on its own."
            return (f"### What-if for {category['category']}: {describe_changes(ch)}\n"
                    f"- Demand: **{b['demand_units']:.1f} → {after['demand_units']:.1f}** units/day (80% interval {after['demand_low']:.0f} to {after['demand_high']:.0f})\n"
                    f"- Gross profit: **€{b['gross_profit_eur']:.2f} → €{after['gross_profit_eur']:.2f}** per day\n"
                    f"- Stockout risk: **{b['stockout_risk_pct']:.1f}% → {after['stockout_risk_pct']:.1f}%**; waste cost proxy €{b['waste_cost_eur']:.2f} → €{after['waste_cost_eur']:.2f}\n"
                    f"- Status: **{b['status']} → {after['status']}**{note}\n\nThis is a model projection, not a measured outcome.")
        if any(t in q for t in ["dashboard", "summarize", "summary", "overall", "all categories", "how are we doing"]):
            lines = [f"### Snapshot for {ASOF.date().isoformat()}", "| Category | Forecast/day | Gross profit/day | Stockout risk | Waste/day | Status |", "|---|---:|---:|---:|---:|---|"]
            for c in cats:
                m = c["metrics"]
                lines.append(f"| {c['category']} | {m['demand_units']:.1f} | €{m['gross_profit_eur']:.2f} | {m['stockout_risk_pct']:.1f}% | €{m['waste_cost_eur']:.2f} | {m['status']} |")
            return "\n".join(lines)
        if category is None and any(t in q for t in ["worst", "risk", "critical", "stockout"]):
            category = max(cats, key=lambda c: c["metrics"]["stockout_risk_pct"])
        if category is None and any(t in q for t in ["best", "most profitable", "highest profit"]):
            category = max(cats, key=lambda c: c["metrics"]["gross_profit_eur"])
        if category:
            p, m = category["profile"], category["metrics"]
            return (f"### {category['category']} on {ASOF.date().isoformat()}\n"
                    f"- Price **€{p['price']:.2f}** vs competitor mean **€{p['competitor_price']:.2f}** ({m['price_gap_pct']:+.1f}%).\n"
                    f"- Forecast demand **{m['demand_units']:.1f}** units/day (80% interval {m['demand_low']:.0f} to {m['demand_high']:.0f}); sales **{m['sales_units']:.1f}**; lost **{m['lost_sales_units']:.1f}**.\n"
                    f"- Revenue **€{m['revenue_eur']:.2f}**, gross profit **€{m['gross_profit_eur']:.2f}**, margin **{m['margin_pct']:.1f}%**.\n"
                    f"- Stockout risk **{m['stockout_risk_pct']:.1f}%**, safety buffer **{m['safety_stock_units']:.0f}** units, waste proxy **€{m['waste_cost_eur']:.2f}**, staffing pressure **{m['staffing_pressure']:.2f}**, compliance flags **{m['compliance_flags']}**.\n\n"
                    "Try: *what if I raise the price by 5%?*, *why is the forecast this high?*, *how accurate is the model?*")
        if any(t in q for t in ["agent", "nine", "9-agent"]):
            return ("Nine agents share one event bus: **demand forecasting, inventory risk, commercial intelligence, replenishment, freshness and waste, workforce planning, promotion and display, profitability, compliance and tasks**. "
                    "They publish and subscribe to messages (see the message log on the dashboard). It is an in-process architecture, not nine separate deployed services.")
        return ("I can summarize the dashboard, describe a category, explain a forecast, report model accuracy, review past decisions, or compute a what-if such as **‘What if I raise Birra price by 5%?’** or **‘What if Ortofrutta delivery delay is 4 hours?’**.")

    welcome = "Hello! I'm Retail Brain's built-in manager assistant. Try ‘Summarize the dashboard’, ‘Why is the Birra forecast this value?’, ‘How accurate is the model?’, or a what-if."
    if "manager_ai_messages" not in st.session_state:
        st.session_state.manager_ai_messages = [{"role": "assistant", "content": welcome}]
    control_col, _ = st.columns([1, 4])
    with control_col:
        if st.button("🗑️ Clear chat", key="clear_manager_ai_chat"):
            st.session_state.manager_ai_messages = []
            st.rerun()
    for msg in st.session_state.manager_ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    handoff_prompt = st.session_state.pop("assistant_prompt", None)
    user_prompt = st.chat_input("Ask about a decision, scenario, or what-if…", key="manager_ai_chat_input") or handoff_prompt
    if user_prompt:
        st.session_state.manager_ai_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        try:
            reply = _local_manager_answer(user_prompt, _assistant_context())
        except Exception as exc:
            reply = f"I couldn't read the current metrics ({exc}). Please try again."
        st.session_state.manager_ai_messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    with st.expander("About this assistant"):
        st.markdown("- Rule-based and keyword-driven; it is not a large language model and does not connect to external services.\n"
                    "- Every number comes from the same model and agents as the dashboard.\n"
                    "- Projections are model estimates, not verified real-world outcomes.")

if selected_page == "🔬 Model explainability":
    st.subheader("Model explainability")
    kind_text = "XGBoost rate model with TreeSHAP explanations" if MODEL.kind == "xgboost" else "category-average baseline (XGBoost unavailable)"
    st.write(f"**Demand estimator:** {kind_text}. It predicts demand per visitor from context, then multiplies by store traffic and applies the price and promotion levers.")
    st.caption("Explanations describe this fitted model. They are not causal claims about the store.")
    st.markdown("#### Global view: what drives forecasts in the last 120 days")
    try:
        gs = core.global_shap(MODEL, DF, 120)
    except Exception:
        gs = pd.DataFrame()
    if gs.empty:
        st.info("Global SHAP values are unavailable (SHAP or XGBoost is not installed).")
    else:
        if ALTAIR:
            st.altair_chart(alt.Chart(gs).mark_bar(color="#6254c7").encode(
                x=alt.X("Mean |effect| (units/day):Q"), y=alt.Y("Feature group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                tooltip=["Feature group", alt.Tooltip("Mean |effect| (units/day):Q", format=".2f")]), use_container_width=True)
        st.caption("Mean absolute SHAP effect on the demand rate, scaled by traffic. Store traffic is handled explicitly in the model, so its effect appears in the local bridge below rather than here.")
    st.markdown("#### Local view: the demand bridge for one category")
    cat = st.selectbox("Category for explanation", CATEGORIES, key="explain_category")
    prof = current_profile(cat)
    bridge = explain_profile(prof)
    render_bridge_chart(bridge, "explain", top_n=12)
    st.caption(bridge_caption(bridge))
    show = bridge[["Driver", "Type", "Contribution (units/day)"]].copy()
    show["Contribution (units/day)"] = show["Contribution (units/day)"].round(2)
    total = pd.DataFrame([{"Driver": "Forecast (sum of all rows)", "Type": "total", "Contribution (units/day)": round(float(bridge.attrs["sum_of_parts"]), 2)}])
    st.dataframe(pd.concat([show, total], ignore_index=True), use_container_width=True, hide_index=True)
    if not bridge.attrs.get("shap"):
        st.warning("SHAP is unavailable, so context drivers are collapsed into a single category-level row.")

    st.markdown("#### Trust calibration: manager agreement with the system's recommendation")
    st.caption("Every scenario carries a default recommendation (approve unless it lowers profit, raises stockout risk by more than 5 points, or leaves the category Critical). "
               "Agreement means the manager's decision matches it; a different decision is an override. Deferrals are excluded. This is a behavioural proxy for Trust in AI and Decision Confidence, not a validated psychometric scale.")
    agreement = compute_agreement_stats(st.session_state.decision_history)
    if agreement["n_decided"] == 0:
        st.info("Approve or reject a scenario to start measuring agreement. Deferred decisions are not counted.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Decision agreement", f"{agreement['overall_rate']*100:.1f}%")
        t2.metric("Approved decisions", agreement["n_approved"])
        t3.metric("Overrides of the recommendation", agreement["n_overrides"])
        if not agreement["by_severity"].empty and ALTAIR:
            st.markdown("**Agreement by scenario severity**")
            st.altair_chart(alt.Chart(agreement["by_severity"]).mark_bar(cornerRadiusEnd=5, color="#6254c7").encode(
                x=alt.X("Agreement rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Severity:N", sort=["Critical", "Warning", "Info"], title=None),
                tooltip=["Severity", "Decisions", alt.Tooltip("Agreement rate:Q", format=".0%")]), use_container_width=True)
        if len(agreement["rolling"]) >= 2 and ALTAIR:
            st.markdown("**Cumulative agreement over recorded decisions**")
            st.altair_chart(alt.Chart(agreement["rolling"]).mark_line(point=True, color="#6254c7").encode(
                x=alt.X("Decision #:Q", title="Decision sequence"),
                y=alt.Y("Cumulative agreement rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=["Decision #", alt.Tooltip("Cumulative agreement rate:Q", format=".0%")]), use_container_width=True)

if selected_page == "✅ Data & validation":
    st.subheader("Data and model validation")
    st.write("This page shows what the store history can and cannot support. Read it before trusting a number.")
    v = MODEL.validation
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows used", f"{len(DF):,}")
    d2.metric("Days of history", f"{DF['Date'].nunique():,}")
    d3.metric("Trained on (before as-of)", f"{MODEL.n_train:,}")
    d4.metric("Held-out test days", v["test_days"])
    st.caption(f"Source: {DATA_SOURCE}. The file's own AI_* output columns (forecasts, risk flags, profit) come from another system and are deliberately **not** used as inputs, to avoid leakage. Training uses only days before {ASOF.date().isoformat()}; the last 15% of those days are held out for the test below. XGBoost settings ({v.get('xgb_params', {})}) were chosen on a separate calibration window, never on the test days.")

    st.markdown("#### 1. Forecast accuracy on unseen days")
    overall = pd.DataFrame(v["overall"]).T.reset_index().rename(columns={"index": "Method"})
    st.dataframe(overall.round(2), use_container_width=True, hide_index=True)
    mm, bb = v["overall"]["Retail Brain model"], v["overall"]["Baseline: category avg rate x traffic"]
    if mm["MAE (units)"] <= bb["MAE (units)"] * 0.98:
        st.success(f"The model beats the traffic baseline (MAE {mm['MAE (units)']:.1f} vs {bb['MAE (units)']:.1f} units).")
    elif mm["MAE (units)"] <= bb["MAE (units)"] * 1.05:
        st.info(f"The model is roughly level with a simple 'category average per visitor × traffic' baseline (MAE {mm['MAE (units)']:.1f} vs {bb['MAE (units)']:.1f} units). Store traffic carries almost all the signal in this data; the other inputs add little. That is a finding about the data, not a reason to hide the model.")
    else:
        st.warning(f"The simple baseline is more accurate than the model (MAE {bb['MAE (units)']:.1f} vs {mm['MAE (units)']:.1f} units). Treat model-specific effects with caution.")
    if not v["per_category"].empty:
        st.dataframe(v["per_category"].round(2), use_container_width=True, hide_index=True)
    st.markdown("#### 2. Prediction intervals")
    cov = v["coverage"]
    c1, c2 = st.columns(2)
    c1.metric("Share of outcomes inside the 80% interval", f"{cov*100:.0f}%", f"{(cov - v['target_coverage'])*100:+.0f} pp vs 80% target")
    c2.metric("Interval half-width", f"±{v['interval_q']*100:.0f}% of forecast")
    st.caption("Split-conformal intervals from a calibration window that sits between training and test. Coverage below target means the interval is slightly optimistic; the app reports it rather than adjusting the test score.")
    series = v["series"]
    vc = st.selectbox("Category to chart", sorted(series["Category"].unique()), key="val_cat")
    sc = series[series["Category"] == vc]
    if ALTAIR and not sc.empty:
        base = alt.Chart(sc).encode(x=alt.X("Date:T", title=None))
        band = base.mark_area(opacity=.18, color="#6254c7").encode(y="lo:Q", y2="hi:Q")
        fl = base.mark_line(color="#6254c7").encode(y=alt.Y("forecast:Q", title="Units per day"))
        al = base.mark_line(color="#ff7a18", strokeDash=[4, 3]).encode(y="actual:Q")
        st.altair_chart(band + fl + al, use_container_width=True)
        st.caption("Purple: forecast and 80% band. Orange dashed: recorded demand. Held-out period only.")

    st.markdown("#### 3. What the model relies on (permutation importance)")
    imp = v["importance"]
    if imp is not None and not imp.empty:
        st.dataframe(imp.round(3), use_container_width=True, hide_index=True)
        st.caption("Increase in mean absolute error when a feature group is shuffled on the test days. Values near or below zero mean the model does not rely on that group in a way that helps on unseen days.")

    st.markdown("#### 4. Can the data identify price and promotion effects?")
    le = MODEL.lever_est
    lever_rows = pd.DataFrame([
        {"Lever": "Price elasticity vs competitors", "Estimate": round(le["elasticity_est"], 2), "Std. error": round(le["elasticity_se"], 2),
         "t-statistic": round(le["elasticity_t"], 1), "Detectable": "Yes" if le["elasticity_detectable"] else "No", "Used in app": f"{LEVERS['elasticity']:.2f} ({LEVERS['elasticity_source']})"},
        {"Lever": "Campaign uplift on demand", "Estimate": f"{le['promo_uplift_est']*100:+.1f}%", "Std. error": round(le["promo_uplift_se"], 2),
         "t-statistic": round(le["promo_t"], 1), "Detectable": "Yes" if le["promo_detectable"] else "No", "Used in app": f"{LEVERS['promo_uplift']*100:+.1f}% ({LEVERS['promo_source']})"},
    ])
    st.dataframe(lever_rows, use_container_width=True, hide_index=True)
    st.caption(f"Own price moves only about ±{le['rel_price_std']*100:.1f}% relative to competitors in this file, and there are {le['promo_days']} campaign days, so price effects are hard to see. "
               "Where an effect is not detectable, the app uses the editable assumption in the sidebar and labels it as an assumption. Association in observational data is not proof of causation.")

    st.markdown("#### 5. Delivery delay and availability")
    fc = pd.DataFrame({"Delivery delay (hours)": MODEL.fill_x, "Expected fill rate": MODEL.fill_y})
    fc = fc.drop_duplicates()
    if ALTAIR:
        st.altair_chart(alt.Chart(fc).mark_line(color="#6254c7", interpolate="step-after").encode(
            x="Delivery delay (hours):Q", y=alt.Y("Expected fill rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1.02])),
            tooltip=["Delivery delay (hours)", alt.Tooltip("Expected fill rate:Q", format=".0%")]), use_container_width=True)
    st.caption("Share of demand actually sold, as a non-increasing function of delivery delay (pool-adjacent-violators fit). The inventory agent uses it to estimate unfulfilled demand.")

    st.markdown("#### 6. Assumptions the data cannot supply")
    asm = pd.DataFrame([{"Category": c, "Shelf life (days)": core.CATEGORY_ASSUMPTIONS.get(c, core.DEFAULT_ASSUMPTION)["shelf_life_days"],
                         "Base waste rate": f"{core.CATEGORY_ASSUMPTIONS.get(c, core.DEFAULT_ASSUMPTION)['base_waste_rate']*100:.1f}%",
                         "Gross margin used": f"{MODEL.margin.get(c, core.DEFAULT_MARGIN)*100:.0f}%",
                         "Unit cost (€)": round(MODEL.ref_price.get(c, 0) * (1 - MODEL.margin.get(c, core.DEFAULT_MARGIN)), 2)} for c in MODEL.categories])
    st.dataframe(asm, use_container_width=True, hide_index=True)
    st.caption("The file has no stock, expiry, waste or unit-cost columns. Shelf life and waste rates are editable constants in retail_brain_core.py; unit cost is derived from the file's gross margin where present. Replace them with store data before operational use.")

# Footer
st.divider()
if st.session_state.get("persistence_warning"):
    st.warning("SQLite audit persistence is unavailable in this environment; decisions remain session-only. Details: " + str(st.session_state.persistence_warning)[:240])
st.caption("Limitations: the bundled data is a dummy operations file, so results show how the tool behaves, not how a real store performs. The demand model is validated on a held-out period (see Data & validation) and its price and promotion effects may be assumptions. "
           "Agents run in one process over an event bus. Decisions are saved to a local SQLite file that is not a managed database and may reset on hosts with ephemeral storage. Forecasts support, and do not replace, manager judgement.")
