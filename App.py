
import random
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Optional model dependencies
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# LIDL STORE OPERATIONS PLATFORM — DEMONSTRATION MVP
#
# All data and events are illustrative.
# This app does not connect to Lidl systems, competitors,
# suppliers, inventory systems, or real-time external feeds.
#
# Managers control whether simulated scenarios are approved.
# No real operational action is executed.
# ============================================================

st.set_page_config(
    page_title="Lidl Store Operations Platform",
    page_icon="🟡",
    layout="wide",
)

CATEGORIES = {
    "Birra": {
        "department": "Beverages",
        "factor": 1.15,
        "price": 1.29,
        "competitor_price": 1.39,
        "promo": 5.0,
        "display": 50.0,
        "temperature": 22.0,
        "traffic": 100.0,
        "stock": 420,
        "lead_time": 2,
        "expiry_days": 120,
        "staffing": 5,
        "unit_cost": 0.72,
        "margin": 0.44,
    },
    "Prodotti da Forno": {
        "department": "Bakery",
        "factor": 0.90,
        "price": 1.49,
        "competitor_price": 1.59,
        "promo": 8.0,
        "display": 65.0,
        "temperature": 20.0,
        "traffic": 100.0,
        "stock": 180,
        "lead_time": 1,
        "expiry_days": 2,
        "staffing": 4,
        "unit_cost": 0.68,
        "margin": 0.54,
    },
    "Ortofrutta": {
        "department": "Fresh Produce",
        "factor": 1.25,
        "price": 2.49,
        "competitor_price": 2.69,
        "promo": 10.0,
        "display": 70.0,
        "temperature": 18.0,
        "traffic": 100.0,
        "stock": 260,
        "lead_time": 1,
        "expiry_days": 3,
        "staffing": 5,
        "unit_cost": 1.42,
        "margin": 0.43,
    },
    "Detersivi": {
        "department": "Cleaning Products",
        "factor": 0.70,
        "price": 3.99,
        "competitor_price": 4.29,
        "promo": 3.0,
        "display": 40.0,
        "temperature": 20.0,
        "traffic": 100.0,
        "stock": 300,
        "lead_time": 3,
        "expiry_days": 365,
        "staffing": 3,
        "unit_cost": 2.20,
        "margin": 0.45,
    },
}

FEATURES = [
    "category_factor",
    "lidl_price",
    "promo_pct",
    "display_pct",
    "temperature_c",
    "traffic_index",
    "competitor_price",
]


# ============================================================
# SESSION STATE
# ============================================================

def initialise_state():
    defaults = {
        "events": [],
        "decision_history": [],
        "approved_event_ids": [],
        "selected_event_id": None,
        "last_auto_scan": None,
        "scan_count": 0,
        "model": None,
        "model_score": None,
        "model_error": None,
        "shap_explainer": None,
        "sound_enabled": False,
        "scenario_counter": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "base_profiles" not in st.session_state:
        st.session_state.base_profiles = {
            category: values.copy()
            for category, values in CATEGORIES.items()
        }


initialise_state()


# ============================================================
# DEMAND MODEL
# ============================================================

@st.cache_resource
def train_demand_model():
    """Train a demonstration model on synthetic historical data."""

    rng = np.random.default_rng(42)
    n = 1800

    category_factors = rng.choice(
        [v["factor"] for v in CATEGORIES.values()],
        size=n,
    )

    lidl_price = rng.uniform(0.8, 5.5, n)
    promo_pct = rng.uniform(0, 35, n)
    display_pct = rng.uniform(20, 100, n)
    temperature_c = rng.uniform(5, 35, n)
    traffic_index = rng.uniform(65, 145, n)
    competitor_price = lidl_price * rng.uniform(0.85, 1.25, n)

    # Synthetic demand-generating process for demonstration only.
    sales = (
        105 * category_factors
        + 1.55 * promo_pct
        + 0.42 * display_pct
        + 0.72 * (traffic_index - 100)
        + 1.15 * (competitor_price - lidl_price) * 10
        + 0.9 * np.maximum(temperature_c - 20, 0)
        - 7.0 * lidl_price
        + rng.normal(0, 14, n)
    )

    sales = np.maximum(sales, 5)

    X = pd.DataFrame({
        "category_factor": category_factors,
        "lidl_price": lidl_price,
        "promo_pct": promo_pct,
        "display_pct": display_pct,
        "temperature_c": temperature_c,
        "traffic_index": traffic_index,
        "competitor_price": competitor_price,
    })

    y = sales

    if XGBOOST_AVAILABLE:
        model = XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )
        model.fit(X, y)
        return model, "XGBoost", X

    # Fallback: least-squares regression using NumPy.
    matrix = np.column_stack([np.ones(len(X)), X.values])
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]

    class LinearFallback:
        def predict(self, frame):
            values = np.column_stack([
                np.ones(len(frame)),
                frame[FEATURES].values,
            ])
            return values @ coefficients

    return LinearFallback(), "Linear regression fallback", X


def ensure_model():
    if st.session_state.model is None:
        try:
            model, model_name, training_data = train_demand_model()
            st.session_state.model = model
            st.session_state.model_name = model_name
            st.session_state.training_data = training_data

            if SHAP_AVAILABLE and XGBOOST_AVAILABLE:
                try:
                    st.session_state.shap_explainer = shap.TreeExplainer(model)
                except Exception:
                    st.session_state.shap_explainer = None

        except Exception as exc:
            st.session_state.model_error = str(exc)


ensure_model()


# ============================================================
# OPERATIONAL PROFILE AND FORECAST
# ============================================================

def effective_profile(category):
    """Apply manager-approved simulated changes to the baseline."""

    profile = st.session_state.base_profiles[category].copy()

    approved_ids = set(st.session_state.approved_event_ids)

    for event in st.session_state.events:
        if event["id"] not in approved_ids:
            continue

        if category not in event["affected_categories"]:
            continue

        changes = event["changes"]

        profile["price"] *= 1 + changes.get("lidl_price_pct", 0) / 100
        profile["competitor_price"] *= (
            1 + changes.get("competitor_price_pct", 0) / 100
        )

        profile["promo"] = float(np.clip(
            profile["promo"] + changes.get("promo_pp", 0),
            0,
            100,
        ))

        profile["display"] = float(np.clip(
            profile["display"] + changes.get("display_pp", 0),
            0,
            100,
        ))

        profile["temperature"] += changes.get("temperature_c", 0)

        profile["traffic"] *= (
            1 + changes.get("traffic_pct", 0) / 100
        )

        profile["stock"] = max(
            0,
            int(profile["stock"] + changes.get("stock_units", 0)),
        )

        profile["lead_time"] = max(
            0,
            int(profile["lead_time"] + changes.get("lead_time_days", 0)),
        )

        profile["expiry_days"] = max(
            0,
            int(profile["expiry_days"] + changes.get("expiry_days", 0)),
        )

        profile["staffing"] = max(
            0,
            int(profile["staffing"] + changes.get("staffing", 0)),
        )

        if changes.get("compliance_issue", False):
            profile["compliance_issue"] = True

    return profile


def predict_demand(category, profile):
    """Predict demand with the trained model."""

    model = st.session_state.model

    if model is None:
        return 100.0

    row = pd.DataFrame([{
        "category_factor": CATEGORIES[category]["factor"],
        "lidl_price": profile["price"],
        "promo_pct": profile["promo"],
        "display_pct": profile["display"],
        "temperature_c": profile["temperature"],
        "traffic_index": profile["traffic"],
        "competitor_price": profile["competitor_price"],
    }], columns=FEATURES)

    try:
        prediction = float(model.predict(row)[0])
        return max(1.0, prediction)
    except Exception:
        return 100.0


def calculate_category_metrics(category, profile):
    """Recalculate demand, stock risk, waste, margin and staffing."""

    demand = predict_demand(category, profile)

    # Illustrative demand horizon: one day.
    stock = profile["stock"]
    lead_time = profile["lead_time"]

    coverage_days = stock / max(demand, 1)
    expected_lead_time_demand = demand * max(lead_time, 1)

    if stock < expected_lead_time_demand:
        stock_status = "Stockout risk"
    elif coverage_days > 10:
        stock_status = "Potential overstock"
    else:
        stock_status = "Stock adequate"

    replenishment_qty = max(
        0,
        int(np.ceil(expected_lead_time_demand * 1.2 - stock)),
    )

    if profile["expiry_days"] <= 2:
        waste_risk = "High"
        markdown_recommendation = "Review markdown and rotation"
    elif profile["expiry_days"] <= 5:
        waste_risk = "Medium"
        markdown_recommendation = "Monitor sell-through"
    else:
        waste_risk = "Low"
        markdown_recommendation = "No immediate expiry action"

    estimated_revenue = demand * profile["price"]
    estimated_cost = demand * CATEGORIES[category]["unit_cost"]
    estimated_margin = estimated_revenue - estimated_cost

    expected_staff_need = max(1, int(np.ceil(demand / 60)))
    staffing_gap = profile["staffing"] - expected_staff_need

    if staffing_gap < 0:
        staffing_status = "Potential understaffing"
    elif staffing_gap >= 2:
        staffing_status = "Potential excess capacity"
    else:
        staffing_status = "Staffing broadly aligned"

    compliance_issue = profile.get("compliance_issue", False)

    if compliance_issue:
        compliance_status = "Review required"
    else:
        compliance_status = "No simulated alert"

    return {
        "forecast_demand": demand,
        "stock": stock,
        "coverage_days": coverage_days,
        "stock_status": stock_status,
        "replenishment_qty": replenishment_qty,
        "waste_risk": waste_risk,
        "markdown_recommendation": markdown_recommendation,
        "estimated_revenue": estimated_revenue,
        "estimated_margin": estimated_margin,
        "staffing_status": staffing_status,
        "staffing_gap": staffing_gap,
        "compliance_status": compliance_status,
    }


def make_dashboard_frame():
    rows = []

    for category in CATEGORIES:
        profile = effective_profile(category)
        metrics = calculate_category_metrics(category, profile)

        rows.append({
            "Category": category,
            "Department": CATEGORIES[category]["department"],
            "Forecast units/day": round(metrics["forecast_demand"]),
            "Stock units": metrics["stock"],
            "Coverage days": round(metrics["coverage_days"], 1),
            "Stock status": metrics["stock_status"],
            "Suggested replenishment": metrics["replenishment_qty"],
            "Waste risk": metrics["waste_risk"],
            "Estimated revenue/day (€)": round(metrics["estimated_revenue"], 2),
            "Estimated margin/day (€)": round(metrics["estimated_margin"], 2),
            "Staffing": metrics["staffing_status"],
            "Compliance": metrics["compliance_status"],
        })

    return pd.DataFrame(rows)


# ============================================================
# SCENARIO GENERATOR
# ============================================================

SCENARIO_TEMPLATES = [
    {
        "name": "Competitor price reduction",
        "description": (
            "A competitor reduces prices while traffic and promotional "
            "conditions may also change."
        ),
        "changes": {
            "competitor_price_pct": -12,
            "traffic_pct": -5,
            "promo_pp": 2,
        },
        "severity": "High",
    },
    {
        "name": "Demand spike and delivery delay",
        "description": (
            "Demand rises while delivery lead time increases, creating "
            "a possible stock availability problem."
        ),
        "changes": {
            "traffic_pct": 18,
            "stock_units": -35,
            "lead_time_days": 2,
            "staffing": -1,
        },
        "severity": "High",
    },
    {
        "name": "Promotion and display opportunity",
        "description": (
            "A proposed promotion and stronger product display may increase "
            "demand and require more inventory."
        ),
        "changes": {
            "promo_pp": 12,
            "display_pp": 20,
            "stock_units": -20,
            "staffing": 1,
        },
        "severity": "Medium",
    },
    {
        "name": "Freshness and waste risk",
        "description": (
            "Shorter remaining shelf life coincides with a possible "
            "demand decline."
        ),
        "changes": {
            "expiry_days": -2,
            "traffic_pct": -8,
            "promo_pp": 5,
        },
        "severity": "High",
    },
    {
        "name": "Hot weather and traffic increase",
        "description": (
            "Higher temperatures and store traffic may change demand, "
            "especially for temperature-sensitive categories."
        ),
        "changes": {
            "temperature_c": 7,
            "traffic_pct": 15,
            "stock_units": -25,
        },
        "severity": "Medium",
    },
    {
        "name": "Cost and margin pressure",
        "description": (
            "Lidl's selling price and competitor pricing change, potentially "
            "affecting demand and commercial positioning."
        ),
        "changes": {
            "lidl_price_pct": 5,
            "competitor_price_pct": -4,
            "promo_pp": -2,
        },
        "severity": "Medium",
    },
    {
        "name": "Staffing and compliance alert",
        "description": (
            "A staffing reduction is combined with a simulated compliance "
            "issue requiring manager review."
        ),
        "changes": {
            "staffing": -2,
            "traffic_pct": 6,
            "compliance_issue": True,
        },
        "severity": "High",
    },
]


def describe_changes(changes):
    labels = {
        "competitor_price_pct": "Competitor price",
        "lidl_price_pct": "Lidl price",
        "promo_pp": "Promotion",
        "display_pp": "Display",
        "temperature_c": "Temperature",
        "traffic_pct": "Store traffic",
        "stock_units": "Stock",
        "lead_time_days": "Delivery lead time",
        "expiry_days": "Remaining shelf life",
        "staffing": "Staffing",
        "compliance_issue": "Compliance",
    }

    descriptions = []

    for key, value in changes.items():
        label = labels.get(key, key)

        if key == "compliance_issue":
            descriptions.append(f"{label}: simulated issue")
        elif key in ("promo_pp", "display_pp"):
            descriptions.append(f"{label}: {value:+g} percentage points")
        elif key in (
            "competitor_price_pct",
            "lidl_price_pct",
            "traffic_pct",
        ):
            descriptions.append(f"{label}: {value:+g}%")
        elif key == "temperature_c":
            descriptions.append(f"{label}: {value:+g}°C")
        elif key == "stock_units":
            descriptions.append(f"{label}: {value:+g} units")
        elif key == "lead_time_days":
            descriptions.append(f"{label}: {value:+g} days")
        elif key == "expiry_days":
            descriptions.append(f"{label}: {value:+g} days")
        elif key == "staffing":
            descriptions.append(f"{label}: {value:+g} people")

    return descriptions


def create_scenario():
    """Create a simulated event affecting one or more categories."""

    template = random.choice(SCENARIO_TEMPLATES)

    category_count = random.choices(
        population=[1, 2, 3, 4],
        weights=[35, 35, 20, 10],
        k=1,
    )[0]

    affected_categories = random.sample(
        list(CATEGORIES.keys()),
        k=category_count,
    )

    event_id = str(uuid.uuid4())[:8]

    # Copy changes so the stored scenario is independent.
    changes = template["changes"].copy()

    # Keep the event object immutable after creation in this demo.
    before_after = {}

    for category in affected_categories:
        before_profile = effective_profile(category)
        before_metrics = calculate_category_metrics(
            category,
            before_profile,
        )

        after_profile = before_profile.copy()

        after_profile["price"] *= (
            1 + changes.get("lidl_price_pct", 0) / 100
        )
        after_profile["competitor_price"] *= (
            1 + changes.get("competitor_price_pct", 0) / 100
        )
        after_profile["promo"] = float(np.clip(
            after_profile["promo"] + changes.get("promo_pp", 0),
            0,
            100,
        ))
        after_profile["display"] = float(np.clip(
            after_profile["display"] + changes.get("display_pp", 0),
            0,
            100,
        ))
        after_profile["temperature"] += changes.get("temperature_c", 0)
        after_profile["traffic"] *= (
            1 + changes.get("traffic_pct", 0) / 100
        )
        after_profile["stock"] = max(
            0,
            int(after_profile["stock"] + changes.get("stock_units", 0)),
        )
        after_profile["lead_time"] = max(
            0,
            int(
                after_profile["lead_time"]
                + changes.get("lead_time_days", 0)
            ),
        )
        after_profile["expiry_days"] = max(
            0,
            int(
                after_profile["expiry_days"]
                + changes.get("expiry_days", 0)
            ),
        )
        after_profile["staffing"] = max(
            0,
            int(after_profile["staffing"] + changes.get("staffing", 0)),
        )

        if changes.get("compliance_issue", False):
            after_profile["compliance_issue"] = True

        after_metrics = calculate_category_metrics(
            category,
            after_profile,
        )

        before_after[category] = {
            "before_demand": before_metrics["forecast_demand"],
            "after_demand": after_metrics["forecast_demand"],
            "before_stock_status": before_metrics["stock_status"],
            "after_stock_status": after_metrics["stock_status"],
            "before_waste_risk": before_metrics["waste_risk"],
            "after_waste_risk": after_metrics["waste_risk"],
            "before_staffing": before_metrics["staffing_status"],
            "after_staffing": after_metrics["staffing_status"],
            "before_compliance": before_metrics["compliance_status"],
            "after_compliance": after_metrics["compliance_status"],
            "before_margin": before_metrics["estimated_margin"],
            "after_margin": after_metrics["estimated_margin"],
            "before_replenishment": before_metrics["replenishment_qty"],
            "after_replenishment": after_metrics["replenishment_qty"],
        }

    event = {
        "id": event_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": template["name"],
        "description": template["description"],
        "severity": template["severity"],
        "status": "Pending",
        "affected_categories": affected_categories,
        "changes": changes,
        "impact": before_after,
        "decision_note": "",
    }

    st.session_state.events.insert(0, event)
    st.session_state.scenario_counter += 1

    return event


def scan_scenarios(number_of_events=1):
    created = []

    for _ in range(number_of_events):
        created.append(create_scenario())

    st.session_state.scan_count += 1
    return created


# ============================================================
# DECISION WORKFLOW
# ============================================================

def get_event(event_id):
    for event in st.session_state.events:
        if event["id"] == event_id:
            return event
    return None


def record_decision(event, decision, note=""):
    event["status"] = decision
    event["decision_note"] = note
    event["decided_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if decision == "Approved":
        if event["id"] not in st.session_state.approved_event_ids:
            st.session_state.approved_event_ids.append(event["id"])

    elif decision == "Rejected":
        if event["id"] in st.session_state.approved_event_ids:
            st.session_state.approved_event_ids.remove(event["id"])

    st.session_state.decision_history.insert(0, {
        "Timestamp": event["decided_at"],
        "Event ID": event["id"],
        "Scenario": event["name"],
        "Decision": decision,
        "Affected categories": ", ".join(event["affected_categories"]),
        "Manager note": note,
    })


def open_review(event_id):
    st.session_state.selected_event_id = event_id
    review_dialog(event_id)


@st.dialog("Review operational scenario", width="large")
def review_dialog(event_id):
    event = get_event(event_id)

    if event is None:
        st.error("This scenario could not be found.")
        return

    st.markdown(f"### {event['name']}")
    st.caption(
        f"Scenario ID: {event['id']} · Created: {event['created_at']}"
    )

    if event["severity"] == "High":
        st.error(f"Severity: {event['severity']}")
    else:
        st.warning(f"Severity: {event['severity']}")

    st.write(event["description"])
    st.write("**Affected categories:** " + ", ".join(
        event["affected_categories"]
    ))

    st.write("**Simulated variable changes**")
    for item in describe_changes(event["changes"]):
        st.write(f"- {item}")

    st.write("**Projected impact by category**")

    impact_rows = []

    for category, impact in event["impact"].items():
        demand_change = (
            impact["after_demand"] - impact["before_demand"]
        )

        impact_rows.append({
            "Category": category,
            "Demand before": round(impact["before_demand"]),
            "Demand after": round(impact["after_demand"]),
            "Demand change": round(demand_change),
            "Stock risk after": impact["after_stock_status"],
            "Waste risk after": impact["after_waste_risk"],
            "Replenishment after": impact["after_replenishment"],
            "Staffing after": impact["after_staffing"],
            "Compliance after": impact["after_compliance"],
        })

    st.dataframe(
        pd.DataFrame(impact_rows),
        use_container_width=True,
        hide_index=True,
    )

    note = st.text_area(
        "Manager note (optional)",
        key=f"note_{event_id}",
        placeholder="Explain the decision or any required follow-up...",
    )

    left, middle, right = st.columns(3)

    with left:
        if st.button(
            "Approve scenario",
            key=f"approve_{event_id}",
            type="primary",
            use_container_width=True,
        ):
            record_decision(event, "Approved", note)
            st.session_state.selected_event_id = None
            st.rerun()

    with middle:
        if st.button(
            "Reject scenario",
            key=f"reject_{event_id}",
            use_container_width=True,
        ):
            record_decision(event, "Rejected", note)
            st.session_state.selected_event_id = None
            st.rerun()

    with right:
        if st.button(
            "Defer review",
            key=f"defer_{event_id}",
            use_container_width=True,
        ):
            record_decision(event, "Deferred", note)
            st.session_state.selected_event_id = None
            st.rerun()

    st.caption(
        "Approval changes this demo's simulated operational state only. "
        "It does not place orders, change prices, or contact suppliers."
    )


# ============================================================
# SOUND ALERT
# ============================================================

def render_sound_alert(pending_count):
    """
    The browser may block automatic sound until the user enables it.
    The embedded control enables sound in this dashboard.
    """

    html = f"""
    <div style="
        font-family: Arial, sans-serif;
        padding: 4px 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    ">
      <button id="soundButton" style="
          background: #1f2937;
          color: white;
          border: 0;
          padding: 9px 13px;
          border-radius: 7px;
          cursor: pointer;
          font-size: 13px;
      ">Enable / test sound</button>
      <span id="soundStatus" style="font-size: 12px; color: #666;">
          Browser sound may require permission or a user click.
      </span>
    </div>

    <script>
    const pendingCount = {int(pending_count)};
    const soundKey = "lidl_demo_sound_enabled";
    const countKey = "lidl_demo_previous_pending_count";

    function playBeep() {{
        try {{
            const AudioContextClass =
                window.AudioContext || window.webkitAudioContext;

            if (!AudioContextClass) return;

            const context = new AudioContextClass();

            const playTone = (start, frequency) => {{
                const oscillator = context.createOscillator();
                const gain = context.createGain();

                oscillator.type = "sine";
                oscillator.frequency.value = frequency;

                gain.gain.setValueAtTime(0.0001, start);
                gain.gain.exponentialRampToValueAtTime(0.15, start + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.22);

                oscillator.connect(gain);
                gain.connect(context.destination);

                oscillator.start(start);
                oscillator.stop(start + 0.23);
            }};

            context.resume().then(() => {{
                const now = context.currentTime;
                playTone(now, 880);
                playTone(now + 0.24, 660);
            }});
        }} catch (error) {{
            console.log("Sound alert unavailable", error);
        }}
    }}

    const button = document.getElementById("soundButton");
    const status = document.getElementById("soundStatus");

    button.addEventListener("click", () => {{
        localStorage.setItem(soundKey, "true");
        playBeep();
        status.textContent = "Sound enabled in this browser where permitted.";
        button.textContent = "Test sound again";
    }});

    const previouslyEnabled = localStorage.getItem(soundKey) === "true";
    const previousCount = parseInt(localStorage.getItem(countKey) || "0");

    if (previouslyEnabled && pendingCount > previousCount && pendingCount > 0) {{
        playBeep();
    }}

    localStorage.setItem(countKey, String(pendingCount));

    if (previouslyEnabled) {{
        status.textContent = "Sound alerts enabled where browser policy permits.";
        button.textContent = "Test sound";
    }}
    </script>
    """

    components.html(html, height=48)


# ============================================================
# HEADER AND SIDEBAR
# ============================================================

st.title("🟡 Lidl Multi-Agent Store Operations Platform")
st.caption(
    "Demonstration MVP · Predictive demand · Operational risk · "
    "Explainability · Manager-controlled scenario decisions"
)

with st.sidebar:
    st.header("Control centre")

    st.markdown("**Scenario engine**")
    auto_monitoring = st.toggle(
        "Auto-generate demo events",
        value=False,
        help=(
            "When enabled, the app periodically creates illustrative "
            "events. It does not monitor live Lidl or competitor data."
        ),
        key="auto_monitoring",
    )

    events_per_scan = st.selectbox(
        "Events per manual scan",
        options=[1, 2, 3],
        index=1,
    )

    if st.button(
        "Run scenario scan",
        type="primary",
        use_container_width=True,
    ):
        created = scan_scenarios(events_per_scan)
        st.session_state.last_auto_scan = datetime.now()
        st.success(f"Generated {len(created)} simulated scenario(s).")
        st.rerun()

    if st.button("Reset demo scenarios", use_container_width=True):
        st.session_state.events = []
        st.session_state.decision_history = []
        st.session_state.approved_event_ids = []
        st.session_state.selected_event_id = None
        st.session_state.scenario_counter = 0
        st.session_state.last_auto_scan = None
        st.rerun()

    st.divider()

    st.markdown("**Model**")
    st.write(
        st.session_state.get("model_name", "Model unavailable")
    )

    if st.session_state.model_error:
        st.error(st.session_state.model_error)

    st.caption(
        "The model is trained on synthetic demonstration data."
    )


# ============================================================
# OPTIONAL PERIODIC SCENARIO MONITOR
# ============================================================

# Streamlit fragments periodically rerun only this monitor.
# Auto-generated scenarios are still simulated, not live alerts.

if hasattr(st, "fragment"):

    @st.fragment(run_every="30s")
    def periodic_scenario_monitor():
        if not st.session_state.get("auto_monitoring", False):
            return

        now = datetime.now()
        last_scan = st.session_state.last_auto_scan

        if last_scan is None:
            st.session_state.last_auto_scan = now
            return

        elapsed = (now - last_scan).total_seconds()

        # Limit automatic scenario generation to once every 120 seconds.
        if elapsed >= 120:
            create_scenario()
            st.session_state.last_auto_scan = now

            # Refresh the main dashboard after the fragment adds an event.
            st.rerun(scope="app")

    periodic_scenario_monitor()


# ============================================================
# ALERT COUNTS AND NOTIFICATION BANNER
# ============================================================

unresolved_events = [
    event for event in st.session_state.events
    if event["status"] in ("Pending", "Deferred")
]

pending_count = sum(
    event["status"] == "Pending"
    for event in st.session_state.events
)

deferred_count = sum(
    event["status"] == "Deferred"
    for event in st.session_state.events
)

approved_count = sum(
    event["status"] == "Approved"
    for event in st.session_state.events
)

rejected_count = sum(
    event["status"] == "Rejected"
    for event in st.session_state.events
)

high_priority_count = sum(
    event["severity"] == "High"
    and event["status"] in ("Pending", "Deferred")
    for event in st.session_state.events
)

if unresolved_events:
    st.warning(
        f"🔔 **Manager attention required:** {len(unresolved_events)} "
        f"unresolved scenario(s), including {high_priority_count} "
        f"high-priority item(s). Review the scenario queue below."
    )
else:
    st.success(
        "✓ No unresolved scenarios. Run a scan to generate "
        "illustrative operational events."
    )

metric_cols = st.columns(5)

metric_cols[0].metric("Pending", pending_count)
metric_cols[1].metric("Deferred", deferred_count)
metric_cols[2].metric("Approved", approved_count)
metric_cols[3].metric("Rejected", rejected_count)
metric_cols[4].metric("High priority", high_priority_count)

render_sound_alert(len(unresolved_events))


# ============================================================
# TOP-LEVEL DASHBOARD
# ============================================================

dashboard_df = make_dashboard_frame()

tab_overview, tab_scenarios, tab_agents, tab_history, tab_model = st.tabs([
    "📊 Store overview",
    "🔔 Scenario centre",
    "🤖 Operational agents",
    "🧾 Decision history",
    "🧠 Model explainability",
])


# ============================================================
# TAB 1 — STORE OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("All-category operating view")
    st.caption(
        "Metrics are calculated for all four categories. Approved "
        "scenarios alter the simulated state; pending scenarios do not."
    )

    c1, c2, c3, c4 = st.columns(4)

    total_forecast = dashboard_df["Forecast units/day"].sum()
    total_stock = dashboard_df["Stock units"].sum()
    total_revenue = dashboard_df["Estimated revenue/day (€)"].sum()
    total_margin = dashboard_df["Estimated margin/day (€)"].sum()

    c1.metric("Forecast units/day", f"{total_forecast:,.0f}")
    c2.metric("Stock units", f"{total_stock:,.0f}")
    c3.metric("Estimated revenue/day", f"€{total_revenue:,.2f}")
    c4.metric("Estimated margin/day", f"€{total_margin:,.2f}")

    st.dataframe(
        dashboard_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Category-level operating conditions")

    selected_category = st.selectbox(
        "Inspect a category",
        options=list(CATEGORIES.keys()),
        key="overview_category",
    )

    profile = effective_profile(selected_category)
    metrics = calculate_category_metrics(selected_category, profile)

    detail_cols = st.columns(4)
    detail_cols[0].metric(
        "Forecast demand",
        f"{metrics['forecast_demand']:.0f} units/day",
    )
    detail_cols[1].metric(
        "Stock coverage",
        f"{metrics['coverage_days']:.1f} days",
    )
    detail_cols[2].metric(
        "Suggested replenishment",
        f"{metrics['replenishment_qty']} units",
    )
    detail_cols[3].metric(
        "Estimated daily margin",
        f"€{metrics['estimated_margin']:.2f}",
    )

    st.write("**Current simulated inputs**")

    input_df = pd.DataFrame([{
        "Lidl price (€)": round(profile["price"], 2),
        "Competitor price (€)": round(profile["competitor_price"], 2),
        "Promotion (%)": round(profile["promo"], 1),
        "Display (%)": round(profile["display"], 1),
        "Temperature (°C)": round(profile["temperature"], 1),
        "Traffic index": round(profile["traffic"], 1),
        "Stock (units)": profile["stock"],
        "Lead time (days)": profile["lead_time"],
        "Shelf life remaining (days)": profile["expiry_days"],
        "Staff on shift": profile["staffing"],
    }])

    st.dataframe(input_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — SCENARIO CENTRE
# ============================================================

with tab_scenarios:
    st.subheader("Scenario review queue")

    st.write(
        "Review proposed events, compare projected impacts, and decide "
        "whether to approve, reject, or defer each scenario."
    )

    if not unresolved_events:
        st.info(
            "No unresolved scenarios. Use **Run scenario scan** in the "
            "sidebar to create illustrative events."
        )

    for event in unresolved_events:
        with st.container(border=True):
            left, right = st.columns([4, 1])

            with left:
                st.markdown(f"**{event['name']}**")

                severity_label = (
                    "🔴 High"
                    if event["severity"] == "High"
                    else "🟠 Medium"
                )

                st.caption(
                    f"{severity_label} · ID {event['id']} · "
                    f"{event['created_at']}"
                )

                st.write(event["description"])

                st.write(
                    "**Affected categories:** "
                    + ", ".join(event["affected_categories"])
                )

                st.write("**Variables:**")
                st.write(" · ".join(describe_changes(event["changes"])))

                if event["status"] == "Deferred":
                    st.info("This scenario has been deferred for later review.")

            with right:
                if st.button(
                    "Review",
                    key=f"review_{event['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    open_review(event["id"])

    st.divider()
    st.subheader("Scenario audit table")

    if st.session_state.events:
        audit_rows = []

        for event in st.session_state.events:
            audit_rows.append({
                "Created": event["created_at"],
                "ID": event["id"],
                "Scenario": event["name"],
                "Severity": event["severity"],
                "Status": event["status"],
                "Categories": ", ".join(event["affected_categories"]),
                "Variables changed": "; ".join(
                    describe_changes(event["changes"])
                ),
            })

        st.dataframe(
            pd.DataFrame(audit_rows),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = pd.DataFrame(audit_rows).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download scenario audit CSV",
            data=csv_data,
            file_name="lidl_scenario_audit.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 3 — OPERATIONAL AGENTS
# ============================================================

with tab_agents:
    st.subheader("Operational agent recommendations")

    st.caption(
        "These agents are rule-based demonstration modules that use the "
        "forecast and simulated operating inputs. They are not autonomous "
        "Lidl systems."
    )

    agent_category = st.selectbox(
        "Category for detailed agent output",
        options=list(CATEGORIES.keys()),
        key="agent_category",
    )

    agent_profile = effective_profile(agent_category)
    agent_metrics = calculate_category_metrics(
        agent_category,
        agent_profile,
    )

    agents = [
        (
            "1. Demand forecasting",
            f"Estimated demand: {agent_metrics['forecast_demand']:.0f} units/day.",
        ),
        (
            "2. Inventory risk",
            f"{agent_metrics['stock_status']}. "
            f"Stock coverage: {agent_metrics['coverage_days']:.1f} days.",
        ),
        (
            "3. Commercial intelligence",
            f"Lidl price €{agent_profile['price']:.2f}; "
            f"competitor reference €{agent_profile['competitor_price']:.2f}.",
        ),
        (
            "4. Replenishment",
            f"Illustrative replenishment suggestion: "
            f"{agent_metrics['replenishment_qty']} units.",
        ),
        (
            "5. Freshness and waste",
            f"Waste risk: {agent_metrics['waste_risk']}. "
            f"{agent_metrics['markdown_recommendation']}.",
        ),
        (
            "6. Workforce planning",
            f"{agent_metrics['staffing_status']}.",
        ),
        (
            "7. Promotion and display",
            f"Promotion: {agent_profile['promo']:.1f}%; "
            f"display index: {agent_profile['display']:.1f}%.",
        ),
        (
            "8. Profitability",
            f"Estimated daily margin: "
            f"€{agent_metrics['estimated_margin']:.2f}.",
        ),
        (
            "9. Compliance and task management",
            f"Status: {agent_metrics['compliance_status']}. "
            "Any simulated compliance alert requires manager review.",
        ),
    ]

    for title, description in agents:
        with st.expander(title):
            st.write(description)

    st.info(
        "Recommendations are decision support only. Confirm the relevant "
        "business constraints before acting on any suggested quantity, "
        "staffing level, markdown, or price."
    )


# ============================================================
# TAB 4 — DECISION HISTORY
# ============================================================

with tab_history:
    st.subheader("Manager decision history")

    if st.session_state.decision_history:
        history_df = pd.DataFrame(
            st.session_state.decision_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        history_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download manager decision history",
            data=history_csv,
            file_name="lidl_manager_decision_history.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No decisions have been recorded. Review a scenario and "
            "approve, reject, or defer it to create a record."
        )

    st.subheader("Approved scenario effects")

    approved_events = [
        event for event in st.session_state.events
        if event["id"] in st.session_state.approved_event_ids
    ]

    if approved_events:
        for event in approved_events:
            st.write(
                f"**{event['name']}** — "
                f"{', '.join(event['affected_categories'])}"
            )
            st.caption(
                "Applied to simulated dashboard state only. "
                "No real operational system was changed."
            )
    else:
        st.info("No approved scenarios are currently applied.")


# ============================================================
# TAB 5 — MODEL EXPLAINABILITY
# ============================================================

with tab_model:
    st.subheader("Demand model and explainability")

    st.write(
        f"**Model in use:** "
        f"{st.session_state.get('model_name', 'Unavailable')}"
    )

    st.write(
        "The model is trained on synthetic data generated inside this "
        "application. It is not a validated production forecasting model."
    )

    st.write("**Features used by the demand model**")

    feature_descriptions = {
        "category_factor": "Synthetic category-level demand factor",
        "lidl_price": "Simulated Lidl selling price",
        "promo_pct": "Promotion percentage",
        "display_pct": "Product display index",
        "temperature_c": "Simulated temperature",
        "traffic_index": "Store traffic index",
        "competitor_price": "Simulated competitor reference price",
    }

    st.dataframe(
        pd.DataFrame([
            {
                "Feature": feature,
                "Description": feature_descriptions[feature],
            }
            for feature in FEATURES
        ]),
        use_container_width=True,
        hide_index=True,
    )

    if (
        SHAP_AVAILABLE
        and XGBOOST_AVAILABLE
        and st.session_state.shap_explainer is not None
    ):
        st.markdown("**SHAP global feature importance**")

        try:
            training_data = st.session_state.training_data
            sample = training_data.sample(
                n=min(100, len(training_data)),
                random_state=42,
            )

            shap_values = st.session_state.shap_explainer(
                sample[FEATURES]
            )

            values = np.asarray(shap_values.values)

            importance = pd.DataFrame({
                "Feature": FEATURES,
                "Mean absolute SHAP value": np.mean(
                    np.abs(values),
                    axis=0,
                ),
            }).sort_values(
                "Mean absolute SHAP value",
                ascending=False,
            )

            st.bar_chart(
                importance.set_index("Feature")
            )

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.warning(
                "SHAP explanation could not be calculated for this run: "
                f"{exc}"
            )
    else:
        st.info(
            "SHAP is unavailable or the fallback regression model is in use. "
            "Install SHAP and XGBoost to enable the model's SHAP importance "
            "view."
        )

    st.caption(
        "Model outputs are illustrative. Validate on representative "
        "historical data and assess forecast accuracy before any real use."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lidl Multi-Agent Store Operations Platform — Demonstration MVP. "
    "Synthetic data, simulated scenarios, and illustrative recommendations "
    "only. Not affiliated with or connected to Lidl operational systems."
)
