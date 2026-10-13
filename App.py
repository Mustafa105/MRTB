import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ============================================================
# Lidl Multi-Agent Store Operations Platform — Demonstration MVP
# IMPORTANT: All generated data and operational recommendations
# are illustrative. This app does not connect to Lidl systems.
# ============================================================

st.set_page_config(
    page_title="Lidl Multi-Agent Store Operations",
    page_icon="🛒",
    layout="wide",
)

st.title(" Lidl Store Operations — Multi Agent Retail Brain MVP")
st.caption(
    "Demonstration only: synthetic data, illustrative coefficients, and simulated recommendations. "
    "No orders, price changes, staff changes, or compliance actions are executed."
)

CATEGORIES = [
    "Birra (Beverages)",
    "Prodotti da Forno (Bakery)",
    "Ortofrutta (Fresh Produce)",
    "Detersivi (Cleaning Products)",
]

FEATURE_COLUMNS = [
    "Price_Euro",
    "Is_Promo_Volantino",
    "In_Store_Display",
    "Temperature_C",
    "Store_Traffic_Index",
    "Is_Weekend",
    "Competitor_Price_Euro",
    "Competitor_Promo",
]

FEATURE_LABELS = {
    "Price_Euro": "Lidl Price",
    "Is_Promo_Volantino": "Lidl Flyer Promotion",
    "In_Store_Display": "Endcap Display",
    "Temperature_C": "Temperature",
    "Store_Traffic_Index": "Store Foot Traffic",
    "Is_Weekend": "Weekend",
    "Competitor_Price_Euro": "Competitor Price",
    "Competitor_Promo": "Competitor Promotion",
}


@st.cache_data
def generate_lidl_data():
    """Generate deterministic synthetic history for demonstration purposes."""
    rng = np.random.default_rng(42)
    days = 240
    all_data = []

    for category in CATEGORIES:
        frame = pd.DataFrame({
            "Category": category,
            "Price_Euro": rng.uniform(0.80, 4.50, days),
            "Is_Promo_Volantino": rng.choice([0, 1], days, p=[0.75, 0.25]),
            "In_Store_Display": rng.choice([0, 1], days, p=[0.85, 0.15]),
            "Temperature_C": rng.uniform(2, 36, days),
            "Store_Traffic_Index": rng.uniform(50, 180, days),
            "Is_Weekend": rng.choice([0, 1], days, p=[5 / 7, 2 / 7]),
            "Competitor_Price_Euro": rng.uniform(0.80, 4.50, days),
            "Competitor_Promo": rng.choice([0, 1], days, p=[0.75, 0.25]),
        })

        # Illustrative effects only; these are not estimated from Lidl data.
        price_gap = frame["Competitor_Price_Euro"] - frame["Price_Euro"]
        competitor_effect = price_gap * 65 - frame["Competitor_Promo"] * 90

        if "Birra" in category:
            baseline = (
                400 + frame["Store_Traffic_Index"] * 4
                + frame["Temperature_C"] * 12
                + frame["Is_Promo_Volantino"] * 500
                + frame["In_Store_Display"] * 80 + competitor_effect
            )
        elif "Forno" in category:
            baseline = (
                600 + frame["Store_Traffic_Index"] * 5
                - frame["Temperature_C"] * 2
                + frame["In_Store_Display"] * 300
                + frame["Is_Promo_Volantino"] * 80 + competitor_effect
            )
        elif "Ortofrutta" in category:
            baseline = (
                500 + frame["Store_Traffic_Index"] * 3
                + frame["Temperature_C"] * 5
                - frame["Price_Euro"] * 80
                + frame["In_Store_Display"] * 60 + competitor_effect
            )
        else:
            baseline = (
                200 + frame["Store_Traffic_Index"] * 2
                + frame["Is_Promo_Volantino"] * 400
                + frame["In_Store_Display"] * 60 + competitor_effect
            )

        weekend_effect = frame["Is_Weekend"] * 50
        noise = rng.normal(0, 35, days)
        frame["Sales_Volume"] = np.maximum(
            np.round(baseline + weekend_effect + noise).astype(int), 10
        )
        # Synthetic commercial values used only for illustrative metrics.
        frame["Unit_Cost_Euro"] = np.maximum(
            frame["Price_Euro"] * rng.uniform(0.48, 0.78, days), 0.05
        )
        frame["Waste_Units"] = rng.poisson(8, days)
        all_data.append(frame)

    return pd.concat(all_data, ignore_index=True)


df_master = generate_lidl_data()

# --------------------------- Sidebar ---------------------------
st.sidebar.header("🎛️ Store Scenario")

selected_cat = st.sidebar.selectbox("Product category", CATEGORIES)
s_traffic = st.sidebar.slider("Store traffic index", 50, 200, 140)
s_weekend = st.sidebar.selectbox("Day type", ["Weekday", "Weekend"])
s_temp = st.sidebar.slider("External temperature (°C)", 0, 42, 24)

st.sidebar.subheader("Lidl merchandising")
s_price = st.sidebar.slider("Lidl retail price (€)", 0.50, 5.00, 1.49, 0.01)
s_promo = st.sidebar.selectbox("Lidl flyer promotion?", ["No", "Yes"])
s_display = st.sidebar.selectbox("Special display?", ["No", "Yes"])

st.sidebar.subheader("Competitor intelligence")
competitor_price = st.sidebar.slider("Competitor retail price (€)", 0.50, 5.00, 1.59, 0.01)
competitor_promo = st.sidebar.selectbox("Competitor promotion?", ["No", "Yes"])

st.sidebar.subheader("Inventory & deliveries")
on_shelf_stock = st.sidebar.number_input("On-shelf stock (units)", 0, 100000, 120, 10)
backroom_stock = st.sidebar.number_input("Backroom stock (units)", 0, 100000, 300, 10)
incoming_delivery = st.sidebar.number_input("Incoming delivery before demand period (units)", 0, 100000, 100, 10)
lead_time_days = st.sidebar.slider("Supplier lead time (days)", 0, 14, 2)
safety_stock_units = st.sidebar.number_input("Target safety stock (units)", 0, 100000, 100, 10)

st.sidebar.subheader("Freshness & waste")
short_dated_units = st.sidebar.number_input("Units nearing expiry (next 48h)", 0, 100000, 35, 5)
estimated_waste_rate = st.sidebar.slider("Estimated waste rate (%)", 0, 50, 5)

st.sidebar.subheader("Workforce")
expected_hourly_customers = st.sidebar.slider("Expected peak-hour customers", 10, 500, 180, 10)
scheduled_staff = st.sidebar.slider("Staff scheduled during peak hour", 1, 30, 8)
customers_per_staff_hour = st.sidebar.slider("Planning assumption: customers per staff-hour", 5, 60, 25)

st.sidebar.subheader("Profitability")
unit_cost = st.sidebar.number_input("Estimated unit cost (€)", min_value=0.01, max_value=1000.0, value=0.85, step=0.05)
markdown_percent = st.sidebar.slider("Potential markdown (%)", 0, 60, 20)
promotion_cost = st.sidebar.number_input("Estimated promotion cost (€)", min_value=0.0, value=50.0, step=10.0)

st.sidebar.subheader("Store tasks & compliance")
overdue_tasks = st.sidebar.number_input("Overdue store tasks", 0, 500, 2)
open_compliance_issues = st.sidebar.number_input("Open compliance issues", 0, 500, 0)
temperature_check_due = st.sidebar.selectbox("Required temperature check completed?", ["Yes", "No"])
cleaning_check_due = st.sidebar.selectbox("Required cleaning check completed?", ["Yes", "No"])

# -------------------- Train demand model --------------------
df_filtered = df_master[df_master["Category"] == selected_cat].copy()
X = df_filtered[FEATURE_COLUMNS]
y = df_filtered["Sales_Volume"]

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.06,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1,
)
model.fit(X, y)

input_row = pd.DataFrame([{
    "Price_Euro": s_price,
    "Is_Promo_Volantino": int(s_promo == "Yes"),
    "In_Store_Display": int(s_display == "Yes"),
    "Temperature_C": s_temp,
    "Store_Traffic_Index": s_traffic,
    "Is_Weekend": int(s_weekend == "Weekend"),
    "Competitor_Price_Euro": competitor_price,
    "Competitor_Promo": int(competitor_promo == "Yes"),
}], columns=FEATURE_COLUMNS)

predicted_sales = max(0, int(round(float(model.predict(input_row)[0]))))
historical_average = float(y.mean())
price_gap = float(s_price - competitor_price)
total_available_stock = int(on_shelf_stock + backroom_stock + incoming_delivery)
projected_remaining_stock = total_available_stock - predicted_sales

# SHAP is used to explain the model's prediction, not to establish causality.
explainer = shap.TreeExplainer(model)
shap_result = explainer(input_row)
shap_values = np.asarray(shap_result.values).reshape(-1)
impact_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Feature label": [FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
    "SHAP contribution": shap_values,
    "Absolute contribution": np.abs(shap_values),
}).sort_values("Absolute contribution", ascending=False)

# --------------------- Agent calculations ---------------------
# Agent 1: Demand and XAI
demand_delta = predicted_sales - historical_average
top_driver = impact_df.iloc[0]

# Agent 2: Inventory and logistics
stock_after_sales = projected_remaining_stock
replenishment_target = max(
    0, predicted_sales + int(safety_stock_units)
    - int(on_shelf_stock + backroom_stock + incoming_delivery)
)
inventory_status = (
    "CRITICAL SHORTAGE" if stock_after_sales < 0
    else "LOW BUFFER" if stock_after_sales < safety_stock_units
    else "STABLE"
)

# Agent 3: Commercial pricing and competitor intelligence
if price_gap > 0.10:
    competitive_position = "Lidl price is above the entered competitor price"
elif price_gap < -0.10:
    competitive_position = "Lidl price is below the entered competitor price"
else:
    competitive_position = "Entered prices are approximately aligned"

# Agent 4: Replenishment optimization (heuristic)
daily_demand = max(1.0, predicted_sales)
lead_time_demand = int(np.ceil(daily_demand * lead_time_days))
recommended_order = max(
    0,
    lead_time_demand + int(safety_stock_units)
    - int(on_shelf_stock + backroom_stock + incoming_delivery),
)
replenishment_message = (
    f"Review a replenishment quantity of approximately {recommended_order:,} units "
    f"using the entered {lead_time_days}-day lead time and safety-stock target."
    if recommended_order > 0
    else "Entered stock and incoming delivery meet the simple lead-time demand plus safety-stock rule."
)

# Agent 5: Freshness, waste and markdown
short_dated_share = short_dated_units / max(1, total_available_stock)
markdown_rate = markdown_percent / 100
estimated_markdown_revenue = short_dated_units * s_price * (1 - markdown_rate)
waste_risk = (
    "HIGH" if short_dated_units >= max(20, predicted_sales * 0.15)
    else "MODERATE" if short_dated_units > 0
    else "LOW"
)

# Agent 6: Workforce planning
required_staff = int(np.ceil(expected_hourly_customers / max(1, customers_per_staff_hour)))
staff_gap = required_staff - scheduled_staff
staff_status = "UNDERSTAFFED RISK" if staff_gap > 0 else "COVERAGE ESTIMATE MET"

# Agent 7: Promotion performance (scenario estimate)
lidl_promo_active = s_promo == "Yes"
competitor_promo_active = competitor_promo == "Yes"
promo_flags = []
if lidl_promo_active:
    promo_flags.append("Lidl flyer promotion is active in the scenario.")
if competitor_promo_active:
    promo_flags.append("Competitor promotion is active in the scenario.")
if lidl_promo_active and competitor_promo_active:
    promo_advice = "Compare incremental volume and margin; both promotions are active in this scenario."
elif competitor_promo_active and not lidl_promo_active:
    promo_advice = "Competitor promotion is active. Evaluate a response only with margin and elasticity evidence."
elif lidl_promo_active:
    promo_advice = "Evaluate Lidl promotion uplift against baseline sales and margin."
else:
    promo_advice = "No promotion is entered. Compare baseline, promotional uplift, and margin before activation."

# Agent 8: Profitability and store performance
estimated_revenue = predicted_sales * s_price
estimated_gross_profit = predicted_sales * (s_price - unit_cost)
estimated_promo_net = estimated_gross_profit - promotion_cost if lidl_promo_active else estimated_gross_profit
estimated_waste_cost = short_dated_units * unit_cost * (estimated_waste_rate / 100)
profitability_status = "NEGATIVE UNIT MARGIN" if s_price < unit_cost else "POSITIVE UNIT MARGIN"

# Agent 9: Compliance and task management
compliance_tasks = []
if temperature_check_due == "No":
    compliance_tasks.append("Complete the required temperature check and record the result.")
if cleaning_check_due == "No":
    compliance_tasks.append("Complete the required cleaning checklist.")
if overdue_tasks > 0:
    compliance_tasks.append(f"Review and assign {int(overdue_tasks)} overdue store task(s).")
if open_compliance_issues > 0:
    compliance_tasks.append(f"Escalate or resolve {int(open_compliance_issues)} open compliance issue(s).")
compliance_status = "ACTION REQUIRED" if compliance_tasks else "NO OPEN ITEMS ENTERED"

# ------------------------ Dashboard ------------------------
st.subheader("Scenario overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Forecast demand", f"{predicted_sales:,} units", f"{demand_delta:+,.0f} vs category average")
m2.metric("Total available stock", f"{total_available_stock:,} units")
m3.metric("Projected remaining stock", f"{projected_remaining_stock:+,} units")
m4.metric("Inventory status", inventory_status)

st.caption("Choose the action below to display all nine agent reports.")
if "run_agents" not in st.session_state:
    st.session_state.run_agents = True
if st.button("🔄 Run / refresh nine-agent analysis", type="primary"):
    st.session_state.run_agents = True

if st.session_state.run_agents:
    st.divider()
    st.subheader("🤖 Nine-Agent Operations Dashboard")

    # Three columns per row for readable side-by-side desktop layout.
    row1 = st.columns(3, gap="medium")

    # Agent 1
    with row1[0]:
        with st.container(border=True):
            st.markdown("### 🛰️ Agent 1 — Demand & XAI")
            st.metric("Forecast demand", f"{predicted_sales:,} units")
            st.caption(f"Historical category average: {historical_average:,.0f} units")
            fig, ax = plt.subplots(figsize=(5.0, 3.0))
            chart_df = impact_df.head(6).sort_values("SHAP contribution")
            bar_colors = ["#008bfb" if val >= 0 else "#ff0051" for val in chart_df["SHAP contribution"]]
            ax.barh(chart_df["Feature label"], chart_df["SHAP contribution"], color=bar_colors)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP contribution (model units)")
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.info(
                f"Top model driver: **{top_driver['Feature label']}** "
                f"({top_driver['SHAP contribution']:+.1f} SHAP units). "
                "This explains the model output; it does not prove causation."
            )

    # Agent 2
    with row1[1]:
        with st.container(border=True):
            st.markdown("### 📦 Agent 2 — Inventory Safety")
            st.metric("On shelf + backroom + incoming", f"{total_available_stock:,} units")
            st.metric("Projected after forecast", f"{projected_remaining_stock:+,} units")
            if inventory_status == "CRITICAL SHORTAGE":
                st.error("Forecast demand exceeds the entered available stock.")
            elif inventory_status == "LOW BUFFER":
                st.warning("Projected stock is below the entered safety-stock target.")
            else:
                st.success("Entered stock covers forecast demand and safety-stock threshold.")
            st.caption("This is a scenario calculation; it does not place an order.")

    # Agent 3
    with row1[2]:
        with st.container(border=True):
            st.markdown("### 🎯 Agent 3 — Commercial Intelligence")
            st.metric("Lidl price", f"€{s_price:.2f}")
            st.metric("Competitor price", f"€{competitor_price:.2f}")
            st.metric("Lidl minus competitor", f"€{price_gap:+.2f}")
            st.write(competitive_position + ".")
            st.write(
                "Competitor promotion: **"
                + ("Yes" if competitor_promo_active else "No") + "**"
            )
            st.caption("Use comparable products, pack sizes, and reliable observed prices.")

    row2 = st.columns(3, gap="medium")

    # Agent 4
    with row2[0]:
        with st.container(border=True):
            st.markdown("### 🚚 Agent 4 — Replenishment")
            st.metric("Lead-time demand estimate", f"{lead_time_demand:,} units")
            st.metric("Indicative order quantity", f"{recommended_order:,} units")
            st.write(replenishment_message)
            st.warning("Indicative rule only: verify order multiples, delivery schedules, and supplier constraints.")

    # Agent 5
    with row2[1]:
        with st.container(border=True):
            st.markdown("### 🥬 Agent 5 — Freshness & Waste")
            st.metric("Units nearing expiry", f"{short_dated_units:,}")
            st.metric("Estimated waste cost", f"€{estimated_waste_cost:,.2f}")
            st.metric("Illustrative markdown revenue", f"€{estimated_markdown_revenue:,.2f}")
            if waste_risk == "HIGH":
                st.error(f"Waste / expiry risk: {waste_risk}. Review batch-level expiry and approved markdown policy.")
            elif waste_risk == "MODERATE":
                st.warning("Some stock is nearing expiry. Check batch quantities and sell-through.")
            else:
                st.success("No near-expiry units entered.")
            st.caption("Markdown revenue is a simple estimate, not a demand or profit forecast.")

    # Agent 6
    with row2[2]:
        with st.container(border=True):
            st.markdown("### 👥 Agent 6 — Workforce Planning")
            st.metric("Estimated staff required", f"{required_staff}")
            st.metric("Staff scheduled", f"{scheduled_staff}")
            st.metric("Staff gap", f"{staff_gap:+d}")
            if staff_gap > 0:
                st.warning(
                    f"Potential gap of {staff_gap} staff member(s) for the selected peak-hour assumption. "
                    "Check labor rules, contracts, breaks, and workload."
                )
            else:
                st.success("Scheduled coverage meets this simplified estimate.")
            st.caption("Planning heuristic only; not an employee scheduling instruction.")

    row3 = st.columns(3, gap="medium")

    # Agent 7
    with row3[0]:
        with st.container(border=True):
            st.markdown("### 🏷️ Agent 7 — Promotion Analysis")
            st.write(f"Lidl flyer promotion: **{'Yes' if lidl_promo_active else 'No'}**")
            st.write(f"Competitor promotion: **{'Yes' if competitor_promo_active else 'No'}**")
            st.write(promo_advice)
            if promo_flags:
                for flag in promo_flags:
                    st.caption("• " + flag)
            st.caption("Promotion effectiveness requires actual baseline and promotional sales data.")

    # Agent 8
    with row3[1]:
        with st.container(border=True):
            st.markdown("### 💶 Agent 8 — Profitability")
            st.metric("Illustrative forecast revenue", f"€{estimated_revenue:,.2f}")
            st.metric("Gross profit before other costs", f"€{estimated_gross_profit:,.2f}")
            st.metric("After entered promotion cost", f"€{estimated_promo_net:,.2f}")
            st.metric("Estimated waste cost", f"€{estimated_waste_cost:,.2f}")
            if s_price < unit_cost:
                st.error("Entered retail price is below entered unit cost.")
            else:
                st.success("Entered retail price exceeds entered unit cost.")
            st.caption("Excludes VAT treatment, fixed costs, shrinkage beyond inputs, and other operating costs.")

    # Agent 9
    with row3[2]:
        with st.container(border=True):
            st.markdown("### ✅ Agent 9 — Compliance & Tasks")
            st.metric("Overdue tasks", f"{int(overdue_tasks)}")
            st.metric("Open compliance issues", f"{int(open_compliance_issues)}")
            st.write(f"Status: **{compliance_status}**")
            if compliance_tasks:
                for task in compliance_tasks:
                    st.warning(task)
            else:
                st.success("No overdue tasks or incomplete checks were entered.")
            st.caption("This is a manual scenario checklist, not a verified compliance system.")

    # Orchestration: prioritize exceptions and show a single action queue.
    st.divider()
    st.subheader("🧭 Orchestrator — Prioritized Manager Action Queue")
    actions = []

    if inventory_status == "CRITICAL SHORTAGE":
        actions.append(("P1 — Critical", "Inventory", "Validate available stock and arrange authorized replenishment review."))
    elif inventory_status == "LOW BUFFER":
        actions.append(("P2 — High", "Inventory", "Check safety stock, incoming delivery, and lead-time coverage."))

    if waste_risk == "HIGH":
        actions.append(("P1 — Critical", "Freshness", "Inspect near-expiry batches and follow approved markdown / rotation procedures."))
    elif waste_risk == "MODERATE":
        actions.append(("P2 — High", "Freshness", "Review near-expiry stock and first-expired-first-out rotation."))

    if staff_gap > 0:
        actions.append(("P2 — High", "Workforce", f"Review peak-hour coverage; simplified estimate shows a gap of {staff_gap}."))

    if compliance_tasks:
        actions.append(("P1 — Critical" if open_compliance_issues > 0 or temperature_check_due == "No" else "P2 — High",
                        "Compliance", " ".join(compliance_tasks)))

    if s_price < unit_cost:
        actions.append(("P1 — Critical", "Profitability", "Review entered price and unit cost; estimated unit margin is negative."))

    if competitor_promo_active:
        actions.append(("P3 — Review", "Commercial", "Assess competitor promotion using comparable-product data and margin constraints."))

    if not actions:
        actions.append(("P3 — Routine", "General", "No high-priority exceptions were identified from the values entered. Review the dashboard and verify source data."))

    action_df = pd.DataFrame(actions, columns=["Priority", "Area", "Recommended review"])
    st.dataframe(action_df, use_container_width=True, hide_index=True)

    # ----------------------- Telemetry -----------------------
    st.divider()
    st.subheader("📋 Scenario telemetry")
    telemetry = input_row.copy()
    telemetry.insert(0, "Selected category", selected_cat)
    telemetry["Forecast demand units"] = predicted_sales
    telemetry["Historical average units"] = round(historical_average, 2)
    telemetry["On-shelf stock"] = int(on_shelf_stock)
    telemetry["Backroom stock"] = int(backroom_stock)
    telemetry["Incoming delivery"] = int(incoming_delivery)
    telemetry["Total available stock"] = total_available_stock
    telemetry["Projected remaining stock"] = projected_remaining_stock
    telemetry["Inventory status"] = inventory_status
    telemetry["Indicative replenishment units"] = recommended_order
    telemetry["Competitor price gap (€)"] = round(price_gap, 2)
    telemetry["Short-dated units"] = int(short_dated_units)
    telemetry["Estimated required staff"] = required_staff
    telemetry["Scheduled staff"] = int(scheduled_staff)
    telemetry["Estimated revenue (€)"] = round(estimated_revenue, 2)
    telemetry["Estimated gross profit (€)"] = round(estimated_gross_profit, 2)
    telemetry["Compliance status"] = compliance_status
    st.dataframe(telemetry, use_container_width=True, hide_index=True)

    csv_data = action_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download manager action queue (CSV)",
        data=csv_data,
        file_name="lidl_manager_action_queue.csv",
        mime="text/csv",
    )

    with st.expander("Model features and SHAP contributions"):
        st.dataframe(
            impact_df[["Feature label", "SHAP contribution", "Absolute contribution"]],
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Validation reminder: this application uses synthetic data and illustrative rules. "
        "It is not an official Lidl tool, does not connect to Lidl systems, and does not execute actions. "
        "Validate models and operational policies with authorized real data before any business use."



# Demo scenarios. Replace with real monitored events when available.
SCENARIOS = [
    {
        "id": "COMP-001",
        "title": "Competitor increases its price",
        "description": "A competitor's observed price increases to €2.00.",
        "competitor_price": 2.00,
        "competitor_promo": 0,
    },
    {
        "id": "COMP-002",
        "title": "Competitor launches a promotion",
        "description": "A competitor starts a promotional campaign.",
        "competitor_price": 1.49,
        "competitor_promo": 1,
    },
    {
        "id": "COMP-003",
        "title": "Competitor cuts its price",
        "description": "A competitor's observed price falls to €1.20.",
        "competitor_price": 1.20,
        "competitor_promo": 0,
    },
]

if "scenario_index" not in st.session_state:
    st.session_state.scenario_index = 0

if "pending_scenario" not in st.session_state:
    st.session_state.pending_scenario = None

if "decision_log" not in st.session_state:
    st.session_state.decision_log = []

if "last_scenario_id" not in st.session_state:
    st.session_state.last_scenario_id = None


def calculate_option(price, scenario):
    """Predict demand and basic economics for one price option."""
    row = input_row.copy()

    row.loc[0, "Price_Euro"] = price
    row.loc[0, "Competitor_Price_Euro"] = (
        scenario["competitor_price"]
    )
    row.loc[0, "Competitor_Promo"] = (
        scenario["competitor_promo"]
    )

    predicted_units = max(
        0, int(round(float(model.predict(row)[0])))
    )

    revenue = predicted_units * price
    gross_profit = predicted_units * (price - unit_cost)

    return {
        "price": price,
        "units": predicted_units,
        "revenue": revenue,
        "gross_profit": gross_profit,
    }


def make_scenario_comparison(scenario):
    """Compare candidate actions against the current price."""
    candidate_prices = {
        "Raise price by 5%": round(s_price * 1.05, 2),
        "Keep current price": round(s_price, 2),
        "Lower price by 5%": round(s_price * 0.95, 2),
    }

    rows = []
    for action, price in candidate_prices.items():
        result = calculate_option(price, scenario)
        result["Action"] = action
        rows.append(result)

    return pd.DataFrame(rows)


@st.dialog("Retail decision required", width="large")
def show_scenario_dialog(scenario):
    st.warning("SIMULATED EVENT — not a verified live competitor update")

    st.subheader(scenario["title"])
    st.write(scenario["description"])
    st.write(f"Category: **{selected_cat}**")
    st.write(
        f"Scenario competitor price: "
        f"**€{scenario['competitor_price']:.2f}**"
    )

    comparison = make_scenario_comparison(scenario)

    st.markdown("### Predicted impact of each option")
    st.dataframe(
        comparison[
            ["Action", "price", "units", "revenue", "gross_profit"]
        ].rename(columns={
            "price": "Lidl price (€)",
            "units": "Forecast units",
            "revenue": "Revenue (€)",
            "gross_profit": "Gross profit (€)",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Estimates use the current predictive model. "
        "They are not causal estimates or guaranteed outcomes."
    )

    best_index = int(
        comparison["gross_profit"].to_numpy().argmax()
    )
    suggested_action = comparison.iloc[best_index]["Action"]

    st.info(
        f"Candidate with the highest modeled gross profit: "
        f"**{suggested_action}**. "
        "Check pricing rules and uncertainty before approving."
    )

    st.markdown("### What should the manager do?")

    for _, option in comparison.iterrows():
        if st.button(
            f"Select: {option['Action']}",
            key=f"decision_{scenario['id']}_{option['Action']}",
            use_container_width=True,
        ):
            st.session_state.decision_log.append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "scenario": scenario["title"],
                "scenario_id": scenario["id"],
                "action": option["Action"],
                "proposed_price": float(option["price"]),
                "predicted_units": int(option["units"]),
                "predicted_revenue": float(option["revenue"]),
                "predicted_gross_profit": float(
                    option["gross_profit"]
                ),
                "status": "Selected for review; not executed",
            })
            st.session_state.pending_scenario = None
            st.rerun()

    if st.button(
        "Defer decision",
        key=f"defer_{scenario['id']}",
        use_container_width=True,
    ):
        st.session_state.decision_log.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "scenario": scenario["title"],
            "scenario_id": scenario["id"],
            "action": "Deferred",
            "proposed_price": None,
            "predicted_units": None,
            "predicted_revenue": None,
            "predicted_gross_profit": None,
            "status": "Awaiting manager review",
        })
        st.session_state.pending_scenario = None
        st.rerun()


@st.fragment(run_every="30s")
def scenario_monitor():
    st.subheader("Autonomous Scenario Monitor")

    if st.session_state.pending_scenario is None:
        scenario = SCENARIOS[
            st.session_state.scenario_index % len(SCENARIOS)
        ]

        st.session_state.scenario_index += 1
        st.session_state.pending_scenario = scenario
        st.session_state.last_scenario_id = scenario["id"]

    pending = st.session_state.pending_scenario

    st.warning(
        f"New scenario awaiting review: {pending['title']}"
    )

    if st.button(
        "Review scenario and take action",
        key="review_scenario",
        type="primary",
    ):
        show_scenario_dialog(pending)

    # Also open the dialog automatically when a scenario is pending.
    # The dialog remains available after the timed fragment reruns.
    show_scenario_dialog(pending)


scenario_monitor()

st.subheader("Manager decision history")
if st.session_state.decision_log:
    st.dataframe(
        pd.DataFrame(st.session_state.decision_log),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No manager decisions have been recorded yet.")

    )
