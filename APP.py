import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Lidl Multi-Agent Branch Orchestrator")
st.title("💛 💙 Lidl Store Operations Multi-Agent Platform (MVP)")
st.caption(
    "Proof of concept simulating automated agent cycles across multiple categories. "
    "Sales data and agent actions are simulated, not actual Lidl operational data."
)

# ---------------------------------------------------------------------
# STEP 1: Synthetic data generation, including competitor variables
# ---------------------------------------------------------------------
@st.cache_data
def generate_lidl_data():
    rng = np.random.default_rng(42)
    days = 150
    categories = [
        "Birra (Beverages)",
        "Prodotti da Forno (Bakery)",
        "Ortofrutta (Fresh Produce)",
        "Detersivi (Cleaning Products)",
    ]
    all_data = []

    for cat in categories:
        df_cat = pd.DataFrame(
            {
                "Category": cat,
                "Price_Euro": rng.uniform(0.80, 4.50, days),
                "Is_Promo_Volantino": rng.choice([0, 1], size=days, p=[0.75, 0.25]),
                "In_Store_Display": rng.choice([0, 1], size=days, p=[0.85, 0.15]),
                "Temperature_C": rng.uniform(2, 36, days),
                "Store_Traffic_Index": rng.uniform(50, 150, days),
                "Is_Weekend": rng.choice([0, 1], size=days, p=[5 / 7, 2 / 7]),
                "Competitor_Price_Euro": rng.uniform(0.80, 4.50, days),
                "Competitor_Promo": rng.choice([0, 1], size=days, p=[0.75, 0.25]),
            }
        )

        # Synthetic sales relationships. These coefficients are illustrative,
        # not estimated from real Lidl or competitor sales data.
        price_gap = df_cat["Competitor_Price_Euro"] - df_cat["Price_Euro"]
        competitor_effect = (price_gap * 65) - (df_cat["Competitor_Promo"] * 90)

        if "Birra" in cat:
            base = (
                400
                + df_cat["Store_Traffic_Index"] * 4
                + df_cat["Temperature_C"] * 12
                + df_cat["Is_Promo_Volantino"] * 500
                + df_cat["In_Store_Display"] * 80
                + competitor_effect
            )
        elif "Forno" in cat:
            base = (
                600
                + df_cat["Store_Traffic_Index"] * 5
                - df_cat["Temperature_C"] * 2
                + df_cat["In_Store_Display"] * 300
                + df_cat["Is_Promo_Volantino"] * 80
                + competitor_effect
            )
        elif "Ortofrutta" in cat:
            base = (
                500
                + df_cat["Store_Traffic_Index"] * 3
                + df_cat["Temperature_C"] * 5
                - df_cat["Price_Euro"] * 80
                + df_cat["In_Store_Display"] * 60
                + competitor_effect
            )
        else:
            base = (
                200
                + df_cat["Store_Traffic_Index"] * 2
                + df_cat["Is_Promo_Volantino"] * 400
                + df_cat["In_Store_Display"] * 60
                + competitor_effect
            )

        df_cat["Sales_Volume"] = np.maximum(
            (base + rng.normal(0, 30, days)).astype(int), 10
        )
        all_data.append(df_cat)

    return pd.concat(all_data, ignore_index=True)


df_master = generate_lidl_data()

# ---------------------------------------------------------------------
# STEP 2: Scenario controls
# ---------------------------------------------------------------------
st.sidebar.header("🕹️ Store Scenario Controls")
selected_cat = st.sidebar.selectbox(
    "Select Target Department/Category:", df_master["Category"].unique().tolist()
)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Environmental Inputs")
s_traffic = st.sidebar.slider("Store Foot Traffic Index", 50, 200, 140)
s_weekend = st.sidebar.selectbox(
    "Calendar Day:", ["Weekday", "Weekend (Saturday/Sunday)"]
)
s_temp = st.sidebar.slider("Forecasted External Temperature (°C)", 0, 42, 32)

st.sidebar.subheader("Current Merchandising Settings")
s_price = st.sidebar.slider(
    "Lidl Retail Price (€)", 0.50, 5.00, 1.49, step=0.01
)
s_promo = st.sidebar.selectbox("Include in Lidl Flyer (Volantino)?", ["No", "Yes"])
s_display = st.sidebar.selectbox("Special Promotional Endcap Display?", ["No", "Yes"])

st.sidebar.markdown("---")
st.sidebar.subheader("🏪 Competitor Intelligence")
competitor_price = st.sidebar.slider(
    "Competitor Retail Price (€)", 0.50, 5.00, 1.59, step=0.01
)
competitor_promo = st.sidebar.selectbox(
    "Competitor Running a Promotion?", ["No", "Yes"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Inventory Status")
current_backroom_stock = st.sidebar.number_input(
    "Available Stock Units", min_value=0, value=1050, step=50
)

# ---------------------------------------------------------------------
# STEP 3: Train category-specific model and predict demand
# ---------------------------------------------------------------------
df_filtered = df_master.loc[df_master["Category"] == selected_cat].copy()
feature_columns = [
    "Price_Euro",
    "Is_Promo_Volantino",
    "In_Store_Display",
    "Temperature_C",
    "Store_Traffic_Index",
    "Is_Weekend",
    "Competitor_Price_Euro",
    "Competitor_Promo",
]
X = df_filtered[feature_columns]
y = df_filtered["Sales_Volume"]

model = xgb.XGBRegressor(
    n_estimators=80,
    max_depth=4,
    learning_rate=0.08,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1,
)
model.fit(X, y)

input_row = pd.DataFrame(
    [
        {
            "Price_Euro": s_price,
            "Is_Promo_Volantino": int(s_promo == "Yes"),
            "In_Store_Display": int(s_display == "Yes"),
            "Temperature_C": s_temp,
            "Store_Traffic_Index": s_traffic,
            "Is_Weekend": int(s_weekend == "Weekend (Saturday/Sunday)"),
            "Competitor_Price_Euro": competitor_price,
            "Competitor_Promo": int(competitor_promo == "Yes"),
        }
    ],
    columns=feature_columns,
)

predicted_sales = max(0, int(round(float(model.predict(input_row)[0]))))
historical_average = float(y.mean())
stock_delta = int(current_backroom_stock) - predicted_sales
price_gap = float(s_price - competitor_price)

# SHAP values for this one scenario
explainer = shap.TreeExplainer(model)
shap_result = explainer(input_row)
shap_vector = np.asarray(shap_result.values).reshape(-1)

clean_labels = {
    "Price_Euro": "Lidl Price",
    "Is_Promo_Volantino": "Lidl Flyer Promotion",
    "In_Store_Display": "Endcap Display",
    "Temperature_C": "Temperature",
    "Store_Traffic_Index": "Store Foot Traffic",
    "Is_Weekend": "Weekend",
    "Competitor_Price_Euro": "Competitor Price",
    "Competitor_Promo": "Competitor Promotion",
}

impact_df = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Feature Label": [clean_labels[f] for f in feature_columns],
        "Impact": shap_vector,
        "Abs_Impact": np.abs(shap_vector),
    }
).sort_values("Abs_Impact", ascending=False)

# ---------------------------------------------------------------------
# STEP 4: Side-by-side multi-agent dashboard
# ---------------------------------------------------------------------
st.subheader("🤖 Multi-Agent Orchestration Dashboard")
st.write(
    "Adjust the controls in the sidebar, then run the simulated agent cycle. "
    "All three agents are displayed side by side on desktop screens."
)

if "simulation_triggered" not in st.session_state:
    st.session_state.simulation_triggered = False

if st.button("🚀 Trigger Agent Cycle", type="primary"):
    st.session_state.simulation_triggered = True

if st.session_state.simulation_triggered:
    # Category-specific commercial recommendations
    if "Birra" in selected_cat:
        rec_bundle = (
            "🍿 **Cross-merchandising:** Consider a seasonal display pairing beer "
            "with complementary snacks, subject to store merchandising rules."
        )
        rec_pricing = (
            "Compare approved price and promotion scenarios using expected volume, "
            "gross margin, and competitor activity. Do not change prices automatically."
        )
    elif "Forno" in selected_cat:
        rec_bundle = (
            "🧀 **Cross-merchandising:** Consider suitable spreads or cheese near "
            "the bakery area, subject to food-safety and layout rules."
        )
        rec_pricing = (
            "For short-shelf-life products, evaluate approved late-day markdowns "
            "against expiry, waste, demand, and margin data."
        )
    elif "Ortofrutta" in selected_cat:
        rec_bundle = (
            "🥗 **Cross-merchandising:** Consider displaying salad dressings or "
            "olive oil near relevant fresh-produce products."
        )
        rec_pricing = (
            "Evaluate targeted promotions against baseline sales, freshness, waste, "
            "competitor prices, and gross margin."
        )
    else:
        rec_bundle = (
            "🧻 **Cross-merchandising:** Consider a household essentials display "
            "pairing complementary cleaning and paper products where appropriate."
        )
        rec_pricing = (
            "Review margin, stock turnover, competitor activity, and shelf availability "
            "before changing prices or shelf allocation."
        )

    # All three agents occupy one horizontal row.
    agent1, agent2, agent3 = st.columns([1.35, 1, 1], gap="medium")

    # -----------------------------------------------------------------
    # AGENT 1: Demand and XAI
    # -----------------------------------------------------------------
    with agent1:
        st.markdown("### 🛰️ Agent 1: Demand & XAI")
        st.metric(
            label=f"Forecast: {selected_cat}",
            value=f"{predicted_sales:,} units",
            delta=f"{predicted_sales - historical_average:+,.0f} vs. category average",
        )
        st.caption("Largest SHAP drivers for this forecast")
        top_rows = impact_df.head(4).sort_values("Impact")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        colors = ["#008bfb" if v >= 0 else "#ff0051" for v in top_rows["Impact"]]
        ax.barh(top_rows["Feature Label"], top_rows["Impact"], color=colors)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP contribution (units)")
        ax.tick_params(axis="both", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        top_feature = impact_df.iloc[0]
        top_direction = "positive" if top_feature["Impact"] >= 0 else "negative"
        st.info(
            f"**Main driver:** {top_feature['Feature Label']} has a {top_direction} "
            "SHAP contribution relative to the model baseline. SHAP describes this "
            "model prediction; it does not establish causation."
        )

    # -----------------------------------------------------------------
    # AGENT 2: Inventory and logistics
    # -----------------------------------------------------------------
    with agent2:
        st.markdown("### 📦 Agent 2: Inventory")
        st.metric("Available Stock", f"{int(current_backroom_stock):,} units")
        st.metric(
            "Projected Stock After Demand",
            f"{stock_delta:+,} units",
            delta="Potential shortage" if stock_delta < 0 else "Stock remains",
            delta_color="inverse" if stock_delta < 0 else "normal",
        )

        if stock_delta < 0:
            suggested_restock = abs(stock_delta)
            st.error(
                f"**Critical shortage (simulation):** forecast demand exceeds stock "
                f"by **{suggested_restock:,} units**. Review replenishment needs; "
                "no real order has been placed."
            )
            stock_status = "CRITICAL_SHORTAGE"
        elif stock_delta < max(50, int(predicted_sales * 0.10)):
            st.warning(
                "**Low stock buffer (simulation):** check lead times and safety-stock "
                "requirements before approving replenishment."
            )
            stock_status = "LOW_STOCK_BUFFER"
        else:
            st.success("**Stock status:** projected inventory covers forecast demand.")
            stock_status = "STABLE"

    # -----------------------------------------------------------------
    # AGENT 3: Commercial and competitor intelligence
    # -----------------------------------------------------------------
    with agent3:
        st.markdown("### 🎯 Agent 3: Commercial")
        st.metric("Lidl Price", f"€{s_price:.2f}")
        st.metric("Competitor Price", f"€{competitor_price:.2f}")
        st.metric(
            "Lidl Price Gap",
            f"€{price_gap:+.2f}",
            help="Positive means Lidl's selected price is higher than the competitor's.",
        )
        if competitor_promo == "Yes":
            st.warning("Competitor promotion: **Yes**")
        else:
            st.success("Competitor promotion: **No**")
        st.markdown("**Merchandising suggestion**")
        st.write(rec_bundle)
        st.markdown("**Pricing suggestion**")
        st.write(rec_pricing)

    # -----------------------------------------------------------------
    # Explanation and telemetry below the three agents
    # -----------------------------------------------------------------
    st.divider()
    with st.expander("🔎 Full SHAP contribution table", expanded=False):
        st.dataframe(
            impact_df[["Feature Label", "Impact", "Abs_Impact"]].rename(
                columns={"Impact": "SHAP Contribution", "Abs_Impact": "Absolute Contribution"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📋 System Telemetry Log")
    telemetry_data = input_row.copy()
    telemetry_data.insert(0, "Category_Selected", selected_cat)
    telemetry_data["Projected_Volume"] = predicted_sales
    telemetry_data["Historical_Category_Average"] = round(historical_average, 2)
    telemetry_data["Available_Stock"] = int(current_backroom_stock)
    telemetry_data["Projected_Stock_After_Demand"] = stock_delta
    telemetry_data["Lidl_vs_Competitor_Price_Gap"] = round(price_gap, 2)
    telemetry_data["Safety_Stock_Status"] = stock_status
    st.dataframe(telemetry_data, use_container_width=True, hide_index=True)

    st.caption(
        "Model limitation: the training data and competitor effects are synthetic. "
        "Use actual comparable-product prices, promotions, and sales history before "
        "using forecasts for operational decisions."
    )
else:
    st.info("Set your scenario in the sidebar, then click **Trigger Agent Cycle**.")
