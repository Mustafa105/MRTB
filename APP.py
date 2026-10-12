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
    "All data and actions in this MVP are simulated."
)

# ---------------------------------------------------------------------
# STEP 1: Synthetic data generation
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
            }
        )

        if "Birra" in cat:
            base = (
                400
                + df_cat["Store_Traffic_Index"] * 4
                + df_cat["Temperature_C"] * 12
                + df_cat["Is_Promo_Volantino"] * 500
            )
        elif "Forno" in cat:
            base = (
                600
                + df_cat["Store_Traffic_Index"] * 5
                - df_cat["Temperature_C"] * 2
                + df_cat["In_Store_Display"] * 300
            )
        elif "Ortofrutta" in cat:
            base = (
                500
                + df_cat["Store_Traffic_Index"] * 3
                + df_cat["Temperature_C"] * 5
                - df_cat["Price_Euro"] * 80
            )
        else:
            base = (
                200
                + df_cat["Store_Traffic_Index"] * 2
                + df_cat["Is_Promo_Volantino"] * 400
            )

        noise = rng.normal(0, 30, days)
        df_cat["Sales_Volume"] = np.maximum((base + noise).astype(int), 10)
        all_data.append(df_cat)

    return pd.concat(all_data, ignore_index=True)


df_master = generate_lidl_data()

# ---------------------------------------------------------------------
# STEP 2: Scenario controls
# ---------------------------------------------------------------------
st.sidebar.header("🕹️ Store Scenario Controls")
selected_cat = st.sidebar.selectbox(
    "Select Target Department/Category:",
    df_master["Category"].unique().tolist(),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Environmental Inputs")
s_traffic = st.sidebar.slider("Store Foot Traffic Index", 50, 200, 140)
s_weekend = st.sidebar.selectbox(
    "Calendar Day:", ["Weekday", "Weekend (Saturday/Sunday)"]
)
s_temp = st.sidebar.slider("Forecasted External Temperature (°C)", 0, 42, 32)

st.sidebar.subheader("Current Merchandising Settings")
s_price = st.sidebar.slider("Current Retail Price (€)", 0.50, 5.00, 1.49, step=0.01)
s_promo = st.sidebar.selectbox("Include in Lidl Flyer (Volantino)?", ["No", "Yes"])
s_display = st.sidebar.selectbox("Special Promotional Endcap Display?", ["No", "Yes"])

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Real-Time Inventory Status")
current_backroom_stock = st.sidebar.number_input(
    "Available Stock Units",
    min_value=0,
    value=350,
    step=50,
)

# ---------------------------------------------------------------------
# Train category-specific model and generate prediction
# ---------------------------------------------------------------------
df_filtered = df_master.loc[df_master["Category"] == selected_cat].copy()
feature_columns = [
    "Price_Euro",
    "Is_Promo_Volantino",
    "In_Store_Display",
    "Temperature_C",
    "Store_Traffic_Index",
    "Is_Weekend",
]
X = df_filtered[feature_columns]
y = df_filtered["Sales_Volume"]

model = xgb.XGBRegressor(
    n_estimators=60,
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
        }
    ],
    columns=feature_columns,
)

predicted_sales = max(0, int(round(float(model.predict(input_row)[0]))))

# SHAP values: one row and one value per feature for this regression model.
explainer = shap.TreeExplainer(model)
shap_result = explainer(input_row)
shap_vector = np.asarray(shap_result.values).reshape(-1)

impact_df = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Impact": shap_vector,
        "Abs_Impact": np.abs(shap_vector),
    }
).sort_values("Abs_Impact", ascending=False)

clean_labels = {
    "Price_Euro": "Pricing Strategy",
    "Is_Promo_Volantino": "Flyer Advertisement",
    "In_Store_Display": "Endcap Placement",
    "Temperature_C": "Weather Variables",
    "Store_Traffic_Index": "Branch Foot Traffic",
    "Is_Weekend": "Weekend Shopper Surge",
}

# ---------------------------------------------------------------------
# STEP 3: Multi-agent orchestration pipeline
# ---------------------------------------------------------------------
st.subheader("🤖 Autonomous Agent Orchestration Pipeline")
st.write(
    "Trigger a simulated multi-agent evaluation cycle using the selected scenario."
)

if "simulation_triggered" not in st.session_state:
    st.session_state.simulation_triggered = False

if st.button("🚀 Trigger Live Agent Cycle", type="primary"):
    st.session_state.simulation_triggered = True

if st.session_state.simulation_triggered:
    # Agent 1: Demand forecast and explainability
    st.markdown("### 🛰️ Agent 1: Demand & Explainable AI (XAI) Agent")
    col1, col2 = st.columns(2)

    with col1:
        historical_average = float(y.mean())
        st.metric(
            label=f"Forecasted Demand Plan for: {selected_cat}",
            value=f"{predicted_sales:,} Units Today",
            delta=f"{predicted_sales - historical_average:,.0f} vs. category average",
        )

        top_feature = impact_df.iloc[0]["Feature"]
        second_feature = impact_df.iloc[1]["Feature"]
        top_impact = float(impact_df.iloc[0]["Impact"])
        second_impact = float(impact_df.iloc[1]["Impact"])

        top_direction = "positive" if top_impact >= 0 else "negative"
        second_direction = "positive" if second_impact >= 0 else "negative"

        st.info(
            f"**Agent analysis:** The model forecasts **{predicted_sales:,} units** "
            f"for the selected scenario.\n\n"
            f"- **Largest SHAP contribution:** {clean_labels[top_feature]} "
            f"has a {top_direction} contribution relative to the model's baseline.\n"
            f"- **Second-largest SHAP contribution:** {clean_labels[second_feature]} "
            f"has a {second_direction} contribution relative to the model's baseline.\n\n"
            "SHAP contributions explain this prediction; they do not establish "
            "causation or prove that changing a feature will produce the same effect."
        )

    with col2:
        plot_df = impact_df.sort_values("Impact")
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = [
            "#008bfb" if value >= 0 else "#ff0051"
            for value in plot_df["Impact"]
        ]
        ax.barh(
            [clean_labels[feature] for feature in plot_df["Feature"]],
            plot_df["Impact"],
            color=bar_colors,
        )
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP contribution (model output units)")
        ax.set_title("Feature Contributions to This Forecast")
        ax.tick_params(axis="both", which="major", labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Agent 2: Inventory and logistics
    st.markdown("### 📦 Agent 2: Inventory & Logistical Safety Orchestrator")

    stock_delta = int(current_backroom_stock) - predicted_sales
    col_inv1, col_inv2 = st.columns(2)

    with col_inv1:
        st.metric(
            label="Projected Stock After Forecast Demand",
            value=f"{stock_delta:,} Units",
            delta="Potential shortage" if stock_delta < 0 else "Stock remains",
            delta_color="inverse" if stock_delta < 0 else "normal",
        )

    with col_inv2:
        if stock_delta < 0:
            suggested_restock = predicted_sales - int(current_backroom_stock)
            st.error(
                f"**CRITICAL INVENTORY ALERT (SIMULATION):** Forecast demand "
                f"({predicted_sales:,} units) exceeds available stock "
                f"({int(current_backroom_stock):,} units) by "
                f"{suggested_restock:,} units.\n\n"
                f"**Suggested action:** Review an expedited replenishment of at least "
                f"**{suggested_restock:,} units**, plus any operational safety stock. "
                "No real order has been placed."
            )
        elif stock_delta < max(50, int(predicted_sales * 0.10)):
            st.warning(
                f"**LOW STOCK BUFFER (SIMULATION):** The projected remaining stock is "
                f"{stock_delta:,} units. Consider checking lead time and safety-stock "
                "requirements before approving replenishment. No real order has been placed."
            )
        else:
            st.success(
                f"**Inventory status (simulation):** Available stock is sufficient "
                f"for the forecast, with approximately {stock_delta:,} units remaining."
            )

    st.markdown("---")

    # Agent 3: Commercial and merchandising recommendations
    st.markdown("### 🎯 Agent 3: Commercial Pricing & Cross-Merchandising Agent")

    if "Birra" in selected_cat:
        rec_bundle = (
            "🍿 **Cross-merchandising suggestion:** Consider a seasonal display "
            "pairing beer with complementary snacks, such as pretzels, popcorn, "
            "and tortilla chips, subject to local merchandising rules."
        )
        rec_pricing = (
            "💰 **Pricing suggestion:** Compare the expected volume and margin under "
            "current pricing against approved promotional scenarios. Do not increase "
            "prices automatically; validate margin, demand, and pricing policies first."
        )
    elif "Forno" in selected_cat:
        rec_bundle = (
            "🧀 **Cross-merchandising suggestion:** Consider placing suitable spreads "
            "or cheese near the bakery area, subject to food-safety and store-layout rules."
        )
        rec_pricing = (
            "💰 **Pricing suggestion:** For products with a short shelf life, evaluate "
            "an approved late-day markdown policy using expiry, waste, and margin data."
        )
    elif "Ortofrutta" in selected_cat:
        rec_bundle = (
            "🥗 **Cross-merchandising suggestion:** Consider displaying salad dressings "
            "or olive oil near relevant fresh-produce products, subject to layout constraints."
        )
        rec_pricing = (
            "💰 **Pricing suggestion:** Evaluate approved targeted promotions against "
            "baseline sales, product freshness, waste, and gross margin."
        )
    else:
        rec_bundle = (
            "🧻 **Cross-merchandising suggestion:** Consider a household essentials "
            "display pairing complementary cleaning and paper products, where appropriate."
        )
        rec_pricing = (
            "💰 **Pricing suggestion:** Review current margin, stock turnover, and "
            "shelf availability before changing prices or allocating additional shelf space."
        )

    col_comm1, col_comm2 = st.columns(2)
    with col_comm1:
        st.info(rec_bundle)
    with col_comm2:
        st.warning(rec_pricing)

    st.markdown("---")
    st.subheader("📋 System Telemetry Log (Scenario Matrix)")

    telemetry_data = input_row.copy()
    telemetry_data["Category_Selected"] = selected_cat
    telemetry_data["Projected_Volume"] = predicted_sales
    telemetry_data["Available_Stock"] = int(current_backroom_stock)
    telemetry_data["Projected_Stock_After_Demand"] = stock_delta

    if stock_delta < 0:
        stock_status = "CRITICAL_SHORTAGE"
    elif stock_delta < max(50, int(predicted_sales * 0.10)):
        stock_status = "LOW_STOCK_BUFFER"
    else:
        stock_status = "STABLE"

    telemetry_data["Safety_Stock_Status"] = stock_status
    st.dataframe(telemetry_data, use_container_width=True)
