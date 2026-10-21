from __future__ import annotations

import json
import re
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

st.set_page_config(page_title="Retail Multi-Agent Store Operations MVP (Retail Brain)", page_icon="🛒", layout="wide")

# Keep explanatory captions and secondary labels readable in both Streamlit themes.
# Use Streamlit's theme text variable so auto/light/dark themes are respected.
st.markdown("""
<style>
/* Retail Brain analytics workspace: inspired by the supplied clean SaaS dashboard */
:root { --rb-purple: #6254c7; --rb-purple-dark: #493ca9; --rb-orange: #ff7a18; --rb-ink: #22243a; --rb-muted: #74778b; --rb-card: var(--secondary-background-color); }
.stApp { background: linear-gradient(135deg, rgba(98,84,199,.045), transparent 42%), var(--background-color); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1600px; }
.stApp h1 { font-size: clamp(1.55rem, 2.3vw, 2.15rem); font-weight: 750; letter-spacing: -.035em; color: var(--text-color); }
.stApp h2, .stApp h3 { letter-spacing: -.02em; font-weight: 700; }
.stApp [data-testid="stMetric"] { background: var(--secondary-background-color); border: 1px solid rgba(120,120,150,.16); border-radius: 15px; padding: 17px 18px; box-shadow: 0 5px 18px rgba(40,35,90,.045); min-height: 108px; }
.stApp [data-testid="stMetricLabel"] { font-size: .82rem; font-weight: 600; }
.stApp [data-testid="stMetricValue"] { font-size: clamp(1.25rem, 1.8vw, 1.8rem); font-weight: 750; color: var(--rb-purple); }
.stApp [data-testid="stMetricDelta"] { font-size: .78rem; }
.stApp [data-testid="stTabs"] [role="tablist"] { gap: 5px; border-bottom: 1px solid rgba(120,120,150,.18); }
.stApp [data-testid="stTabs"] button[role="tab"] { border-radius: 10px 10px 0 0; padding: .7rem .85rem; font-weight: 650; }
.stApp [data-testid="stTabs"] button[aria-selected="true"] { color: var(--rb-purple) !important; border-bottom: 3px solid var(--rb-purple) !important; }
.stApp [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid rgba(120,120,150,.16); }
.stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.5; }
.stApp [data-testid="stSidebar"] button[kind="primary"], .stApp button[kind="primary"] { background: var(--rb-purple); border-color: var(--rb-purple); color: #fff; border-radius: 10px; font-weight: 650; }
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
