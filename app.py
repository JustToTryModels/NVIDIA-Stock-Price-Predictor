import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ==============================
# 🔹 Page Configuration
# ==============================
st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🔹 Theme State Init (MUST be before any CSS)
# ==============================
if 'theme' not in st.session_state:
    st.session_state.theme = 'System 🖳'

# ==============================
# 🔹 Detect System Theme & Resolve Effective Theme
# ==============================
def get_effective_theme(theme_choice):
    if theme_choice == 'System 🖳':
        system_theme_js = """
        <script>
        (function() {
            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            const theme = isDark ? 'dark' : 'light';
            if (!sessionStorage.getItem('systemThemeSet')) {
                sessionStorage.setItem('systemTheme', theme);
                sessionStorage.setItem('systemThemeSet', 'true');
            }
            const url = new URL(window.location);
            const current = url.searchParams.get('sys_theme');
            if (current !== theme) {
                url.searchParams.set('sys_theme', theme);
                window.history.replaceState({}, '', url);
            }
        })();
        </script>
        """
        st.markdown(system_theme_js, unsafe_allow_html=True)
        query_params = st.query_params
        sys_theme = query_params.get('sys_theme', 'dark')
        return 'Dark' if sys_theme == 'dark' else 'Light'
    elif theme_choice == 'Light ☼':
        return 'Light'
    else:
        return 'Dark'

effective_theme = get_effective_theme(st.session_state.theme)
IS_DARK = effective_theme == 'Dark'

# ==============================
# 🔹 Global CSS Styling (Theme-Aware)
# ==============================
def apply_theme_css(theme):
    if theme == 'Light':
        st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #f0f7e6 0%, #ffffff 50%, #f5faf0 100%);
        color: #1a2e05;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f2f9ea 100%);
        border-right: 1px solid rgba(118, 185, 0, 0.35);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 { color: #000000; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(242,249,234,0.95));
        border: 1px solid rgba(118,185,0,0.35);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(118,185,0,0.08), inset 0 1px 0 rgba(255,255,255,0.9);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(118,185,0,0.18), inset 0 1px 0 rgba(255,255,255,0.9);
    }
    [data-testid="stMetricLabel"] {
        color: #5a7a3a !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #1a2e05 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; font-weight: 500 !important; }

    [data-testid="stSlider"] > div > div > div > div { background: #76b900 !important; }
    .stSlider [data-baseweb="slider"] { padding-top: 10px; }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(135deg, #76b900 0%, #5a8f00 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(118,185,0,0.35);
        width: 100%;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(118,185,0,0.55);
        background: linear-gradient(135deg, #8fd400 0%, #76b900 100%);
    }
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(118,185,0,0.3);
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        background: rgba(242,249,234,0.9);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(118,185,0,0.2);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        flex-grow: 1;
        justify-content: center;
        background: transparent;
        border-radius: 8px;
        color: #5a7a3a !important;
        font-weight: 500;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, #76b900, #5a8f00) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stSpinner"] { color: #76b900; }

    [data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        border-radius: 10px !important;
        color: #1a2e05 !important;
    }
    [data-testid="stSelectbox"] > div > div > div {
        color: #1a2e05 !important;
    }
    [data-baseweb="popover"] [data-baseweb="menu"] {
        background: #ffffff !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(118,185,0,0.12) !important;
    }
    [data-baseweb="popover"] [role="option"] {
        background: #ffffff !important;
        color: #1a2e05 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background: rgba(118,185,0,0.12) !important;
        color: #1a2e05 !important;
    }

    hr { border: none; border-top: 1px solid rgba(118,185,0,0.2); margin: 24px 0; }

    .glass-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(242,249,234,0.92));
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 20px;
        padding: 28px 32px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(118,185,0,0.07), inset 0 1px 0 rgba(255,255,255,0.95);
        margin-bottom: 24px;
    }
    .glass-card h2, .glass-card h3 { color: #1a2e05; margin-bottom: 8px; }

    .hero-banner {
        background: linear-gradient(135deg, rgba(118,185,0,0.10) 0%, rgba(255,255,255,0.97) 60%, rgba(242,249,234,0.97) 100%);
        border: 1px solid rgba(118,185,0,0.35);
        border-radius: 24px;
        padding: 40px 48px;
        margin-bottom: 32px;
        box-shadow: 0 16px 48px rgba(118,185,0,0.10), inset 0 1px 0 rgba(255,255,255,0.95);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(118,185,0,0.10) 0%, transparent 70%);
        pointer-events: none;
    }

    .section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .section-header h3 { color: #1a2e05; font-size: 1.25rem; font-weight: 700; margin: 0; }

    .section-badge {
        background: rgba(118,185,0,0.12);
        border: 1px solid rgba(118,185,0,0.45);
        color: #4a7c00;
        font-size: 0.7rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.08em;
    }

    .info-box {
        background: rgba(118,185,0,0.07);
        border-left: 3px solid #76b900;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px; margin: 12px 0;
        color: #2d4a0e; font-size: 0.9rem; line-height: 1.6;
    }
    .warning-box {
        background: rgba(251,191,36,0.08);
        border-left: 3px solid #d97706;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px; margin: 12px 0;
        color: #78350f; font-size: 0.9rem; line-height: 1.6;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(242,249,234,0.7); }
    ::-webkit-scrollbar-thumb { background: rgba(118,185,0,0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(118,185,0,0.7); }

    .sidebar-stat {
        background: rgba(118,185,0,0.07);
        border: 1px solid rgba(118,185,0,0.22);
        border-radius: 10px; padding: 12px 16px; margin: 8px 0;
        display: flex; justify-content: space-between; align-items: center;
    }
    .sidebar-stat-label {
        color: #6b8f3a; font-size: 0.78rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .sidebar-stat-value { color: #1a2e05; font-size: 0.95rem; font-weight: 700; }

    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        background: #16a34a; border-radius: 50%;
        margin-right: 6px; animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(22,163,74,0.4); }
        50%       { opacity: 0.8; box-shadow: 0 0 0 4px rgba(22,163,74,0); }
    }

    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
        border: none; border-radius: 12px;
        padding: 14px 32px;
        font-size: 1rem !important; font-weight: 700 !important;
        letter-spacing: 0.03em; cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(229,46,113,0.4);
        width: 100%; text-transform: uppercase;
    }
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(255,138,0,0.5);
        background: linear-gradient(90deg, #ff9e23, #fa4a88);
    }
    [data-testid="stDownloadButton"] > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(229,46,113,0.4);
    }
</style>
""", unsafe_allow_html=True)

    else:  # Dark
        st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid rgba(118,185,0,0.2);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 { color: #cbd5e1; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22,27,39,0.9), rgba(15,20,30,0.9));
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 16px; padding: 20px 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(118,185,0,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important; font-size: 0.8rem !important;
        font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important; font-size: 1.6rem !important; font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; font-weight: 500 !important; }

    [data-testid="stSlider"] > div > div > div > div { background: #76b900 !important; }
    .stSlider [data-baseweb="slider"] { padding-top: 10px; }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(118,185,0,0.2);
        border-radius: 12px; overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(135deg, #76b900 0%, #5a8f00 100%);
        color: #0a0a0f !important;
        border: none; border-radius: 12px;
        padding: 14px 32px;
        font-size: 1rem !important; font-weight: 700 !important;
        letter-spacing: 0.03em; cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(118,185,0,0.3);
        width: 100%; text-transform: uppercase;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(118,185,0,0.5);
        background: linear-gradient(135deg, #8fd400 0%, #76b900 100%);
    }
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(118,185,0,0.3);
    }

    [data-testid="stExpander"] {
        background: rgba(22,27,39,0.6);
        border: 1px solid rgba(118,185,0,0.2);
        border-radius: 12px; backdrop-filter: blur(10px);
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex; width: 100%;
        background: rgba(15,20,30,0.8);
        border-radius: 12px; padding: 4px; gap: 4px;
        border: 1px solid rgba(118,185,0,0.15);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        flex-grow: 1; justify-content: center;
        background: transparent; border-radius: 8px;
        color: #94a3b8 !important; font-weight: 500;
        padding: 10px 24px; transition: all 0.2s ease;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, #76b900, #5a8f00) !important;
        color: #0a0a0f !important; font-weight: 700 !important;
    }

    [data-testid="stSpinner"] { color: #76b900; }

    [data-testid="stSelectbox"] > div > div {
        background: rgba(22,27,39,0.95) !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSelectbox"] > div > div > div,
    [data-testid="stSelectbox"] > div > div svg {
        color: #e2e8f0 !important;
        fill: #e2e8f0 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] [data-baseweb="menu"] {
        background: #161b27 !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
    }

    [data-baseweb="popover"] li,
    [data-baseweb="popover"] [role="option"] {
        background: #161b27 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] [role="option"]:hover {
        background: rgba(118,185,0,0.15) !important;
        color: #e2e8f0 !important;
    }

    [data-baseweb="popover"] [aria-selected="true"],
    [data-baseweb="popover"] li[aria-selected="true"] {
        background: rgba(118,185,0,0.22) !important;
        color: #76b900 !important;
        font-weight: 600 !important;
    }

    [data-baseweb="popover"] [aria-selected="true"]:hover,
    [data-baseweb="popover"] li[aria-selected="true"]:hover {
        background: rgba(118,185,0,0.30) !important;
        color: #76b900 !important;
        font-weight: 600 !important;
    }

    [data-baseweb="popover"] ::-webkit-scrollbar { width: 4px; }
    [data-baseweb="popover"] ::-webkit-scrollbar-track { background: #0d1117; }
    [data-baseweb="popover"] ::-webkit-scrollbar-thumb { background: rgba(118,185,0,0.4); border-radius: 2px; }

    hr { border: none; border-top: 1px solid rgba(118,185,0,0.15); margin: 24px 0; }

    .glass-card {
        background: linear-gradient(135deg, rgba(22,27,39,0.85), rgba(15,20,30,0.85));
        border: 1px solid rgba(118,185,0,0.2);
        border-radius: 20px; padding: 28px 32px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 24px;
    }
    .glass-card h2, .glass-card h3 { color: #f1f5f9; margin-bottom: 8px; }

    .hero-banner {
        background: linear-gradient(135deg, rgba(118,185,0,0.08) 0%, rgba(22,27,39,0.95) 60%, rgba(15,20,30,0.95) 100%);
        border: 1px solid rgba(118,185,0,0.3);
        border-radius: 24px; padding: 40px 48px; margin-bottom: 32px;
        box-shadow: 0 16px 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative; overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute; top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(118,185,0,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .section-header h3 { color: #f1f5f9; font-size: 1.25rem; font-weight: 700; margin: 0; }

    .section-badge {
        background: rgba(118,185,0,0.15);
        border: 1px solid rgba(118,185,0,0.4);
        color: #76b900; font-size: 0.7rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.08em;
    }

    .info-box {
        background: rgba(118,185,0,0.06);
        border-left: 3px solid #76b900;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px; margin: 12px 0;
        color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;
    }
    .warning-box {
        background: rgba(251,191,36,0.06);
        border-left: 3px solid #fbbf24;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px; margin: 12px 0;
        color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15,20,30,0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(118,185,0,0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(118,185,0,0.7); }

    .sidebar-stat {
        background: rgba(118,185,0,0.07);
        border: 1px solid rgba(118,185,0,0.18);
        border-radius: 10px; padding: 12px 16px; margin: 8px 0;
        display: flex; justify-content: space-between; align-items: center;
    }
    .sidebar-stat-label {
        color: #cbd5e1; font-size: 0.78rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .sidebar-stat-value { color: #f1f5f9; font-size: 0.95rem; font-weight: 700; }

    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        background: #4ade80; border-radius: 50%;
        margin-right: 6px; animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }
        50%       { opacity: 0.8; box-shadow: 0 0 0 4px rgba(74,222,128,0); }
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label { color: #cbd5e1 !important; }

    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
        border: none; border-radius: 12px;
        padding: 14px 32px;
        font-size: 1rem !important; font-weight: 700 !important;
        letter-spacing: 0.03em; cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(229,46,113,0.4);
        width: 100%; text-transform: uppercase;
    }
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(255,138,0,0.5);
        background: linear-gradient(90deg, #ff9e23, #fa4a88);
    }
    [data-testid="stDownloadButton"] > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(229,46,113,0.4);
    }
</style>
""", unsafe_allow_html=True)

apply_theme_css(effective_theme)

# ==============================
# 🔹 Helper: Plotly Theme Config
# ==============================
def get_plotly_layout():
    if IS_DARK:
        return dict(
            paper_bgcolor='rgba(13,17,23,0)',
            plot_bgcolor='rgba(13,17,23,0)',
            font=dict(family='Inter', color='#94a3b8', size=12),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                linecolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#64748b'),
                title_font=dict(color='#94a3b8'),
                showgrid=True, zeroline=False
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                linecolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#64748b'),
                title_font=dict(color='#94a3b8'),
                showgrid=True, zeroline=False
            ),
            legend=dict(
                bgcolor='rgba(15,20,30,0.8)',
                bordercolor='rgba(118,185,0,0.3)',
                borderwidth=1,
                font=dict(color='#cbd5e1')
            ),
            margin=dict(l=16, r=16, t=48, b=16),
            hoverlabel=dict(
                bgcolor='rgba(15,20,30,0.95)',
                bordercolor='rgba(118,185,0,0.5)',
                font=dict(color='#f1f5f9', family='Inter')
            )
        )
    else:
        return dict(
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,255,255,0)',
            font=dict(family='Inter', color='#3d5a1a', size=12),
            xaxis=dict(
                gridcolor='rgba(118,185,0,0.10)',
                linecolor='rgba(118,185,0,0.20)',
                tickfont=dict(color='#5a7a3a'),
                title_font=dict(color='#3d5a1a'),
                showgrid=True, zeroline=False
            ),
            yaxis=dict(
                gridcolor='rgba(118,185,0,0.10)',
                linecolor='rgba(118,185,0,0.20)',
                tickfont=dict(color='#5a7a3a'),
                title_font=dict(color='#3d5a1a'),
                showgrid=True, zeroline=False
            ),
            legend=dict(
                bgcolor='rgba(255,255,255,0.90)',
                bordercolor='rgba(118,185,0,0.35)',
                borderwidth=1,
                font=dict(color='#2d4a0e')
            ),
            margin=dict(l=16, r=16, t=48, b=16),
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.97)',
                bordercolor='rgba(118,185,0,0.5)',
                font=dict(color='#1a2e05', family='Inter')
            )
        )


# ==============================
# 🔹 Load Model (Cached)
# ==============================
@st.cache_resource
def load_nvidia_model():
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        return model
    except Exception as e:
        return None


# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data(ttl=3600)
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max', auto_adjust=True)
        return data
    except Exception as e:
        return None


# NOTE: TTL lowered to keep the displayed quote blocks more real-time
@st.cache_data(ttl=60)
def get_live_quote(ticker='NVDA'):
    try:
        t = yf.Ticker(ticker)
        # Fetch standard info dict safely
        try:
            info = t.info
        except Exception:
            info = {}
            
        hist = t.history(period='5d')
        if len(hist) >= 2:
            prev_close = float(hist['Close'].iloc[-2])
            curr_price = float(hist['Close'].iloc[-1])
            recent_vol = float(hist['Volume'].mean()) # Guaranteed solid fallback
        else:
            curr_price = float(info.get('currentPrice', info.get('previousClose', 0.0)))
            prev_close = float(info.get('previousClose', curr_price))
            recent_vol = 0.0
            
        change = curr_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
        
        # Safely extract Market Cap and Volume
        market_cap = info.get('marketCap')
        if not market_cap:
            try:
                market_cap = getattr(t.fast_info, 'market_cap', None)
            except Exception:
                market_cap = None
                
        # Better volume extraction: try standard keys, then fast_info, then fallback to historical mean
        volume = info.get('averageVolume') or info.get('averageDailyVolume10Day') or info.get('regularMarketVolume')
        if not volume:
            try:
                volume = getattr(t.fast_info, 'last_volume', None)
            except Exception:
                pass
        # Final fallback guarantees a number if history works
        if not volume and recent_vol > 0:
            volume = recent_vol

        return {
            'price': curr_price,
            'prev_close': prev_close,
            'change': change,
            'change_pct': change_pct,
            'market_cap': market_cap,
            'volume': volume,
        }
    except Exception:
        # Fallback to ensure the variables are ALWAYS visible even if API limits are hit completely
        return {
            'price': 0.0,
            'prev_close': 0.0,
            'change': 0.0,
            'change_pct': 0.0,
            'market_cap': None,
            'volume': None
        }


# ==============================
# 🔹 Business Days
# ==============================
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()


# ==============================
# 🔹 Prediction Function
# ==============================
def predict_next_business_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    last_sequence = data_scaled[-look_back:]
    predictions = []
    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


# ==============================
# 🔹 Chart Builders
# ==============================
def build_candlestick_chart(stock_data, predictions, prediction_dates, lookback_days=90):
    PLOTLY_LAYOUT = get_plotly_layout()
    df = stock_data.tail(lookback_days).copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=('', '')
    )

    if IS_DARK:
        inc_line, inc_fill   = '#4ade80', 'rgba(74,222,128,0.8)'
        dec_line, dec_fill   = '#f87171', 'rgba(248,113,113,0.8)'
        ma20_color, ma50_color = '#f59e0b', '#60a5fa'
        marker_border        = '#0a0a0f'
        vol_up, vol_dn       = '#4ade80', '#f87171'
        title_color          = '#f1f5f9'
        grid_color           = 'rgba(255,255,255,0.04)'
        tri_up_color         = '#4ade80'
        tri_dn_color         = '#f87171'
    else:
        inc_line, inc_fill   = '#16a34a', 'rgba(22,163,74,0.85)'
        dec_line, dec_fill   = '#dc2626', 'rgba(220,38,38,0.85)'
        ma20_color, ma50_color = '#d97706', '#2563eb'
        marker_border        = '#ffffff'
        vol_up, vol_dn       = '#16a34a', '#dc2626'
        title_color          = '#1a2e05'
        grid_color           = 'rgba(118,185,0,0.08)'
        tri_up_color         = '#16a34a'
        tri_dn_color         = '#dc2626'

    close_series = df['Close'].squeeze()
    open_series  = df['Open'].squeeze()
    high_series  = df['High'].squeeze()
    low_series   = df['Low'].squeeze()
    ma20 = close_series.rolling(window=20).mean()
    ma50 = close_series.rolling(window=50).mean()

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        name='OHLC',
        increasing=dict(line=dict(color=inc_line, width=1), fillcolor=inc_fill),
        decreasing=dict(line=dict(color=dec_line, width=1), fillcolor=dec_fill),
        whiskerwidth=0.5,
        hoverinfo='none',
    ), row=1, col=1)

    # ── Build per-point hovertext: centered date + colored triangles ──
    hover_texts = []
    for i in range(len(df)):
        date_str = df.index[i].strftime('%b %d, %Y')
        o = open_series.iloc[i]
        h = high_series.iloc[i]
        l = low_series.iloc[i]
        c = close_series.iloc[i]
        m20 = ma20.iloc[i]
        m50 = ma50.iloc[i]

        is_green = c >= o
        triangle  = '▲' if is_green else '▼'
        tri_color = tri_up_color if is_green else tri_dn_color

        txt = (
            f"<b style='display:block; text-align:center;'>"
            f"<span style='color:{tri_color};'>{triangle}</span>"
            f"&nbsp;{date_str}&nbsp;"
            f"</b>"
            f"<br>Open  : ${o:.2f}"
            f"<br>High  : ${h:.2f}"
            f"<br>Low   : ${l:.2f}"
            f"<br>Close : ${c:.2f}"
        )
        if not np.isnan(m20):
            txt += f"<br>MA 20 : ${m20:.2f}"
        if not np.isnan(m50):
            txt += f"<br>MA 50 : ${m50:.2f}"

        hover_texts.append(txt)

    # ── Invisible ghost scatter carrying the full custom tooltip ──
    fig.add_trace(go.Scatter(
        x=df.index,
        y=close_series,
        mode='markers',
        marker=dict(opacity=0, size=12),
        showlegend=False,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts,
        name='',
    ), row=1, col=1)

    # ── MA20 ──
    fig.add_trace(go.Scatter(
        x=df.index, y=ma20, name='MA 20',
        line=dict(color=ma20_color, width=1.5, dash='dot'),
        opacity=0.90,
        hoverinfo='none',
    ), row=1, col=1)

    # ── MA50 ──
    fig.add_trace(go.Scatter(
        x=df.index, y=ma50, name='MA 50',
        line=dict(color=ma50_color, width=1.5, dash='dot'),
        opacity=0.90,
        hoverinfo='none',
    ), row=1, col=1)

    # ── Forecast band & line ──
    if predictions is not None and prediction_dates is not None:
        pred_flat = predictions.flatten()
        last_actual_price = float(df['Close'].iloc[-1])
        pred_x = [df.index[-1]] + list(prediction_dates)
        pred_y = [last_actual_price] + list(pred_flat)

        fig.add_trace(go.Scatter(
            x=pred_x + pred_x[::-1],
            y=[p * 1.015 for p in pred_y] + [p * 0.985 for p in pred_y[::-1]],
            fill='toself', fillcolor='rgba(118,185,0,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Forecast Band', showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y, name='Forecast',
            line=dict(color='#76b900', width=2.5, dash='dash'),
            mode='lines+markers',
            marker=dict(size=7, color='#76b900', symbol='circle',
                        line=dict(color=marker_border, width=1.5)),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Forecast : $%{y:.2f}<extra></extra>',
        ), row=1, col=1)

    # ── Volume bars ──
    colors_vol = [vol_up if c >= o else vol_dn
                  for c, o in zip(close_series, open_series)]

    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'].squeeze(),
        name='Volume', marker_color=colors_vol,
        opacity=0.60, showlegend=False,
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Volume : %{y:,.0f}<extra></extra>',
    ), row=2, col=1)

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>NVDA · Price Action & Forecast</b>',
                   font=dict(size=16, color=title_color), x=0.02),
        xaxis2=dict(**PLOTLY_LAYOUT['xaxis'], rangeslider=dict(visible=False)),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Price (USD)'),
        yaxis2=dict(**PLOTLY_LAYOUT['yaxis'], title='Volume'),
        height=560, dragmode='pan',
        hovermode='closest',
    ))
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)
    return fig


def build_forecast_chart(prediction_dates, predictions, last_actual_price):
    PLOTLY_LAYOUT = get_plotly_layout()
    pred_flat = predictions.flatten()
    dates_full = [pd.Timestamp(prediction_dates[0]) - timedelta(days=1)] + list(prediction_dates)
    prices_full = [last_actual_price] + list(pred_flat)

    if IS_DARK:
        start_color = '#f1f5f9'
        up_color, dn_color = '#4ade80', '#f87171'
        marker_border = '#0a0a0f'
        ref_line_color = 'rgba(148,163,184,0.4)'
        ref_font_color = '#94a3b8'
        title_color = '#f1f5f9'
    else:
        start_color = '#1a2e05'
        up_color, dn_color = '#16a34a', '#dc2626'
        marker_border = '#ffffff'
        ref_line_color = 'rgba(90,122,58,0.45)'
        ref_font_color = '#5a7a3a'
        title_color = '#1a2e05'

    colors = [start_color] + [up_color if p >= last_actual_price else dn_color for p in pred_flat]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_full, y=prices_full,
        fill='tozeroy', fillcolor='rgba(118,185,0,0.06)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=dates_full, y=prices_full, name='Forecast',
        line=dict(color='#76b900', width=2.5),
        mode='lines+markers',
        marker=dict(size=10, color=colors, symbol='circle',
                    line=dict(color=marker_border, width=2)),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price : <b>$%{y:.2f}</b><extra></extra>',
    ))
    fig.add_hline(
        y=last_actual_price,
        line=dict(color=ref_line_color, width=1.5, dash='dot'),
        annotation_text=f'  Last Close: ${last_actual_price:.2f}',
        annotation_font=dict(color=ref_font_color, size=11)
    )

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>Forecast · Next Business Days</b>',
                   font=dict(size=16, color=title_color), x=0.02),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis'], tickformat='%b %d', title='Date'),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Predicted Price (USD)'),
        height=380,
        hovermode='closest',
        showlegend=False
    ))
    fig.update_layout(**layout)
    return fig


def build_returns_chart(stock_data, days=252):
    PLOTLY_LAYOUT = get_plotly_layout()
    df = stock_data.tail(days).copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df['Close'].squeeze()
    returns = close.pct_change().dropna() * 100

    up_color = '#4ade80' if IS_DARK else '#16a34a'
    dn_color = '#f87171' if IS_DARK else '#dc2626'
    title_color = '#f1f5f9' if IS_DARK else '#1a2e05'
    colors = [up_color if r >= 0 else dn_color for r in returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns.index, y=returns.values,
        marker_color=colors, opacity=0.80,
        name='Daily Return %',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Return : <b>%{y:.2f}%</b><extra></extra>',
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>Daily Returns (1Y)</b>',
                   font=dict(size=16, color=title_color), x=0.02),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Return (%)'),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis'], title='Date'),
        height=320,
        hovermode='closest',
    ))
    fig.update_layout(**layout)
    return fig


def build_volume_profile(stock_data, days=90):
    PLOTLY_LAYOUT = get_plotly_layout()
    df = stock_data.tail(days).copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    volume = df['Volume'].squeeze()

    if IS_DARK:
        fill_color = 'rgba(96,165,250,0.15)'
        line_color = '#60a5fa'
        title_color = '#f1f5f9'
    else:
        fill_color = 'rgba(37,99,235,0.10)'
        line_color = '#2563eb'
        title_color = '#1a2e05'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=volume,
        fill='tozeroy', fillcolor=fill_color,
        line=dict(color=line_color, width=1.5),
        name='Volume',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Volume : <b>%{y:,.0f}</b><extra></extra>',
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>Trading Volume (90D)</b>',
                   font=dict(size=16, color=title_color), x=0.02),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Volume'),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis'], title='Date'),
        height=280,
        hovermode='closest',
        showlegend=False
    ))
    fig.update_layout(**layout)
    return fig


# ==============================
# 🔹 Load Model & Initial Data
# ==============================
model = load_nvidia_model()
STOCK = 'NVDA'

# ==============================
# 🔹 Sidebar
# ==============================
with st.sidebar:

    logo_url = (
        'https://raw.githubusercontent.com/MarpakaPradeepSai/NVIDIA-Stock-Price-Predictor/'
        '53b81d17aa5dbac6c1a29830ad4974ecd510a22d/Data/Images%20%26%20GIF/NVIDIA_logo_white.svg'
        if IS_DARK else
        'https://raw.githubusercontent.com/MarpakaPradeepSai/NVIDIA-Stock-Price-Predictor/'
        '53b81d17aa5dbac6c1a29830ad4974ecd510a22d/Data/Images%20%26%20GIF/NVIDIA_logo_black.svg'
    )
    subtitle_color = '#64748b' if IS_DARK else '#6b8f3a'
    logo_filter = 'brightness(1.1)' if IS_DARK else 'brightness(0.85) saturate(1.2)'

    st.markdown(f"""
    <div style='text-align:center; padding: 8px 0 16px 0;'>
        <img src='{logo_url}' style='width:160px; filter: {logo_filter};'>
        <p style='color:{subtitle_color}; font-size:0.75rem; margin-top:10px; letter-spacing:0.1em;'>
            STOCK INTELLIGENCE PLATFORM
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Appearance Dropdown ──
    st.markdown("#### 🎨 Appearance")

    theme_options = ['System 🖳', 'Light ☼', 'Dark ⏾']

    current_theme = st.session_state.theme
    if current_theme not in theme_options:
        current_theme = 'System 🖳'
    current_index = theme_options.index(current_theme)

    selected_theme = st.selectbox(
        label="Theme",
        options=theme_options,
        index=current_index,
        key='theme_selectbox',
        label_visibility='collapsed'
    )

    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown("---")

    # Model Status
    if model is not None:
        online_color = '#4ade80' if IS_DARK else '#16a34a'
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:16px;'>
            <span class='status-dot'></span>
            <span style='color:{online_color}; font-size:0.85rem; font-weight:600;'>LSTM Model Online</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        offline_color = '#f87171' if IS_DARK else '#dc2626'
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:16px;'>
            <span style='display:inline-block; width:8px; height:8px; background:{offline_color};
                         border-radius:50%; margin-right:6px;'></span>
            <span style='color:{offline_color}; font-size:0.85rem; font-weight:600;'>Model Offline</span>
        </div>
        """, unsafe_allow_html=True)

    # Model Specs
    st.markdown("#### ⚙️ Model Architecture")
    specs = [
        ("Architecture", "LSTM"),
        ("Look-Back Window", "5 Days"),
        ("Hidden Units", "150"),
        ("RMSE", "1.32"),
        ("Trained On", "Max History"),
    ]
    for label, val in specs:
        st.markdown(f"""
        <div class='sidebar-stat'>
            <span class='sidebar-stat-label'>{label}</span>
            <span class='sidebar-stat-value'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Forecast Settings
    st.markdown("#### 🎯 Forecast Settings")
    num_days = st.slider("Forecast Horizon (Days)", 1, 30, 5,
                         help="Number of business days to predict ahead")

    lookback_chart = st.selectbox(
        "Chart History Window",
        options=[30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"{x} Days"
    )

    st.markdown("---")
    disclaimer_color = '#cbd5e1' if IS_DARK else '#7a9a50'
    st.markdown(f"""
    <div style='color:{disclaimer_color}; font-size:0.72rem; text-align:center; line-height:1.7;'>
        ⚠️ For educational purposes only.<br>
        Not financial advice.<br><br>
        Model predictions are based on<br>
        historical price patterns only.
    </div>
    """, unsafe_allow_html=True)


# ==============================
# 🔹 Hero Header
# ==============================
hero_logo_url = (
    'https://raw.githubusercontent.com/MarpakaPradeepSai/NVIDIA-Stock-Price-Predictor/'
    '53b81d17aa5dbac6c1a29830ad4974ecd510a22d/Data/Images%20%26%20GIF/NVIDIA_logo_white.svg'
    if IS_DARK else
    'https://raw.githubusercontent.com/MarpakaPradeepSai/NVIDIA-Stock-Price-Predictor/'
    '53b81d17aa5dbac6c1a29830ad4974ecd510a22d/Data/Images%20%26%20GIF/NVIDIA_logo_black.svg'
)
hero_logo_filter  = 'brightness(1.1)' if IS_DARK else 'brightness(0.82) saturate(1.2)'
hero_logo_opacity = '0.92' if IS_DARK else '0.90'
hero_subtitle_color = '#64748b' if IS_DARK else '#5a7a3a'
hero_tag_color    = '#76b900' if IS_DARK else '#4a7c00'
hero_h1_color     = '#f1f5f9' if IS_DARK else '#1a2e05'

st.markdown(f"""
<div class='hero-banner'>
    <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:20px;'>
        <div>
            <div style='color:{hero_tag_color}; font-size:0.8rem; font-weight:700;
                        letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;'>
                AI-Powered · LSTM Neural Network
            </div>
            <h1 style='color:{hero_h1_color}; font-size:2.4rem; font-weight:800; margin:0; line-height:1.2;'>
                NVIDIA Stock <span style='color:#76b900;'>Predictor</span>
            </h1>
            <p style='color:{hero_subtitle_color}; margin-top:10px; font-size:1rem;
                      max-width:520px; line-height:1.6;'>
                Deep learning-powered price forecasting using Long Short-Term Memory networks
                trained on NVDA's complete trading history.
            </p>
        </div>
        <div style='text-align:right;'>
            <img src='{hero_logo_url}'
                 style='width:200px; opacity:{hero_logo_opacity}; filter: {hero_logo_filter};'>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# 🔹 Session State Init
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'last_num_days' not in st.session_state:
    st.session_state.last_num_days = 5


# ==============================
# 🔹 Live Quote Blocks (Placed right above the Forecast button)
# ==============================
quote = get_live_quote(STOCK)

if quote:
    change_arrow = '▲' if quote['change'] >= 0 else '▼'

    # Make "NVIDIA LAST PRICE" slightly bigger (wider column + delta line)
    c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])

    with c1:
        st.metric(
            "NVIDIA LAST PRICE",
            f"${quote['price']:.2f}",
            delta=f"{change_arrow} {abs(quote['change']):.2f} ({abs(quote['change_pct']):.2f}%)"
        )

    with c2:
        st.metric("PREVIOUS CLOSE", f"${quote['prev_close']:.2f}")

    with c3:
        mktcap = quote.get('market_cap')
        if mktcap and mktcap > 0:
            mktcap_str = f"${mktcap/1e12:.2f}T" if mktcap >= 1e12 else f"${mktcap/1e9:.1f}B"
        else:
            mktcap_str = "N/A"
        st.metric("MARKET CAP", mktcap_str)

    with c4:
        vol = quote.get('volume')
        vol_str = f"{vol/1e6:.1f}M" if vol and vol > 0 else "N/A"
        st.metric("AVG VOLUME", vol_str)


# ==============================
# 🔹 Predict Button
# ==============================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_prediction = st.button(
        f"🚀 Generate {num_days}-Day Forecast",
        key='forecast-button',
        use_container_width=True
    )

st.markdown("---")

if run_prediction:
    if model is None:
        st.markdown("""
        <div class='warning-box'>
            <b>⚠️ Model Not Available</b><br>
            The LSTM model file could not be loaded. Please verify the model path and file integrity.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("⚡ Running LSTM inference..."):
            stock_data = get_stock_data(STOCK)

            if stock_data is None or stock_data.empty:
                st.markdown("""
                <div class='warning-box'>
                    ❌ Failed to fetch stock data. Please check your internet connection.
                </div>
                """, unsafe_allow_html=True)
            else:
                close_prices = stock_data['Close'].values.reshape(-1, 1)
                dates = stock_data.index

                predictions = predict_next_business_days(
                    model, close_prices, look_back=5, days=num_days
                )

                last_date = dates[-1]
                prediction_dates = generate_business_days(
                    last_date + timedelta(days=1), num_days
                )

                st.session_state.prediction_results = {
                    'stock_data': stock_data,
                    'close_prices': close_prices,
                    'dates': dates,
                    'predictions': predictions,
                    'prediction_dates': prediction_dates,
                    'num_days': num_days,
                    'stock': STOCK,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

# ==============================
# 🔹 Display Results
# ==============================
if st.session_state.prediction_results is not None:
    r = st.session_state.prediction_results

    stock_data       = r['stock_data']
    close_prices     = r['close_prices']
    predictions      = r['predictions']
    prediction_dates = r['prediction_dates']
    stored_num_days  = r['num_days']
    ts               = r.get('timestamp', '')
    pred_flat        = predictions.flatten()

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data_display = stock_data.copy()
        stock_data_display.columns = [col[0] for col in stock_data_display.columns]
    else:
        stock_data_display = stock_data

    last_actual_price = float(stock_data_display['Close'].iloc[-1])
    final_pred_price  = float(pred_flat[-1])
    pred_change       = final_pred_price - last_actual_price
    pred_change_pct   = (pred_change / last_actual_price) * 100

    summary_ts_color = '#64748b' if IS_DARK else '#6b8f3a'
    st.markdown(f"""
    <div class='glass-card'>
        <div class='section-header'>
            <h3>📊 Forecast Summary</h3>
            <span class='section-badge'>LSTM Prediction</span>
        </div>
        <p style='color:{summary_ts_color}; font-size:0.82rem; margin:-8px 0 16px 0;'>
            Generated at {ts} · {stored_num_days}-day horizon
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Last Close", f"${last_actual_price:.2f}")
    with m2:
        st.metric("Predicted Day 1", f"${pred_flat[0]:.2f}",
                  delta=f"{pred_flat[0]-last_actual_price:+.2f}")
    with m3:
        direction = "▲" if pred_change >= 0 else "▼"
        st.metric(f"End of Forecast ({stored_num_days}D)", f"${final_pred_price:.2f}",
                  delta=f"{direction} {abs(pred_change_pct):.2f}%")
    with m4:
        avg_pred = float(np.mean(pred_flat))
        st.metric("Avg Forecast Price", f"${avg_pred:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Price Action & Forecast",
        "🔮  Forecast Detail",
        "📉  Returns Analysis",
        "📋  Historical Data"
    ])

    with tab1:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        fig_candle = build_candlestick_chart(
            stock_data, predictions, prediction_dates,
            lookback_days=lookback_chart
        )
        st.plotly_chart(fig_candle, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'displaylogo': False,
            'scrollZoom': True
        })
        st.markdown("""
        <div class='info-box'>
            🕯️ <b>Reading the chart:</b> Green candles indicate price closed higher than open;
            red candles indicate the opposite. The dashed green line represents the LSTM model's
            forecast trajectory. MA 20 and MA 50 are moving averages overlaid for trend reference.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        fig_forecast = build_forecast_chart(prediction_dates, predictions, last_actual_price)
        st.plotly_chart(fig_forecast, use_container_width=True, config={
            'displayModeBar': False, 'displaylogo': False
        })

        st.markdown("<br>", unsafe_allow_html=True)

        pred_df = pd.DataFrame({
            'Business Day':    [f"Day {i+1}" for i in range(stored_num_days)],
            'Date':            [d.strftime('%A, %b %d %Y') for d in prediction_dates],
            'Predicted Price': [f"${p:.2f}" for p in pred_flat],
            'Change vs Close': [f"{p - last_actual_price:+.2f}" for p in pred_flat],
            'Change %':        [f"{((p - last_actual_price)/last_actual_price*100):+.2f}%" for p in pred_flat],
            'Signal':          ['🟢 BUY' if p >= last_actual_price else '🔴 SELL' for p in pred_flat]
        })

        st.markdown("""
        <div class='section-header' style='margin-bottom:12px;'>
            <h3>Detailed Forecast Table</h3>
            <span class='section-badge'>Day-by-Day</span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            pred_df, use_container_width=True, hide_index=True,
            column_config={
                'Signal': st.column_config.TextColumn('Signal', width='small'),
                'Predicted Price': st.column_config.TextColumn('Predicted Price', width='medium'),
            }
        )

        st.markdown("""
        <div class='warning-box'>
            ⚠️ <b>Disclaimer:</b> Signals shown are derived purely from model output relative to last
            close price. They are <b>not</b> financial advice. Past model performance does not
            guarantee future accuracy. Always consult a licensed financial advisor.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        fig_ret = build_returns_chart(stock_data, days=252)
        st.plotly_chart(fig_ret, use_container_width=True, config={
            'displayModeBar': False, 'displaylogo': False
        })

        fig_vol = build_volume_profile(stock_data, days=lookback_chart)
        st.plotly_chart(fig_vol, use_container_width=True, config={
            'displayModeBar': False, 'displaylogo': False
        })

        close_s = stock_data_display['Close'].squeeze()
        ret_1y  = close_s.tail(252).pct_change().dropna() * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <h3>Return Statistics (1Y)</h3>
            <span class='section-badge'>Annualized</span>
        </div>
        """, unsafe_allow_html=True)

        rs1, rs2, rs3, rs4 = st.columns(4)
        with rs1:
            st.metric("Mean Daily Return", f"{ret_1y.mean():.3f}%")
        with rs2:
            st.metric("Std Deviation", f"{ret_1y.std():.3f}%")
        with rs3:
            st.metric("Best Day", f"+{ret_1y.max():.2f}%")
        with rs4:
            st.metric("Worst Day", f"{ret_1y.min():.2f}%")

    with tab4:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='section-header'>
            <h3>Full Historical Dataset</h3>
            <span class='section-badge'>Max History</span>
        </div>
        """, unsafe_allow_html=True)

        disp = stock_data_display.sort_index(ascending=False).copy()
        for col in disp.select_dtypes(include=np.number).columns:
            disp[col] = disp[col].round(4)

        st.dataframe(disp, height=480, use_container_width=True)

        csv = disp.to_csv().encode('utf-8')
        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
        with col_dl2:
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"NVDA_historical_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                use_container_width=True
            )

else:
    placeholder_h2 = '#f1f5f9' if IS_DARK else '#1a2e05'
    placeholder_p  = '#64748b' if IS_DARK else '#5a7a3a'
    placeholder_b  = '#76b900' if IS_DARK else '#4a7c00'

    st.markdown(f"""
    <div class='glass-card' style='text-align:center; padding: 60px 40px;'>
        <div style='font-size:4rem; margin-bottom:16px;'>📡</div>
        <h2 style='color:{placeholder_h2}; font-size:1.6rem; margin-bottom:12px;'>Ready to Forecast</h2>
        <p style='color:{placeholder_p}; max-width:420px; margin:0 auto; line-height:1.7; font-size:0.95rem;'>
            Configure your forecast horizon in the sidebar, then click
            <b style='color:{placeholder_b};'>Generate Forecast</b> to run the LSTM model
            and visualize predicted NVDA price movements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading market data..."):
        stock_data_preview = get_stock_data(STOCK)
        if stock_data_preview is not None and not stock_data_preview.empty:
            fig_prev = build_candlestick_chart(
                stock_data_preview, None, None, lookback_days=90
            )
            st.plotly_chart(fig_prev, use_container_width=True, config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'displaylogo': False,
                'scrollZoom': True
            })
