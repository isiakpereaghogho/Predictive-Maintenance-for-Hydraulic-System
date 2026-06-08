import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title='Hydraulic System Predictive Maintenance',
    page_icon='🏭',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── API Base URL ──────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary    : #0a0e1a;
        --bg-secondary  : #111827;
        --bg-card       : #1a2235;
        --accent-cyan   : #00d4ff;
        --accent-orange : #ff6b35;
        --accent-green  : #00ff88;
        --accent-red    : #ff4757;
        --text-primary  : #e8eaf0;
        --text-secondary: #8892a4;
        --border        : rgba(0, 212, 255, 0.15);
    }

    /* ── Global ── */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Barlow', sans-serif;
        color: var(--text-primary);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1420 0%, #111827 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-secondary) !important;
        font-family: 'Barlow', sans-serif;
        font-size: 0.9rem;
        padding: 8px 0;
    }

    /* ── Headers ── */
    h1, h2, h3 {
        font-family: 'Barlow', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), transparent);
    }

    .metric-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-cyan);
        line-height: 1;
    }

    .metric-unit {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .status-critical {
        background: rgba(255, 71, 87, 0.15);
        border: 1px solid var(--accent-red);
        color: var(--accent-red);
    }

    .status-warning {
        background: rgba(255, 107, 53, 0.15);
        border: 1px solid var(--accent-orange);
        color: var(--accent-orange);
    }

    .status-normal {
        background: rgba(0, 255, 136, 0.15);
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
    }

    /* ── Section Header ── */
    .section-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    /* ── Input Fields ── */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), #0099cc) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Barlow', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
    }

    /* ── Alert Box ── */
    .alert-box {
        background: var(--bg-card);
        border-left: 4px solid var(--accent-cyan);
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .alert-critical { border-left-color: var(--accent-red); }
    .alert-warning  { border-left-color: var(--accent-orange); }
    .alert-success  { border-left-color: var(--accent-green); }

    /* ── Logo ── */
    .logo-text {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        color: var(--accent-cyan);
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .logo-sub {
        font-size: 0.65rem;
        color: var(--text-secondary);
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* ── Divider ── */
    .cyber-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
        margin: 24px 0;
        opacity: 0.4;
    }

    /* ── Result Panel ── */
    .result-panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }

    .rul-display {
        font-family: 'Share Tech Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }

    /* ── Table ── */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
    }

    .styled-table th {
        background: rgba(0, 212, 255, 0.1);
        color: var(--accent-cyan);
        padding: 10px 14px;
        text-align: left;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid var(--border);
    }

    .styled-table td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        color: var(--text-primary);
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def get_rul_status(rul):
    if rul <= 24:
        return 'CRITICAL', 'status-critical', 'alert-critical'
    elif rul <= 72:
        return 'WARNING', 'status-warning', 'alert-warning'
    else:
        return 'NORMAL', 'status-normal', 'alert-success'


def call_predict_api(payload):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the FastAPI server is running."
    except Exception as e:
        return None, str(e)


def call_train_api():
    try:
        response = requests.post(f"{API_URL}/train", timeout=300)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Training Error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API."
    except Exception as e:
        return None, str(e)


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def make_gauge(value, max_value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': '#8892a4', 'size': 12, 'family': 'Share Tech Mono'}},
        number={'font': {'color': color, 'size': 28, 'family': 'Share Tech Mono'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickcolor': '#8892a4',
                     'tickfont': {'color': '#8892a4', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#1a2235',
            'borderwidth': 0,
            'steps': [
                {'range': [0, max_value * 0.33], 'color': 'rgba(255,71,87,0.1)'},
                {'range': [max_value * 0.33, max_value * 0.66], 'color': 'rgba(255,107,53,0.1)'},
                {'range': [max_value * 0.66, max_value], 'color': 'rgba(0,255,136,0.1)'}
            ],
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Share Tech Mono')
    )
    return fig


def make_rul_timeline(rul_value, max_rul=336):
    remaining_pct = rul_value / max_rul
    color = '#ff4757' if remaining_pct < 0.2 else '#ff6b35' if remaining_pct < 0.4 else '#00ff88'

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[rul_value],
        y=['RUL'],
        orientation='h',
        marker=dict(
            color=color,
            line=dict(width=0)
        ),
        showlegend=False
    ))
    fig.add_trace(go.Bar(
        x=[max_rul - rul_value],
        y=['RUL'],
        orientation='h',
        marker=dict(color='rgba(255,255,255,0.05)', line=dict(width=0)),
        showlegend=False
    ))
    fig.update_layout(
        barmode='stack',
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, max_rul],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div class='logo-text'>⚙ HYDRA<span style='color:#ff6b35'>SYS</span></div>
        <div class='logo-sub'>Predictive Maintenance</div>
    </div>
    <div class='cyber-divider'></div>
    """, unsafe_allow_html=True)

    # API Status
    api_online = check_api_health()
    status_color = '#00ff88' if api_online else '#ff4757'
    status_text  = 'ONLINE' if api_online else 'OFFLINE'
    st.markdown(f"""
    <div style='margin-bottom:20px;'>
        <span style='font-family:Share Tech Mono;font-size:0.65rem;
        color:#8892a4;letter-spacing:2px;'>API STATUS</span><br>
        <span style='font-family:Share Tech Mono;font-size:0.85rem;
        color:{status_color};'>● {status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)

    # Navigation
    st.markdown("""
    <div style='font-family:Share Tech Mono;font-size:0.65rem;
    color:#8892a4;letter-spacing:3px;margin-bottom:12px;'>NAVIGATION</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=["🏠  Home", "🔮  RUL Prediction", "📊  Dashboard", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)

    # Machine selector
    st.markdown("""
    <div style='font-family:Share Tech Mono;font-size:0.65rem;
    color:#8892a4;letter-spacing:3px;margin-bottom:12px;'>MACHINE SELECT</div>
    """, unsafe_allow_html=True)

    selected_machine = st.selectbox(
        label="",
        options=[f"HPU_0{i}" for i in range(1, 10)],
        label_visibility="collapsed"
    )

    st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Share Tech Mono;font-size:0.6rem;
    color:#3a4255;letter-spacing:1px;margin-top:auto;'>
    v1.0.0 · GB Model · R²=0.9365
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════

if "🏠" in page:

    st.markdown("""
    <h1 style='font-size:2.2rem;margin-bottom:4px;'>
        Hydraulic System
        <span style='color:#00d4ff;'>Predictive Maintenance</span>
    </h1>
    <p style='color:#8892a4;font-size:0.95rem;margin-bottom:32px;'>
        Real-time Remaining Useful Life prediction for hydraulic pump units
    </p>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────
    st.markdown("<div class='section-header'>System Overview</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Active Machines</div>
            <div class='metric-value'>9</div>
            <div class='metric-unit'>HPU Units Monitored</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Model Accuracy</div>
            <div class='metric-value'>93.6<span style='font-size:1.2rem'>%</span></div>
            <div class='metric-unit'>Test R² Score</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Prediction Error</div>
            <div class='metric-value'>14.3</div>
            <div class='metric-unit'>RMSE in Hours</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Max RUL Range</div>
            <div class='metric-value'>336</div>
            <div class='metric-unit'>Hours (~14 Days)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # ── Model Info + Pipeline ─────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Best Model — Gradient Boosting</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <table class='styled-table'>
                <tr><th>Metric</th><th>Train</th><th>Test</th><th>Gap</th></tr>
                <tr><td>RMSE</td><td>17.04</td><td style='color:#00ff88'>14.32</td><td>2.71</td></tr>
                <tr><td>MAE</td><td>10.54</td><td style='color:#00ff88'>11.16</td><td>0.62</td></tr>
                <tr><td>R²</td><td>0.9611</td><td style='color:#00ff88'>0.9365</td><td>0.0247</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header'>Pipeline Architecture</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card' style='font-family:Share Tech Mono;font-size:0.78rem;line-height:2;'>
            <span style='color:#00d4ff;'>01</span>
            <span style='color:#8892a4;'> ──</span>
            <span style='color:#e8eaf0;'> Raw Sensor Data (864K rows)</span><br>
            <span style='color:#00d4ff;'>02</span>
            <span style='color:#8892a4;'> ──</span>
            <span style='color:#e8eaf0;'> Filter → is_anomaly = 1</span><br>
            <span style='color:#00d4ff;'>03</span>
            <span style='color:#8892a4;'> ──</span>
            <span style='color:#e8eaf0;'> Feature Engineering (36 features)</span><br>
            <span style='color:#00d4ff;'>04</span>
            <span style='color:#8892a4;'> ──</span>
            <span style='color:#e8eaf0;'> Chronological 80/20 Split</span><br>
            <span style='color:#00d4ff;'>05</span>
            <span style='color:#8892a4;'> ──</span>
            <span style='color:#e8eaf0;'> Gradient Boosting Regressor</span><br>
            <span style='color:#00d4ff;'>06</span>
            <span style='color:#8892a4;'> ──</span>
            <span style='color:#e8eaf0;'> MLflow + DagsHub Tracking</span><br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # ── Features Used ─────────────────────────────────────
    st.markdown("<div class='section-header'>Feature Groups</div>",
                unsafe_allow_html=True)

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    feature_groups = [
        ("Sensor Signals", "2", "temp_celsius, pump_rpm"),
        ("Delta Features", "6", "Rate of change per sensor"),
        ("Lag Features", "12", "Lags at 1, 3, 6 minutes"),
        ("Rolling Stats", "8", "15-min mean & std"),
        ("Interaction", "5", "vibration_magnitude, hydraulic_power..."),
    ]
    for col, (name, count, desc) in zip([fc1, fc2, fc3, fc4, fc5], feature_groups):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div class='metric-value' style='font-size:1.8rem;'>{count}</div>
                <div class='metric-label' style='margin-top:8px;'>{name}</div>
                <div style='font-size:0.65rem;color:#3a4255;margin-top:6px;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: RUL PREDICTION
# ══════════════════════════════════════════════════════════

elif "🔮" in page:

    st.markdown("""
    <h1 style='font-size:2rem;margin-bottom:4px;'>
        RUL <span style='color:#00d4ff;'>Prediction</span>
    </h1>
    <p style='color:#8892a4;font-size:0.9rem;margin-bottom:28px;'>
        Enter live sensor readings to predict Remaining Useful Life
    </p>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-header'>Sensor Input</div>",
                    unsafe_allow_html=True)

        with st.container():
            machine_id = st.selectbox(
                "Machine ID",
                options=[f"HPU_0{i}" for i in range(1, 10)],
                index=[f"HPU_0{i}" for i in range(1, 10)].index(selected_machine)
            )

            st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-family:Share Tech Mono;font-size:0.65rem;
            color:#8892a4;letter-spacing:2px;margin-bottom:12px;'>
            PRESSURE & TEMPERATURE</div>
            """, unsafe_allow_html=True)

            p1, p2 = st.columns(2)
            with p1:
                pressure_bar = st.number_input(
                    "Pressure (bar)", min_value=0.0,
                    max_value=500.0, value=115.0, step=0.1,
                    format="%.2f"
                )
            with p2:
                temp_celsius = st.number_input(
                    "Temperature (°C)", min_value=0.0,
                    max_value=150.0, value=52.0, step=0.1,
                    format="%.2f"
                )

            st.markdown("""
            <div style='font-family:Share Tech Mono;font-size:0.65rem;
            color:#8892a4;letter-spacing:2px;margin:12px 0;'>
            FLOW & VIBRATION</div>
            """, unsafe_allow_html=True)

            f1, f2, f3 = st.columns(3)
            with f1:
                flow_lpm = st.number_input(
                    "Flow (lpm)", min_value=0.0,
                    max_value=200.0, value=88.0, step=0.1,
                    format="%.2f"
                )
            with f2:
                vibration_x = st.number_input(
                    "Vibration X (g)", min_value=0.0,
                    max_value=10.0, value=0.35, step=0.01,
                    format="%.3f"
                )
            with f3:
                vibration_y = st.number_input(
                    "Vibration Y (g)", min_value=0.0,
                    max_value=10.0, value=0.35, step=0.01,
                    format="%.3f"
                )

            st.markdown("""
            <div style='font-family:Share Tech Mono;font-size:0.65rem;
            color:#8892a4;letter-spacing:2px;margin:12px 0;'>
            PUMP SPEED</div>
            """, unsafe_allow_html=True)

            pump_rpm = st.number_input(
                "Pump RPM", min_value=0.0,
                max_value=3000.0, value=1473.0, step=1.0,
                format="%.1f"
            )

            st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

            predict_btn = st.button(
                "⚡  PREDICT REMAINING USEFUL LIFE",
                use_container_width=True
            )

    with col_result:
        st.markdown("<div class='section-header'>Prediction Result</div>",
                    unsafe_allow_html=True)

        if predict_btn:
            payload = {
                "machine_id"   : machine_id,
                "pressure_bar" : pressure_bar,
                "temp_celsius" : temp_celsius,
                "flow_lpm"     : flow_lpm,
                "vibration_x_g": vibration_x,
                "vibration_y_g": vibration_y,
                "pump_rpm"     : pump_rpm
            }

            with st.spinner("Querying prediction model..."):
                result, error = call_predict_api(payload)

            if error:
                st.markdown(f"""
                <div class='alert-box alert-critical'>
                    <div style='font-family:Share Tech Mono;font-size:0.75rem;
                    color:#ff4757;letter-spacing:1px;'>API ERROR</div>
                    <div style='color:#e8eaf0;margin-top:6px;font-size:0.85rem;'>{error}</div>
                </div>""", unsafe_allow_html=True)

            else:
                rul = result.get("predicted_rul", 0)
                status, badge_class, alert_class = get_rul_status(rul)

                st.markdown(f"""
                <div class='result-panel'>
                    <div style='font-family:Share Tech Mono;font-size:0.65rem;
                    color:#8892a4;letter-spacing:3px;margin-bottom:12px;'>
                    PREDICTED RUL</div>
                    <div class='rul-display'>{rul:.1f}</div>
                    <div style='font-family:Share Tech Mono;font-size:0.9rem;
                    color:#8892a4;margin-top:8px;'>HOURS REMAINING</div>
                    <div style='margin-top:16px;'>
                        <span class='status-badge {badge_class}'>{status}</span>
                    </div>
                    <div style='margin-top:8px;font-size:0.75rem;color:#3a4255;
                    font-family:Share Tech Mono;'>
                    ≈ {rul/24:.1f} days · ±14.32 hrs margin
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin:16px 0;'></div>",
                            unsafe_allow_html=True)
                st.plotly_chart(
                    make_rul_timeline(rul),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )

                # Maintenance recommendation
                if status == 'CRITICAL':
                    msg = "⚠️ Immediate maintenance required. Schedule within 24 hours."
                    ac  = "alert-critical"
                elif status == 'WARNING':
                    msg = "🔶 Plan maintenance within the next 3 days."
                    ac  = "alert-warning"
                else:
                    msg = "✅ Machine operating normally. Next check as scheduled."
                    ac  = "alert-success"

                st.markdown(f"""
                <div class='alert-box {ac}' style='margin-top:16px;'>
                    <div style='font-size:0.85rem;color:#e8eaf0;'>{msg}</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class='result-panel' style='opacity:0.5;'>
                <div style='font-family:Share Tech Mono;font-size:0.65rem;
                color:#8892a4;letter-spacing:3px;margin-bottom:16px;'>
                AWAITING INPUT</div>
                <div style='font-size:3rem;margin:16px 0;'>⚙</div>
                <div style='font-family:Share Tech Mono;font-size:0.8rem;
                color:#8892a4;'>Enter sensor readings and click predict</div>
            </div>""", unsafe_allow_html=True)

    # ── Sensor Gauges ──────────────────────────────────────
    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Live Sensor Gauges</div>",
                unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    gauge_data = [
        (pressure_bar, 500,  "PRESSURE (bar)",   "#00d4ff"),
        (temp_celsius, 150,  "TEMPERATURE (°C)", "#ff6b35"),
        (flow_lpm,     200,  "FLOW (lpm)",       "#00ff88"),
        (pump_rpm,     3000, "PUMP RPM",         "#a78bfa"),
    ]
    for col, (val, max_v, title, color) in zip([g1, g2, g3, g4], gauge_data):
        with col:
            st.plotly_chart(
                make_gauge(val, max_v, title, color),
                use_container_width=True,
                config={'displayModeBar': False}
            )


# ══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════

elif "📊" in page:

    st.markdown("""
    <h1 style='font-size:2rem;margin-bottom:4px;'>
        Analytics <span style='color:#00d4ff;'>Dashboard</span>
    </h1>
    <p style='color:#8892a4;font-size:0.9rem;margin-bottom:28px;'>
        Model performance and fleet health overview
    </p>
    """, unsafe_allow_html=True)

    # ── Model Comparison Chart ─────────────────────────────
    st.markdown("<div class='section-header'>Model Comparison</div>",
                unsafe_allow_html=True)

    models      = ['Ridge', 'Random Forest', 'XGBoost', 'Gradient Boosting']
    test_r2     = [0.6482, 0.8942, 0.9329, 0.9365]
    test_rmse   = [33.71,  18.49,  14.72,  14.32]
    gaps        = [0.1980, 0.1294, 0.0292, 0.0247]
    colors      = ['#3a4255', '#8892a4', '#00d4ff', '#00ff88']

    mc1, mc2 = st.columns(2)

    with mc1:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=models, y=test_r2,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.4f}" for v in test_r2],
            textposition='outside',
            textfont=dict(family='Share Tech Mono', size=10, color='#e8eaf0')
        ))
        fig_r2.update_layout(
            title=dict(text='Test R² Score', font=dict(
                color='#8892a4', size=11, family='Share Tech Mono')),
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono'),
                range=[0, 1.1]
            ),
            xaxis=dict(tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono')),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_r2, use_container_width=True,
                        config={'displayModeBar': False})

    with mc2:
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(
            x=models, y=test_rmse,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}h" for v in test_rmse],
            textposition='outside',
            textfont=dict(family='Share Tech Mono', size=10, color='#e8eaf0')
        ))
        fig_rmse.update_layout(
            title=dict(text='Test RMSE (hours)', font=dict(
                color='#8892a4', size=11, family='Share Tech Mono')),
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono')
            ),
            xaxis=dict(tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono')),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_rmse, use_container_width=True,
                        config={'displayModeBar': False})

    # ── Feature Importance ────────────────────────────────
    st.markdown("<div class='section-header'>GB Feature Importance (Top 10)</div>",
                unsafe_allow_html=True)

    features    = ['temp_celsius_roll_mean_15m', 'pressure_bar_roll_mean_15m',
                   'pressure_temp_ratio', 'pump_rpm', 'hydraulic_power',
                   'vibration_x_g_lag6', 'vibration_x_g_lag3', 'vibration_x_g_lag1',
                   'fluid_type', 'vibration_y_g_lag3']
    importances = [0.42, 0.18, 0.14, 0.06, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02]

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        y=features[::-1],
        x=importances[::-1],
        orientation='h',
        marker=dict(
            color=importances[::-1],
            colorscale=[[0, '#1a2235'], [0.5, '#00d4ff'], [1, '#00ff88']],
            line=dict(width=0)
        ),
        text=[f"{v:.2%}" for v in importances[::-1]],
        textposition='outside',
        textfont=dict(family='Share Tech Mono', size=9, color='#e8eaf0')
    ))
    fig_fi.update_layout(
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono'),
            gridcolor='rgba(255,255,255,0.03)'
        ),
        margin=dict(l=0, r=60, t=10, b=0)
    )
    st.plotly_chart(fig_fi, use_container_width=True,
                    config={'displayModeBar': False})

    # ── RUL Distribution ──────────────────────────────────
    st.markdown("<div class='section-header'>Simulated RUL Distribution</div>",
                unsafe_allow_html=True)

    np.random.seed(42)
    rul_sim = np.concatenate([
        np.random.normal(130, 85, 800),
        np.random.normal(50,  30, 200)
    ])
    rul_sim = rul_sim[(rul_sim >= 0) & (rul_sim <= 336)]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=rul_sim, nbinsx=40,
        marker=dict(
            color='#00d4ff',
            opacity=0.7,
            line=dict(width=0)
        ),
        name='RUL Distribution'
    ))
    fig_dist.update_layout(
        height=240,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='RUL (hours)',
            titlefont=dict(color='#8892a4', size=10, family='Share Tech Mono'),
            tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono'),
            gridcolor='rgba(255,255,255,0.04)'
        ),
        yaxis=dict(
            title='Frequency',
            titlefont=dict(color='#8892a4', size=10, family='Share Tech Mono'),
            tickfont=dict(color='#8892a4', size=9, family='Share Tech Mono'),
            gridcolor='rgba(255,255,255,0.04)'
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True,
                    config={'displayModeBar': False})

    # ── Retrain Section ───────────────────────────────────
    st.markdown("<div class='section-header'>Model Retraining</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='alert-box'>
        <div style='font-family:Share Tech Mono;font-size:0.75rem;
        color:#00d4ff;letter-spacing:1px;'>RETRAIN MODEL</div>
        <div style='color:#8892a4;margin-top:6px;font-size:0.85rem;'>
        Trigger a full retraining pipeline using the latest available data.
        This will update the model and reload artifacts automatically.
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("🔄  TRIGGER RETRAINING PIPELINE", use_container_width=False):
        with st.spinner("Training in progress — this may take several minutes..."):
            result, error = call_train_api()
        if error:
            st.markdown(f"""
            <div class='alert-box alert-critical'>
                <div style='font-size:0.85rem;color:#e8eaf0;'>{error}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-box alert-success'>
                <div style='font-size:0.85rem;color:#e8eaf0;'>
                ✅ {result.get('status', 'Training completed successfully')}
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════

elif "ℹ️" in page:

    st.markdown("""
    <h1 style='font-size:2rem;margin-bottom:4px;'>
        About <span style='color:#00d4ff;'>This System</span>
    </h1>
    <p style='color:#8892a4;font-size:0.9rem;margin-bottom:28px;'>
        Bosch Hydraulic Predictive Maintenance Project
    </p>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("<div class='section-header'>Project Overview</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card' style='line-height:1.8;font-size:0.88rem;color:#8892a4;'>
            This system predicts the <span style='color:#e8eaf0;'>Remaining Useful Life (RUL)</span>
            of hydraulic pump units using real-time sensor telemetry data.<br><br>
            The pipeline ingests pressure, temperature, flow, vibration, and RPM
            sensor readings, engineers domain-specific features, and applies a
            <span style='color:#00d4ff;'>Gradient Boosting Regressor</span>
            to predict how many hours remain before maintenance is required.<br><br>
            Experiment tracking is handled via
            <span style='color:#ff6b35;'>MLflow + DagsHub</span>,
            model artifacts are stored in
            <span style='color:#ff6b35;'>AWS S3</span>, and
            predictions are served through a
            <span style='color:#ff6b35;'>FastAPI</span> backend.
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("<div class='section-header'>Tech Stack</div>",
                    unsafe_allow_html=True)
        tech_stack = [
            ("🐍", "Python 3.x",        "Core language"),
            ("🌲", "Scikit-learn",       "ML modelling"),
            ("⚡", "XGBoost",           "Gradient boosting"),
            ("📊", "MLflow + DagsHub",  "Experiment tracking"),
            ("☁️", "AWS S3",            "Artifact storage"),
            ("🚀", "FastAPI",           "Prediction API"),
            ("🎯", "Streamlit",         "This dashboard"),
        ]
        st.markdown("""<div class='metric-card'>
        <table class='styled-table'>
        <tr><th>Tool</th><th>Purpose</th></tr>""" +
        "".join([f"<tr><td>{icon} {tool}</td><td style='color:#8892a4'>{desc}</td></tr>"
                 for icon, tool, desc in tech_stack]) +
        "</table></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Dataset Summary</div>",
                unsafe_allow_html=True)

    ds1, ds2, ds3, ds4 = st.columns(4)
    dataset_stats = [
        ("864,000",  "Total Sensor Rows"),
        ("126,720",  "Anomaly Rows (RUL Data)"),
        ("9",        "HPU Machines"),
        ("36",       "Engineered Features"),
    ]
    for col, (val, label) in zip([ds1, ds2, ds3, ds4], dataset_stats):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div class='metric-value' style='font-size:1.6rem;'>{val}</div>
                <div class='metric-label' style='margin-top:8px;'>{label}</div>
            </div>""", unsafe_allow_html=True)
