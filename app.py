import os
import json
import yaml
import urllib.parse
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from streamlit_autorefresh import st_autorefresh

from algo_layer import HierarchicalMTFDSFusion

# Auto-refresh UI state every 30 seconds
st_autorefresh(interval=30 * 1000, key="data_refresher")

# 1. Page Configuration
st.set_page_config(
    page_title="ALPHA-D | Institutional XAU/USD Intelligence Terminal",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

load_dotenv()

# =============================================================================
# Custom Institutional Dark Mode CSS
# =============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0c0d10;
        color: #e1e3e6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    header[data-testid="stHeader"] { background-color: transparent; }
    footer { visibility: hidden; }

    .dashboard-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .dashboard-subtitle {
        font-size: 12px;
        color: #787b86;
        margin-bottom: 16px;
    }

    div[data-testid="stMetric"] {
        background-color: #131722;
        border: 1px solid #1e222d;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetricLabel"] {
        color: #787b86 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #1e222d;
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #787b86;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #2962ff !important;
        border-bottom: 2px solid #2962ff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #08090b;
        border-right: 1px solid #1a1d24;
    }

    .stChatMessage {
        background-color: #131722 !important;
        border: 1px solid #1e222d;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb { background: #2a2e39; border-radius: 2px; }
    ::-webkit-scrollbar-track { background: #0c0d10; }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_yaml(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def load_jsonl(filepath: str) -> list:
    if os.path.exists(filepath):
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records
    return []

# Sidebar configurations
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    strategy_cfg = load_yaml("state/strategy.yaml")
    goal_cfg = load_yaml("state/goal.yaml")

    current_version = strategy_cfg.get("version", "02")
    st.markdown("**Asset**: `XAU/USD (Spot Gold)`")
    st.markdown("**Fusion Engine**: `Hierarchical MTF D-S`")
    st.markdown(f"**Strategy Version**: `{current_version}`")
    st.markdown(f"**Target Win Rate**: `{goal_cfg.get('target_win_rate', 0.65) * 100}%`")

    st.divider()
    st.markdown("### 🧬 Core Hyperparameters")
    if "decision_rules" in strategy_cfg:
        st.json(strategy_cfg["decision_rules"])
    if "macro_veto" in strategy_cfg:
        st.json(strategy_cfg["macro_veto"])

# Core inference calculation
df_1m_path = "data/XAU_USD_prices.csv"
if not os.path.exists(df_1m_path):
    st.error("Error: Underlying data file data/XAU_USD_prices.csv not found. Please start data_agent.py.")
    st.stop()

df_1m = pd.read_csv(df_1m_path)
fusion_engine = HierarchicalMTFDSFusion(config_path="state/strategy.yaml")
res = fusion_engine.evaluate_global_decision(df_1m)

latest_price = res["price"]
decision = res["decision"]
p_buy = res["m_final"]["BUY"]
p_sell = res["m_final"]["SELL"]
p_hold = res["m_final"]["HOLD"]
psi = res["psi_global"]
macro_veto = res["macro_veto"]

# Top Header & Metrics
st.markdown('<div class="dashboard-title">⚡ ALPHA-D // Quantitative Intelligence Terminal</div>', unsafe_allow_html=True)
st.markdown(f'<div class="dashboard-subtitle">Last Synchronized: {df_1m["Date"].iloc[-1]} | Attribution: {res["reason"]}</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spot Price", f"${latest_price:.2f}")
decision_display = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}
col2.metric("Consensus Action", decision_display.get(decision, decision))
col3.metric("Belief Probability", f"B: {p_buy:.2f} | S: {p_sell:.2f}")
col4.metric("Conflict Index (Ψ)", f"{psi:.3f}", delta="Optimal" if psi <= fusion_engine.psi_max else "High Conflict", delta_color="inverse")
col5.metric("Veto Factor", f"B: {macro_veto['P_BUY']} | S: {macro_veto['P_SELL']}")

st.divider()

# Main Container Layout
col_main, col_chat = st.columns([3.8, 1.8])

with col_main:
    tab_chart, tab_mtf, tab_trades, tab_evolution = st.tabs([
        "📈 TradingView Terminal", "🧠 MTF Belief Distribution", "📋 Virtual Ledger", "🧬 Strategy Evolution"
    ])

    with tab_chart:
        tv_component_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body { background-color: #0c0d10; height: 100vh; overflow: hidden; }
                #tv_chart_container { height: 1200px; width: 100%; }
            </style>
        </head>
        <body>
            <div id="tv_chart_container"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({
                    "autosize": true,
                    "symbol": "OANDA:XAUUSD",
                    "interval": "5",
                    "timezone": "Asia/Kuala_Lumpur",
                    "theme": "dark",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#0c0d10",
                    "enable_publishing": false,
                    "allow_symbol_change": true,
                    "hide_side_toolbar": false,
                    "container_id": "tv_chart_container",
                    "studies": [
                        "RSI@tv-basicstudies",
                        "CCI@tv-basicstudies",
                        "MACD@tv-basicstudies",
                        "Stochastic@tv-basicstudies",
                        "Momentum@tv-basicstudies"
                    ]
                });
            </script>
        </body>
        </html>
        """
        encoded_html = urllib.parse.quote(tv_component_html)
        st.components.v1.iframe(src=f"data:text/html;charset=utf-8,{encoded_html}", height=1200, scrolling=False)

    with tab_mtf:
        st.markdown("#### Intra-Timeframe Belief Allocation Vector $M_\\tau = [m(BUY), m(SELL), m(HOLD)]$")
        tf_beliefs = res["tf_beliefs"]
        tf_df = pd.DataFrame(tf_beliefs, index=["m(BUY)", "m(SELL)", "m(HOLD)"]).T
        st.dataframe(tf_df.style.highlight_max(axis=1, color="#1e3a8a"), width="stretch")

        st.markdown("#### Macro Timeframe Veto Gate (1H / M30)")
        st.json({
            "1H Signals": tf_beliefs.get("1H", []),
            "M30 Signals": tf_beliefs.get("M30", []),
            "P_BUY (Penalty Multiplier)": macro_veto["P_BUY"],
            "P_SELL (Penalty Multiplier)": macro_veto["P_SELL"]
        })

    with tab_trades:
        trades = load_jsonl("state/trades.jsonl")
        active_trade_file = "state/active_trade.json"

        if os.path.exists(active_trade_file):
            with open(active_trade_file, "r") as f:
                active = json.load(f)
            st.info(f"🔵 **Active Order**: {active['direction']} | Entry: ${active['entry_price']} | SL: ${active['sl']} | TP: ${active['tp']}")

        if trades:
            t_df = pd.DataFrame(trades)
            wins = (t_df['outcome'] == 'WIN').sum()
            losses = (t_df['outcome'] == 'LOSS').sum()
            wr = (wins / len(t_df)) * 100
            total_pnl = t_df['pnl'].sum()

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Trades", len(t_df))
            m_col2.metric("Win Rate", f"{wr:.1f}% ({wins}W / {losses}L)")
            m_col3.metric("Net Cumulative P&L", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")

            st.dataframe(t_df[['entry_time', 'direction', 'entry_price', 'exit_price', 'outcome', 'pnl', 'strategy_version']].tail(15), width="stretch")
        else:
            st.write("No closed trade records found.")

    with tab_evolution:
        st.markdown("#### Autonomous Scientific Hypotheses & Mutations")
        hypotheses = load_jsonl("state/hypotheses.jsonl")
        if hypotheses:
            for item in reversed(hypotheses):
                with st.expander(f"Mutation: {item.get('from_version')} ➔ {item.get('to_version')} ({item.get('timestamp')})"):
                    st.write(f"**Attribution Diagnosis**: {item.get('diagnosis')}")
                    st.write(f"**Parameter Shift**: `{item.get('mutation')}`")
                    st.write(f"**Empirical Hypothesis**: {item.get('hypothesis')}")
        else:
            st.write("System operating under baseline version. Evolution has not triggered yet.")

# Right Panel: Quant Copilot Terminal
with col_chat:
    st.markdown("### 💬 Quant Copilot")
    st.caption("Context-aware decision assistant backed by D-S evidence")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Quant Copilot initialized. Cross-timeframe evidence distribution loaded. Ask any question regarding market regime or parameters."}
        ]

    prompt = st.chat_input("Ask Copilot (e.g. Why is the system recommending HOLD?)")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        system_context = f"""
You are an institutional quantitative trading researcher specializing in XAU/USD.
Current engine: Hierarchical Multi-Timeframe Dempster-Shafer (D-S) Evidence Fusion Model.

Latest Engine State:
- Price: ${latest_price:.2f}
- Decision: {decision} (Attribution: {res['reason']})
- Belief Scores: BUY={p_buy:.3f}, SELL={p_sell:.3f}, HOLD={p_hold:.3f}
- Conflict Metric (Psi_global): {psi:.3f} (Max threshold: {fusion_engine.psi_max})
- Stop-Loss (SL): ${res['sl']} | Take-Profit (TP): ${res['tp']}

User Inquiry: {prompt}

Provide a concise, quantitative, and professional response.
"""
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            st.session_state.chat_history.append({"role": "assistant", "content": "Error: GEMINI_API_KEY not configured in environment."})
        else:
            try:
                client = genai.Client(api_key=gemini_key)
                chat_session = client.chats.create(model="gemini-2.0-flash")
                response = chat_session.send_message(system_context)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Inference Error: {str(e)}"})

    chat_box = st.container(height=660)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])