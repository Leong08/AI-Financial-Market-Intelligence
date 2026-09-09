import os
import json
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from google import genai
from streamlit_autorefresh import st_autorefresh

from algo_layer import HierarchicalMTFDSFusion

st_autorefresh(interval=10 * 1000, key = "data_refresher")

# 页面基础配置
st.set_page_config(
    page_title="XAU/USD AI Market Intelligence Dashboard",
    layout="wide",
    page_icon="📈"
)

load_dotenv()


# =============================================================================
# 辅助函数：安全读取文件
# =============================================================================
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


# =============================================================================
# 侧边栏：系统状态与参数监控
# =============================================================================
st.sidebar.title("⚙️ 交易系统监控台")
strategy_cfg = load_yaml("state/strategy.yaml")
goal_cfg = load_yaml("state/goal.yaml")

current_version = strategy_cfg.get("version", "v01")
st.sidebar.markdown(f"**标的**: `XAU/USD (现货黄金)`")
st.sidebar.markdown(f"**当前策略版本**: `{current_version}`")
st.sidebar.markdown(f"**目标胜率**: `{goal_cfg.get('target_win_rate', 0.65) * 100}%`")

# 周期选择器
tf_display_map = {
    "1分钟 (M1)": "data/XAU_USD_prices.csv",
    "5分钟 (M5)": "data/XAU_USD_5min.csv",
    "15分钟 (M15)": "data/XAU_USD_15min.csv",
    "30分钟 (M30)": "data/XAU_USD_30min.csv",
    "1小时 (1H)": "data/XAU_USD_1h.csv"
}
selected_tf_label = st.sidebar.selectbox("选择图表时间周期", list(tf_display_map.keys()), index=0)
selected_file = tf_display_map[selected_tf_label]

st.sidebar.divider()
st.sidebar.markdown("### 🧬 当前核心超参数")
if "decision_rules" in strategy_cfg:
    st.sidebar.json(strategy_cfg["decision_rules"])

# =============================================================================
# 核心数据加载与 D-S 模型实时推理
# =============================================================================
df_1m_path = "data/XAU_USD_prices.csv"
if not os.path.exists(df_1m_path):
    st.error("❌ 未检测到底层数据文件 `data/XAU_USD_prices.csv`，请先启动 `python data_agent.py` 采集行情。")
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

# =============================================================================
# 顶部：核心指标看板 (KPI Metrics)
# =============================================================================
st.title("🏛️ XAU/USD 多时间帧证据融合交易决策看板")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("最新现价", f"${latest_price:.2f}")

decision_colors = {"BUY": "🟢 买入 (BUY)", "SELL": "🔴 卖出 (SELL)", "HOLD": "⚪ 观望 (HOLD)"}
col2.metric("当前综合决策", decision_colors.get(decision, decision))
col3.metric("买/卖信念概率", f"B: {p_buy:.2f} | S: {p_sell:.2f}")
col4.metric("全局分歧度 (Ψ)", f"{psi:.3f}", delta="正常" if psi <= fusion_engine.psi_max else "分歧过大避险",
            delta_color="inverse")
col5.metric("大趋势惩罚乘数", f"P_BUY: {macro_veto['P_BUY']} | P_SELL: {macro_veto['P_SELL']}")

st.caption(f"决策归因诊断: `{res['reason']}` | 评估时间: `{df_1m['Date'].iloc[-1]}`")
st.divider()

# =============================================================================
# 主视窗布局：左侧图表与分析，右侧 AI 助手
# =============================================================================
col_main, col_chat = st.columns([3, 2])

with col_main:
    tab_chart, tab_mtf, tab_trades, tab_evolution = st.tabs([
        "📊 交互式行情图表", "🧠 6帧证据信念分布", "📋 虚拟订单账本", "🧬 策略自进化轨迹"
    ])

    # Tab 1: K线图表
    with tab_chart:
        target_csv = selected_file if os.path.exists(selected_file) else df_1m_path
        chart_df = pd.read_csv(target_csv).tail(100)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # 主 K 线
        fig.add_trace(go.Candlestick(
            x=chart_df['Date'],
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name="K-Line"
        ), row=1, col=1)

        # 动态止损止盈参考线（非 HOLD 状态展示）
        if decision != "HOLD":
            fig.add_hline(y=res["sl"], line_dash="dot", line_color="red", annotation_text=f"建议止损: ${res['sl']}",
                          row=1, col=1)
            fig.add_hline(y=res["tp"], line_dash="dot", line_color="green", annotation_text=f"建议止盈: ${res['tp']}",
                          row=1, col=1)

        # 副图：成交量
        vol_colors = ['red' if c >= o else 'green' for c, o in zip(chart_df['Close'], chart_df['Open'])]
        fig.add_trace(go.Bar(
            x=chart_df['Date'],
            y=chart_df['Volume'],
            name="Volume",
            marker_color=vol_colors
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: 多时间帧信念矩阵
    with tab_mtf:
        st.markdown("#### 各独立时间帧信念分配向量 $M_\\tau = [m(BUY), m(SELL), m(HOLD)]$")
        tf_beliefs = res["tf_beliefs"]
        tf_df = pd.DataFrame(tf_beliefs, index=["m(BUY)", "m(SELL)", "m(HOLD)"]).T
        st.dataframe(tf_df.style.highlight_max(axis=1, color="#2e7bcf"), use_container_width=True)

        st.markdown("#### 宏观时间帧趋势一票否决状态 (1H / M30)")
        st.json({
            "1H 信号": tf_beliefs.get("1H", []),
            "M30 信号": tf_beliefs.get("M30", []),
            "P_BUY (多头惩罚乘数)": macro_veto["P_BUY"],
            "P_SELL (空头惩罚乘数)": macro_veto["P_SELL"]
        })

    # Tab 3: 虚拟订单账本统计
    with tab_trades:
        trades = load_jsonl("state/trades.jsonl")
        active_trade_file = "state/active_trade.json"

        if os.path.exists(active_trade_file):
            with open(active_trade_file, "r") as f:
                active = json.load(f)
            st.info(
                f"🔵 **当前活跃持仓**: {active['direction']} | 入场价: ${active['entry_price']} | 止损: ${active['sl']} | 止盈: ${active['tp']}")

        if trades:
            t_df = pd.DataFrame(trades)
            wins = (t_df['outcome'] == 'WIN').sum()
            losses = (t_df['outcome'] == 'LOSS').sum()
            wr = (wins / len(t_df)) * 100
            total_pnl = t_df['pnl'].sum()

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("累计订单数", len(t_df))
            m_col2.metric("当前胜率", f"{wr:.1f}% ({wins}胜/{losses}负)")
            m_col3.metric("累计净收益", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")

            st.dataframe(t_df[['entry_time', 'direction', 'entry_price', 'exit_price', 'outcome', 'pnl',
                               'strategy_version']].tail(10), use_container_width=True)
        else:
            st.write("暂无已平仓历史交易。")

    # Tab 4: 策略自进化历史
    with tab_evolution:
        st.markdown("#### AI 科学假设与参数自进化演变记录")
        hypotheses = load_jsonl("state/hypotheses.jsonl")
        if hypotheses:
            for item in reversed(hypotheses):
                with st.expander(
                        f"版本迭代: {item.get('from_version')} ➔ {item.get('to_version')} ({item.get('timestamp')})"):
                    st.write(f"**诊断归因**: {item.get('diagnosis')}")
                    st.write(f"**单变量参数调整**: `{item.get('mutation')}`")
                    st.write(f"**科学假设**: {item.get('hypothesis')}")
        else:
            st.write("系统处于初始策略版本，尚未触发周期性复盘自进化。")

# =============================================================================
# 右侧：基于 Gemini 的智能决策问答终端
# =============================================================================
with col_chat:
    st.subheader("💬 AI 市场情报研判助手")
    st.caption("基于当前分层 D-S 证据模型计算结果提供分析支撑（不构成自动化下单操作）")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "你好，我是你的 XAU/USD 量化决策助手。当前系统的多时间帧证据融合矩阵已加载，欢迎就当下的指标分歧度、宏观否决状态或止损止盈设置进行提问。"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("向 AI 提问当前行情或模型状态..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 构造上下文 Prompt
        system_context = f"""
你是一个专业的 XAU/USD（现货黄金）量化交易研判助手。
系统当前正在运行一套学术级【分层多时间帧 Dempster-Shafer 证据理论融合决策模型】。
模型当前实时计算状态如下：
- 当前黄金价格: ${latest_price:.2f}
- 最终融合决策: {decision} (归因: {res['reason']})
- 最终信念概率: BUY={p_buy:.3f}, SELL={p_sell:.3f}, HOLD={p_hold:.3f}
- 全局时间帧分歧度 (Psi_global): {psi:.3f} (阈值上限: {fusion_engine.psi_max})
- 宏观趋势否决惩罚乘数 (1H/M30): P_BUY={macro_veto['P_BUY']}, P_SELL={macro_veto['P_SELL']}
- 各时间帧具体信念分布: {res['tf_beliefs']}
- 建议止损位 (SL): ${res['sl']} | 建议止盈位 (TP): ${res['tp']}

用户的提问是: "{user_query}"

请依据上述客观量化数据，专业、严谨、客观地给出解答。不得编造任何未在上下文中出现的价格数据。
"""
        with st.chat_message("assistant"):
            try:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=system_context
                )
                bot_reply = response.text
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                err_msg = f"调用 AI 服务失败: {e}"
                st.error(err_msg)