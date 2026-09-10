import os
import json
import yaml
from datetime import datetime
from self_improve import StrategySelfImprove


def setup_mock_test_environment():
    print("\n" + "=" * 80)
    print("🧪 [测试启动] 开始构建 AI 自动进化触发测试环境...")
    print("=" * 80)

    state_dir = "state"
    trades_file = os.path.join(state_dir, "trades.jsonl")
    strategy_file = os.path.join(state_dir, "strategy.yaml")

    # 1. 检查基础环境
    if not os.path.exists(strategy_file):
        print("❌ 未检测到 state/strategy.yaml，请确认配置文件存在！")
        return

    with open(strategy_file, "r", encoding="utf-8") as f:
        strat = yaml.safe_load(f)
    print(f"当前策略版本: {strat.get('version', 'unknown')}")

    # 2. 模拟构造 5 笔交易数据 (4 负 1 胜，强制制造亏损归因场景)
    print("📝 正在注入 5 笔模拟历史订单数据...")
    mock_trades = [
        {
            "trade_id": "MOCK_T01",
            "strategy_version": strat.get('version', 'v02'),
            "direction": "BUY",
            "entry_time": "2026-09-10 10:00:00",
            "entry_price": 4390.0,
            "exit_time": "2026-09-10 10:15:00",
            "exit_price": 4385.0,
            "outcome": "LOSS",
            "pnl": -5.0,
            "entry_snapshot": {"psi_global": 0.68, "m_final": {"BUY": 0.46, "SELL": 0.35, "HOLD": 0.19}}
        },
        {
            "trade_id": "MOCK_T02",
            "strategy_version": strat.get('version', 'v02'),
            "direction": "SELL",
            "entry_time": "2026-09-10 11:00:00",
            "entry_price": 4395.0,
            "exit_time": "2026-09-10 11:30:00",
            "exit_price": 4402.0,
            "outcome": "LOSS",
            "pnl": -7.0,
            "entry_snapshot": {"psi_global": 0.72, "m_final": {"BUY": 0.20, "SELL": 0.48, "HOLD": 0.32}}
        },
        {
            "trade_id": "MOCK_T03",
            "strategy_version": strat.get('version', 'v02'),
            "direction": "BUY",
            "entry_time": "2026-09-10 12:00:00",
            "entry_price": 4388.0,
            "exit_time": "2026-09-10 12:20:00",
            "exit_price": 4398.0,
            "outcome": "WIN",
            "pnl": 10.0,
            "entry_snapshot": {"psi_global": 0.35, "m_final": {"BUY": 0.65, "SELL": 0.10, "HOLD": 0.25}}
        },
        {
            "trade_id": "MOCK_T04",
            "strategy_version": strat.get('version', 'v02'),
            "direction": "BUY",
            "entry_time": "2026-09-10 13:00:00",
            "entry_price": 4400.0,
            "exit_time": "2026-09-10 13:10:00",
            "exit_price": 4393.0,
            "outcome": "LOSS",
            "pnl": -7.0,
            "entry_snapshot": {"psi_global": 0.65, "m_final": {"BUY": 0.47, "SELL": 0.38, "HOLD": 0.15}}
        },
        {
            "trade_id": "MOCK_T05",
            "strategy_version": strat.get('version', 'v02'),
            "direction": "SELL",
            "entry_time": "2026-09-10 14:00:00",
            "entry_price": 4392.0,
            "exit_time": "2026-09-10 14:40:00",
            "exit_price": 4399.0,
            "outcome": "LOSS",
            "pnl": -7.0,
            "entry_snapshot": {"psi_global": 0.70, "m_final": {"BUY": 0.25, "SELL": 0.46, "HOLD": 0.29}}
        }
    ]

    with open(trades_file, "a", encoding="utf-8") as f:
        for t in mock_trades:
            f.write(json.dumps(t) + "\n")

    print(f"✅ 成功追加 5 笔测试订单！")

    # 3. 实例化并执行进化评估
    improver = StrategySelfImprove()
    print("🔍 检查是否达到进化复盘节奏点...")

    if improver.should_reflect():
        print("🎯 触发条件满足！正在唤醒 Gemini 2.5 进行科学复盘与单变量参数突变...")
        improver.run_reflection()

        # 4. 读取最新版本以确认结果
        with open(strategy_file, "r", encoding="utf-8") as f:
            new_strat = yaml.safe_load(f)
        print("=" * 80)
        print(f"🎉 [测试完成] 策略已成功自我进化至: 【{new_strat.get('version')}】")
        print("=" * 80)
    else:
        print("⚠️ 未能触发复盘，可能 trades 数量与 cadence 计算不符，强制执行反思...")
        improver.run_reflection()


if __name__ == "__main__":
    setup_mock_test_environment()