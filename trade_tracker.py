import os
import json
import yaml
import pandas as pd
from datetime import datetime
from algo_layer import HierarchicalMTFDSFusion

class TradeTracker:
    def __init__(self,
                 state_dir: str = "state",
                 data_file: str = "data/XAU_USD_prices.csv"):
        self.state_dir = state_dir
        self.data_file = data_file
        self.strategy_file = os.path.join(state_dir, "strategy.yaml")
        self.trades_file = os.path.join(state_dir, "trades.jsonl")
        self.active_trade_file = os.path.join(state_dir, "active_trade.json")
        os.makedirs(self.state_dir, exist_ok=True)

        self._load_strategy()
        # 初始化多时间帧融合决策引擎
        self.engine = HierarchicalMTFDSFusion(config_path=self.strategy_file)

    def _load_strategy(self):
        if not os.path.exists(self.strategy_file):
            raise FileNotFoundError(f"未找到策略文件: {self.strategy_file}")
        with open(self.strategy_file, "r", encoding="utf-8") as f:
            self.strategy = yaml.safe_load(f)

    def update_and_track(self):
        """执行多时间帧推理，处理持仓止损止盈结算，或建立新虚拟订单"""
        # 1. 基础数据校验
        if not os.path.exists(self.data_file):
            print("行情文件不存在，跳过本次追踪。")
            return

        df_1m = pd.read_csv(self.data_file)
        if len(df_1m) < 60:
            print("1分钟行情数据不足 60 根，无法支持多时间帧重采样，跳过本次追踪。")
            return

        # 2. 运行分层 D-S 多时间帧证据融合模型
        res = self.engine.evaluate_global_decision(df_1m)
        current_time = str(df_1m['Date'].iloc[-1])
        current_price = float(res['price'])
        signal = res['decision']

        # 3. 检查是否存在未平仓活跃单
        if os.path.exists(self.active_trade_file):
            with open(self.active_trade_file, "r", encoding="utf-8") as f:
                active = json.load(f)

            direction = active['direction']
            entry_price = float(active['entry_price'])
            sl = float(active['sl'])
            tp = float(active['tp'])

            is_closed = False
            outcome = None
            exit_price = current_price

            # 评估买单止盈止损
            if direction == "BUY":
                if current_price <= sl:
                    is_closed = True
                    outcome = "LOSS"
                    exit_price = sl
                elif current_price >= tp:
                    is_closed = True
                    outcome = "WIN"
                    exit_price = tp

            # 评估卖单止盈止损
            elif direction == "SELL":
                if current_price >= sl:
                    is_closed = True
                    outcome = "LOSS"
                    exit_price = sl
                elif current_price <= tp:
                    is_closed = True
                    outcome = "WIN"
                    exit_price = tp

            if is_closed:
                pnl = round(exit_price - entry_price if direction == "BUY" else entry_price - exit_price, 2)
                record = {
                    "trade_id": active['trade_id'],
                    "strategy_version": active.get('strategy_version', 'v02'),
                    "direction": direction,
                    "entry_time": active['entry_time'],
                    "entry_price": entry_price,
                    "exit_time": current_time,
                    "exit_price": exit_price,
                    "outcome": outcome,
                    "pnl": pnl,
                    "sl": sl,
                    "tp": tp,
                    "entry_snapshot": active.get('entry_snapshot', {}),
                    "strategy_params": active.get('strategy_params', {})
                }

                # 写入历史账本
                with open(self.trades_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                os.remove(self.active_trade_file)
                print(f"[{current_time}] 订单结算: {direction} | 结果: {outcome} | PnL: ${pnl:.2f} | 离场价: ${exit_price:.2f}")

            else:
                print(f"[{current_time}] 持仓中: {direction} | 当前价: ${current_price:.2f} | 止盈: ${tp:.2f} | 止损: ${sl:.2f}")

        # 4. 无活跃持仓时，根据融合信号执行虚拟开仓
        else:
            if signal in ["BUY", "SELL"]:
                trade_id = f"T_{int(datetime.now().timestamp())}"
                trade_data = {
                    "trade_id": trade_id,
                    "strategy_version": self.strategy.get('version', 'v02'),
                    "direction": signal,
                    "entry_time": current_time,
                    "entry_price": current_price,
                    "sl": float(res['sl']),
                    "tp": float(res['tp']),
                    # 记录 6 帧 D-S 证据融合特征快照供后续 AI 反思优化
                    "entry_snapshot": {
                        "m_final": res['m_final'],
                        "psi_global": res['psi_global'],
                        "macro_veto": res['macro_veto'],
                        "tf_beliefs": res['tf_beliefs']
                    },
                    "strategy_params": self.strategy
                }

                with open(self.active_trade_file, "w", encoding="utf-8") as f:
                    json.dump(trade_data, f, indent=4)

                print(f"[{current_time}] 触发开仓: {signal} | 进场价: ${current_price:.2f} | 止损: ${res['sl']} | 止盈: ${res['tp']}")
            else:
                psi = res.get('psi_global', 0.0)
                reason = res.get('reason', '')
                print(f"[{current_time}] 维持观望 (HOLD) | 全局分歧度: {psi:.3f} | 原因: {reason}")

if __name__ == '__main__':
    tracker = TradeTracker()
    tracker.update_and_track()