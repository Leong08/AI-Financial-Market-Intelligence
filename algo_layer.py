import os
import math
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class HierarchicalMTFDSFusion:
    def __init__(self, config_path: str = "state/strategy.yaml"):
        self.config_path = config_path
        self._load_config()

        # 时间帧映射
        self.timeframes = ['M5', 'M1']
        self.tf_resample_rules = {
            'M5': '5min',
            'M1': '1min'
        }

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        self.k_rsi = cfg.get('fuzzification', {}).get('k_rsi', 0.15)
        self.k_cci = cfg.get('fuzzification', {}).get('k_cci', 0.02)
        self.w_base = cfg.get('timeframe_weights', {
            '1H': 0.35, 'M30': 0.25, 'M15': 0.15, 'M10': 0.10, 'M5': 0.10, 'M1': 0.05
        })
        self.voter_weights = np.array(list(cfg.get('voter_weights', {
            'rsi': 0.2, 'cci': 0.2, 'stoch': 0.2, 'macd': 0.2, 'mom': 0.2
        }).values()))

        rules = cfg.get('decision_rules', {})
        self.delta_signal = rules.get('delta_signal', 0.55)
        self.delta_margin = rules.get('delta_margin', 0.15)
        self.psi_max = rules.get('psi_max', 0.60)

        veto = cfg.get('macro_veto', {})
        self.veto_trigger = veto.get('veto_trigger_threshold', 0.50)
        self.veto_pass = veto.get('veto_pass_threshold', 0.60)

    # =========================================================================
    # 多时间帧数据本地重采样引擎
    # =========================================================================
    def resample_timeframes(self, df_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """将 1 分钟连续数据在本地重采样为 6 个时间帧"""
        dfs = {}
        df_work = df_1m.copy()
        df_work['Date'] = pd.to_datetime(df_work['Date'])
        df_work.set_index('Date', inplace=True)

        for tf, rule in self.tf_resample_rules.items():
            if rule == '1min':
                res = df_work.copy()
            else:
                res = df_work.resample(rule, label='right', closed='right').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            # 补充指标计算
            res = self._compute_indicators_for_tf(res)
            dfs[tf] = res

        return dfs

    def _compute_indicators_for_tf(self, df: pd.DataFrame) -> pd.DataFrame:
        """为特定时间帧计算指标"""
        # 1. RSI(14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. CCI(20)
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        sma_tp = tp.rolling(20).mean()
        mad = (tp - sma_tp).abs().rolling(20).mean()
        df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-9)

        # 3. Stochastic Oscillator (14, 3)
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        fast_k = 100 * ((df['Close'] - low14) / (high14 - low14 + 1e-9))
        df['Slow_K'] = fast_k.rolling(3).mean()
        df['Slow_D'] = df['Slow_K'].rolling(3).mean()

        # 4. MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd - signal
        df['sigma_macd'] = df['MACD_Hist'].rolling(20).std().replace(0, 1e-4)

        # 5. Momentum (10)
        df['Mom'] = 100 * (df['Close'] / df['Close'].shift(10).replace(0, np.nan))
        df['sigma_mom'] = df['Mom'].rolling(20).std().replace(0, 1e-4)

        # 6. Volume & RVol
        df['sma_vol20'] = df['Volume'].rolling(20).mean().replace(0, 1e-4)
        df['RVol'] = df['Volume'] / df['sma_vol20']

        # 7. ATR(14)
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        df['ATR14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        return df.bfill().ffill()

    # =========================================================================
    # Tier 1: 单时间帧选民模糊化与单帧 D-S 融合
    # =========================================================================
    def fuzzify_voters_per_tf(self, row: pd.Series) -> List[np.ndarray]:
        """按数学模型计算 5 个选民在当前时间帧的 BBA [BUY, SELL, HOLD]"""
        voters = []

        # 2.1 RSI Voter
        rsi = row['RSI']
        m1_sell = 1.0 / (1.0 + np.exp(np.clip(-self.k_rsi * (rsi - 70.0), -50, 50)))
        m1_buy = 1.0 / (1.0 + np.exp(np.clip(self.k_rsi * (rsi - 30.0), -50, 50)))
        m1_hold = max(0.0, 1.0 - m1_buy - m1_sell)
        tot1 = m1_buy + m1_sell + m1_hold
        voters.append(np.array([m1_buy/tot1, m1_sell/tot1, m1_hold/tot1]))

        # 2.2 CCI Voter
        cci = row['CCI']
        m2_sell = 1.0 / (1.0 + np.exp(np.clip(-self.k_cci * (cci - 100.0), -50, 50)))
        m2_buy = 1.0 / (1.0 + np.exp(np.clip(self.k_cci * (cci + 100.0), -50, 50)))
        m2_hold = max(0.0, 1.0 - m2_buy - m2_sell)
        tot2 = m2_buy + m2_sell + m2_hold
        voters.append(np.array([m2_buy/tot2, m2_sell/tot2, m2_hold/tot2]))

        # 2.3 Stochastic Oscillator Voter
        delta_stoch = row['Slow_K'] - row['Slow_D']
        s_bar = (row['Slow_K'] + row['Slow_D']) / 2.0
        m3_buy = (1.0 / (1.0 + np.exp(-0.1 * delta_stoch))) * (1.0 / (1.0 + np.exp(0.08 * (s_bar - 20.0))))
        m3_sell = (1.0 / (1.0 + np.exp(0.1 * delta_stoch))) * (1.0 / (1.0 + np.exp(-0.08 * (s_bar - 80.0))))
        m3_hold = max(0.0, 1.0 - m3_buy - m3_sell)
        tot3 = m3_buy + m3_sell + m3_hold
        voters.append(np.array([m3_buy/tot3, m3_sell/tot3, m3_hold/tot3]))

        # 2.4 MACD Histogram Voter
        h_hat = row['MACD_Hist'] / (row['sigma_macd'] + 1e-9)
        th = math.tanh(h_hat)
        m4_buy = max(0.0, th)
        m4_sell = max(0.0, -th)
        m4_hold = max(0.0, 1.0 - abs(th))
        voters.append(np.array([m4_buy, m4_sell, m4_hold]))

        # 2.5 Momentum Voter
        delta_mom = (row['Mom'] - 100.0) / (row['sigma_mom'] + 1e-9)
        m5_buy = (1.0 / (1.0 + np.exp(-2.0 * delta_mom))) if delta_mom > 0 else 0.0
        m5_sell = (1.0 / (1.0 + np.exp(2.0 * delta_mom))) if delta_mom < 0 else 0.0
        m5_hold = max(0.0, 1.0 - m5_buy - m5_sell)
        tot5 = m5_buy + m5_sell + m5_hold
        voters.append(np.array([m5_buy/tot5, m5_sell/tot5, m5_hold/tot5]))

        # 3. Volume-RVol Dynamic Multiplier
        rvol = row['RVol']
        vol = row['Volume']
        sma_vol = row['sma_vol20']
        gamma = min(2.0, max(0.5, rvol * math.log10(1.0 + vol / (sma_vol + 1e-9))))

        # 修正方向性信念
        discounted_voters = []
        for v in voters:
            b = np.clip(v[0] * gamma, 0.0, 1.0)
            s = np.clip(v[1] * gamma, 0.0, 1.0)
            h = max(0.0, 1.0 - b - s)
            tot = b + s + h
            discounted_voters.append(np.array([b/tot, s/tot, h/tot]))

        return discounted_voters

    @staticmethod
    def _dempster_combine(m_a: np.ndarray, m_b: np.ndarray) -> np.ndarray:
        """D-S 正交递推"""
        b_a, s_a, h_a = m_a[0], m_a[1], m_a[2]
        b_b, s_b, h_b = m_b[0], m_b[1], m_b[2]

        k = b_a * s_b + s_a * b_b
        if k >= 0.9999:
            return np.array([0.0, 0.0, 1.0])

        b_fused = (b_a * b_b + b_a * h_b + h_a * b_b) / (1.0 - k)
        s_fused = (s_a * s_b + s_a * h_b + h_a * s_b) / (1.0 - k)
        h_fused = (h_a * h_b) / (1.0 - k)

        vec = np.clip(np.array([b_fused, s_fused, h_fused]), 0.0, 1.0)
        return vec / vec.sum()

    def fuse_single_timeframe(self, row: pd.Series) -> np.ndarray:
        """第 4 节：单帧内 D-S 证据融合，输出 M_tau = [m(BUY), m(SELL), m(HOLD)]"""
        voters = self.fuzzify_voters_per_tf(row)

        # 4.1 绩效权重折现
        discounted = []
        for i in range(5):
            alpha = self.voter_weights[i]
            b = voters[i][0] * alpha
            s = voters[i][1] * alpha
            h = max(0.0, 1.0 - b - s)
            discounted.append(np.array([b, s, h]))

        # 4.3 递推组合
        accum = discounted[0]
        for i in range(1, 5):
            accum = self._dempster_combine(accum, discounted[i])

        return accum

    # =========================================================================
    # Tier 2: 跨时间帧分层融合、宏观大趋势否决与决策引擎
    # =========================================================================
    def evaluate_global_decision(self, df_1m: pd.DataFrame) -> Dict:
        """主评估流程：执行全部 6 帧分析、宏观否决与最终信号生成"""
        tf_dfs = self.resample_timeframes(df_1m)

        # 获取各时间帧最新截面的单帧融合向量 M_tau
        m_tf = {}
        for tf in self.timeframes:
            latest_row = tf_dfs[tf].iloc[-1]
            m_tf[tf] = self.fuse_single_timeframe(latest_row)

        # 5.2 基于信息确定性的动态权重修正 Omega_tau
        certainty = {tf: 1.0 - m_tf[tf][2] for tf in self.timeframes}  # eta = 1 - m(HOLD)
        weighted_certainty = {tf: self.w_base[tf] * certainty[tf] for tf in self.timeframes}
        sum_wc = sum(weighted_certainty.values()) + 1e-9
        omega = {tf: weighted_certainty[tf] / sum_wc for tf in self.timeframes}

        # 5.3 线性多时间帧合成
        m_hat_buy = sum(omega[tf] * m_tf[tf][0] for tf in self.timeframes)
        m_hat_sell = sum(omega[tf] * m_tf[tf][1] for tf in self.timeframes)

        # 5.4 大趋势一票否决机制 (1H 与 M30)
        macro_sell = max(m_tf['M5'][1], m_tf['M15'][1])
        macro_buy = max(m_tf['M5'][0], m_tf['M15'][0])

        p_buy_penalty = (1.0 - macro_sell) if macro_sell > self.veto_trigger else 1.0
        p_sell_penalty = (1.0 - macro_buy) if macro_buy > self.veto_trigger else 1.0

        m_final_buy = m_hat_buy * p_buy_penalty
        m_final_sell = m_hat_sell * p_sell_penalty
        m_final_hold = max(0.0, 1.0 - m_final_buy - m_final_sell)

        # 6.1 全局分歧度量化 Psi_global
        diff_sq = sum((m_tf[tf][0] - m_tf[tf][1]) ** 2 for tf in self.timeframes)
        psi_global = math.sqrt(diff_sq / len(self.timeframes))

        # 6.2 最终开仓逻辑判断
        decision = "HOLD"
        reason = "信号未满足多时间帧共振条件"

        buy_cond = (
            (m_final_buy >= self.delta_signal) and
            ((m_final_buy - m_final_sell) >= self.delta_margin) and
            (p_buy_penalty > self.veto_pass) and
            (psi_global <= self.psi_max)
        )

        sell_cond = (
            (m_final_sell >= self.delta_signal) and
            ((m_final_sell - m_final_buy) >= self.delta_margin) and
            (p_sell_penalty > self.veto_pass) and
            (psi_global <= self.psi_max)
        )

        if buy_cond:
            decision = "BUY"
            reason = f"多时间帧看涨共振 (Score: {m_final_buy:.2f}, 逆势惩罚乘数: {p_buy_penalty:.2f})"
        elif sell_cond:
            decision = "SELL"
            reason = f"多时间帧看跌共振 (Score: {m_final_sell:.2f}, 逆势惩罚乘数: {p_sell_penalty:.2f})"
        else:
            if psi_global > self.psi_max:
                reason = f"全局跨时间帧分歧过大避险 (Psi={psi_global:.2f} > {self.psi_max})"
            elif p_buy_penalty <= self.veto_pass or p_sell_penalty <= self.veto_pass:
                reason = "大时间帧(1H/M30)反向趋势一票否决拦截"

        # 提取基准价格与 ATR 风控目标
        p_current = tf_dfs['M1']['Close'].iloc[-1]
        atr_1m = tf_dfs['M1']['ATR14'].iloc[-1]
        sl = round(p_current - 1.8 * atr_1m if decision == "BUY" else p_current + 1.8 * atr_1m, 2)
        tp = round(p_current + 3.2 * atr_1m if decision == "BUY" else p_current - 3.2 * atr_1m, 2)

        return {
            "decision": decision,
            "reason": reason,
            "price": p_current,
            "sl": sl,
            "tp": tp,
            "m_final": {"BUY": round(m_final_buy, 4), "SELL": round(m_final_sell, 4), "HOLD": round(m_final_hold, 4)},
            "macro_veto": {"P_BUY": round(p_buy_penalty, 3), "P_SELL": round(p_sell_penalty, 3)},
            "psi_global": round(psi_global, 4),
            "tf_beliefs": {tf: [round(x, 3) for x in m_tf[tf]] for tf in self.timeframes}
        }

if __name__ == "__main__":
    import pandas as pd
    file = "data/XAU_USD_prices.csv"
    if os.path.exists(file):
        df_1m = pd.read_csv(file)
        model = HierarchicalMTFDSFusion()
        res = model.evaluate_global_decision(df_1m)
        print("\n=== 分层多时间帧 D-S 融合决策评估结果 ===")
        print(f"标的价格: ${res['price']:.2f}")
        print(f"最终决策: 【{res['decision']}】 -> {res['reason']}")
        print(f"最终信念得分: {res['m_final']}")
        print(f"跨时间帧全局分歧度 (Psi): {res['psi_global']}")
        print(f"各时间帧输出 [BUY, SELL, HOLD]: {res['tf_beliefs']}")
        print(f"大趋势否决惩罚乘数: {res['macro_veto']}")
        if res['decision'] != "HOLD":
            print(f"建议止损(SL): ${res['sl']} | 止盈(TP): ${res['tp']}")