import os
import pandas as pd
import numpy as np
from datetime import datetime


class TimeframeResampler:
    def __init__(self, source_file="data/XAU_USD_prices.csv", output_dir="data"):
        self.source_file = source_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Mapping table: target timeframe -> Pandas resampling rule
        self.timeframe_map = {
            "5min": "5min",
            "10min": "10min",
            "15min": "15min",
            "30min": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "1d": "1D"
        }

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pure Pandas/Numpy local high-precision calculation of indicators (aligned with Twelve Data)"""
        df = df.copy()

        # 1. RSI (14-period, Wilder’s smoothing consistent with TradingView/TwelveData)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = (100 - (100 / (1 + rs))).round(2).fillna(50.0)

        # 2. MACD (12, 26, 9)
        ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (ema_fast - ema_slow).round(4)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().round(4)
        df['macd_hist'] = (df['macd'] - df['macd_signal']).round(4)

        # 3. CCI (20-period)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
        df['cci'] = ((tp - sma_tp) / (0.015 * mad.replace(0, np.nan))).round(2).fillna(0.0)

        # 4. Stochastic Oscillator (14, 1, 3)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        fast_k = 100 * (df['Close'] - low_14) / (high_14 - low_14).replace(0, np.nan)
        df['slow_k'] = fast_k.rolling(window=1).mean().round(2).fillna(50.0)
        df['slow_d'] = df['slow_k'].rolling(window=3).mean().round(2).fillna(50.0)

        # 5. Momentum (9-period)
        df['mom'] = (df['Close'] - df['Close'].shift(9)).round(4).fillna(0.0)

        # 6. Relative Volume (14-period)
        avg_vol = df['Volume'].rolling(window=14).mean()
        df['rvol'] = (df['Volume'] / avg_vol.replace(0, 1)).round(4).fillna(1.0)

        return df

    def resample_single_timeframe(self, df_1min: pd.DataFrame, tf_name: str, rule: str) -> pd.DataFrame:
        """Aggregate 1-minute data into the specified timeframe OHLCV and recalculate indicators"""
        # Strict OHLCV aggregation rules
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        # label='left', closed='left' ensures alignment with the opening timestamp of the candle
        df_resampled = df_1min.resample(rule, label='left', closed='left').agg(agg_rules).dropna()

        # Recalculate indicators on the new resampled candles
        df_final = self._calculate_indicators(df_resampled)
        df_final.reset_index(inplace=True)
        df_final['Date'] = df_final['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df_final

    def generate_all_timeframes(self, quiet=False):
        """一键把 1min 转换成全部 8 种高阶周期并保存至 data 目录"""
        if not os.path.exists(self.source_file):
            return

        if not quiet:
            print(f"\n🔄 [多周期重采样] 正在由 1min 底座同步生成全部 8 大周期数据...")

        df_raw = pd.read_csv(self.source_file)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_1min = df_raw.set_index('Date')

        for tf_name, rule in self.timeframe_map.items():
            df_tf = self.resample_single_timeframe(df_1min, tf_name, rule)
            save_path = os.path.join(self.output_dir, f"XAU_USD_{tf_name}.csv")
            df_tf.to_csv(save_path, index=False)

            if not quiet:
                last_row = df_tf.iloc[-1]
                print(f"   ✅ [{tf_name:>5}] 成功生成 -> {save_path}")

        if not quiet:
            print("🎉 全部周期数据同步完毕！\n")

# ==========================================
# How to run:
# ==========================================
if __name__ == "__main__":
    resampler = TimeframeResampler()

    # Just call this line to instantly generate all 8 timeframe files!
    resampler.generate_all_timeframes()
