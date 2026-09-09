import os
import time
import pandas as pd
from dotenv import load_dotenv
from twelvedata import TDClient
from datetime import datetime, timedelta

from timeframe_resampler import TimeframeResampler
from self_improve import StrategySelfImprove
from trade_tracker import TradeTracker

load_dotenv()

class DataAcquisition:
    def __init__(self):
        self.symbol = "XAU/USD"
        self.interval = "1min"
        self.file_path = "data/XAU_USD_prices.csv"
        self.improver = StrategySelfImprove()
        os.makedirs("data", exist_ok=True)

        self.api_keys = []
        for i in range(1, 9):
            k = os.getenv(f"TWELVE_DATA_API_KEY_{i}")
            if k:
                self.api_keys.append((i, k))

        if not self.api_keys:
            raise Exception("No API keys found")

        self.state_file = "data/.key_index"
        print("Data Acquisition Initialized + API Key Rotator Ready.")

        self.resampler = TimeframeResampler(source_file=self.file_path)

    def _get_next_key(self):
        index = 0

        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    index = int(f.read().strip())
            except:
                index = 0

        key_id, key_val = self.api_keys[index % len(self.api_keys)]
        with open(self.state_file, "w") as f:
            f.write(str((index + 1) % len(self.api_keys)))
        return key_id, key_val

    def _format_dataframe(self, ts):
        df = ts.as_pandas()
        if df is None or df.empty:
            return None

        df = df.sort_index(ascending=True).reset_index()

        df.rename(columns={
            'datetime': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        }, inplace=True)

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        price_spread = (df['High'] - df['Low']).replace(0, 0.05)
        df['Volume'] = (price_spread * 1000).round(1)

        indicator_cols = ['rsi', 'cci', 'slow_k', 'slow_d', 'macd', 'macd_signal', 'macd_hist', 'mom']
        for col in indicator_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        avg_vol = df['Volume'].rolling(14).mean()
        df['rvol'] = (df['Volume'] / avg_vol).round(4)
        df['rvol'] = df['rvol'].fillna(1.0)

        return df

    def backfill_full_month_distribution(self):
        print("\n" + "="*80)
        print("Chatbot Initialization Started: Pulling the past 1 month 1-min data")
        print("\n" + "=" * 80)

        now = datetime.now()
        start_date_total = now - timedelta(days=30)
        num_chunks = len(self.api_keys)
        chunk_delta = (now - start_date_total) / num_chunks

        collected_dfs = []

        for idx, (key_id, api_key) in enumerate(self.api_keys):
            chunk_start = start_date_total + (idx * chunk_delta)
            chunk_end = start_date_total + ((idx + 1) * chunk_delta)

            s_str = chunk_start.strftime('%Y-%m-%d %H:%M:%S')
            e_str = chunk_end.strftime('%Y-%m-%d %H:%M:%S')

            print(f"[{idx+1}/{num_chunks}] Used Key_{key_id} to get data between {s_str} - {e_str}.")

            try:
                td = TDClient(apikey=api_key)

                ts = td.time_series(
                    symbol=self.symbol,
                    interval=self.interval,
                    start_date=s_str,
                    end_date=e_str,
                    outputsize=5000,
                    timezone="Asia/Kuala_Lumpur"
                ).with_rsi(time_period=14).with_cci(time_period=20).with_stoch(fast_k_period=14, slow_k_period=1, slow_d_period=3)\
                .with_macd(fast_period=12, slow_period=26, signal_period=9).with_mom(time_period=9)

                df_part = self._format_dataframe(ts)
                if df_part is not None and not df_part.empty:
                    collected_dfs.append(df_part)
                    print(f"Key[{key_id}] Fetched {len(df_part)} rows of K-line successfully.")
                else:
                    print("Market might closed.")

            except Exception as e:
                print(f"Key_{key_id} request failed: {e}")

            time.sleep(0.5)

        if collected_dfs:
            print("Perform data cleaning and combination for the past 1 month 1-min data.")
            full_df: pd.DataFrame = pd.concat(collected_dfs, ignore_index=True)
            full_df['Date'] = pd.to_datetime(full_df['Date'])
            full_df = full_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
            full_df['Date'] = full_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            full_df.to_csv(self.file_path, index=False)
            print("="*80)
            print(f"Pass one month 1-min data combined successfully with total {len(full_df)}, duration from {full_df['Date'].iloc[0]} to {full_df['Date'].iloc[-1]}")
        else:
            print("No data found for the past 1 month 1-min data.")

    def _distributed_pull_range(self, start_dt: datetime, end_dt: datetime):
        total_seconds = (end_dt - start_dt).total_seconds()
        num_chunks = len(self.api_keys)
        chunk_delta = timedelta(seconds=total_seconds/ num_chunks)

        collected_dfs = []
        for idx, (key_id, api_key) in enumerate(self.api_keys):
            chunk_start = start_dt + (idx * chunk_delta)
            chunk_end = start_dt + ((idx + 1) * chunk_delta)

            s_str = chunk_start.strftime('%Y-%m-%d %H:%M:%S')
            e_str = chunk_end.strftime('%Y-%m-%d %H:%M:%S')

            print(f"[{idx+1}/{num_chunks}]  Key_{key_id} handle data between {s_str} - {e_str}.")

            try:
                td = TDClient(apikey=api_key)
                ts = td.time_series(
                    symbol=self.symbol,
                    interval=self.interval,
                    start_date=s_str,
                    end_date=e_str,
                    outputsize=5000,
                    timezone="Asia/Kuala_Lumpur"
                ).with_rsi(time_period=14) \
                    .with_cci(time_period=20) \
                    .with_stoch(fast_k_period=14, slow_k_period=1, slow_d_period=3) \
                    .with_macd(fast_period=12, slow_period=26, signal_period=9) \
                    .with_mom(time_period=9)

                df_part = self._format_dataframe(ts)
                if df_part is not None and not df_part.empty:
                    collected_dfs.append(df_part)
                    print(f"(Key_{key_id}) fetched {len(df_part)} rows of K-line successfully.\n")
                else:
                    print("Market might closed.")

            except Exception as e:
                print(f"Key_{key_id} request failed: {e}")

            time.sleep(0.5)

        if collected_dfs:
            new_data: pd.DataFrame = pd.concat(collected_dfs, ignore_index=True)
            if os.path.exists(self.file_path):
                old_df = pd.read_csv(self.file_path)
                full_df: pd.DataFrame = pd.concat([old_df, new_data], ignore_index=True)
            else:
                full_df = new_data

            full_df['Date'] = pd.to_datetime(full_df['Date'])
            full_df = full_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
            full_df['Date'] = full_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            full_df.to_csv(self.file_path, index=False)
            print(f"Combination Completed. Current dataset size is {len(full_df)} rows.")

            self.resampler.generate_all_timeframes(quiet=False)

        else:
            print("Nothing to do.")

    def auto_heal_and_sync(self):
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)

        print("\n" + "="*80)
        print(f"Checking dataset integrity for 30 days before {now.strftime('%Y-%m-%d %H:%M:%S')}.")

        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            print("Not found local dataset, start using 8 API key to get complete 30 days historical data.")
            self._distributed_pull_range(thirty_days_ago, now)
            print("=" * 80 + "\n")
            return

        df = pd.read_csv(self.file_path)
        if df.empty or 'Date' not in df.columns:
            print("Can't read local dataset, start using 8 API key to get complete 30 days historical data.")
            self._distributed_pull_range(thirty_days_ago, now)
            print("=" * 80 + "\n")
            return

        first_date = pd.to_datetime(df['Date'].iloc[0]).to_pydatetime()
        if first_date > thirty_days_ago + timedelta(days=2):
            print(f"Historical data not enough 30 days, start data acquisition again.")
            self._distributed_pull_range(thirty_days_ago, first_date)
            df = pd.read_csv(self.file_path)

        last_date = pd.to_datetime(df['Date'].iloc[-1]).to_pydatetime()
        gap_seconds = (now - last_date).total_seconds()
        gap_minutes = int(gap_seconds / 60)

        if gap_minutes >= 1:
            print(f"[Inconsistent dataset] Last record time: {last_date}, total missed records: {gap_minutes} minutes.")

            if gap_minutes >= 8:
                print(f"Spliting the {gap_minutes} minutes to 8 API keys, each API Key handle {gap_minutes // 8 + 1} minutes.")
                self._distributed_pull_range(last_date, now)
            else:
                key_id, api_key = self._get_next_key()
                print(f"(Key_{key_id}) handling the data acquisition")
                try:
                    td = TDClient(apikey=api_key)
                    ts = td.time_series(symbol=self.symbol, interval=self.interval, outputsize=gap_minutes + 5,
                                        timezone="Asia/Kuala_Lumpur") \
                        .with_rsi(time_period=14) \
                        .with_cci(time_period=20) \
                        .with_stoch(fast_k_period=14, slow_k_period=1, slow_d_period=3) \
                        .with_macd(fast_period=12, slow_period=26, signal_period=9) \
                        .with_mom(time_period=9)
                    df_gap = self._format_dataframe(ts)
                    if df_gap is not None:
                        merged: pd.DataFrame = pd.concat([df, df_gap], ignore_index=True)
                        merged['Date'] = pd.to_datetime(merged['Date'])
                        merged = merged.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
                        merged['Date'] = merged['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        merged.to_csv(self.file_path, index=False)
                        print(f"Data integrity check completed.")

                        self.resampler.generate_all_timeframes(quiet=True)

                except Exception as e:
                    print(f"Key_{key_id} request failed: {e}")
        else:
            print(f"Dataset integrity is perfect, no need sync missing data.")

        print("=" * 80 + "\n")

    def _sleep_until_next_minute(self):
        now = datetime.now()
        sleep_seconds = 60 - now.second + 2
        if sleep_seconds > 60:
            sleep_seconds -= 60
        time.sleep(sleep_seconds)

    def start_minute_crawler(self):
        print(f"\nOne minute real-time data acquisition started from {datetime.now()}.")
        print("Ctrl + C to exit.")
        print("="*80 + "\n")

        try:
            while True:
                self._sleep_until_next_minute()

                key_id, api_key = self._get_next_key()
                now_str = datetime.now().strftime('%H:%M:%S')

                try:
                    td = TDClient(apikey=api_key)
                    ts = td.time_series(
                        symbol=self.symbol,
                        interval=self.interval,
                        outputsize=15,
                        timezone="Asia/Kuala_Lumpur"
                    ).with_rsi(time_period=14).with_cci(time_period=20).with_stoch(fast_k_period=14, slow_k_period=1, slow_d_period=3)\
                        .with_macd(fast_period=12, slow_period=26, signal_period=9).with_mom(time_period=9)

                    df_new = self._format_dataframe(ts)

                    if df_new is not None and not df_new.empty:
                        df_old = pd.read_csv(self.file_path)
                        prev_count = len(df_old)

                        merged: pd.DataFrame = pd.concat([df_old, df_new], ignore_index=True)
                        merged['Date'] = pd.to_datetime(merged['Date'])
                        merged = merged.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
                        merged['Date'] = merged['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                        merged.to_csv(self.file_path, index=False)

                        tracker = TradeTracker()
                        tracker.update_and_track()

                        self.resampler.generate_all_timeframes(quiet=True)

                        if hasattr(self, 'improver') and self.improver.should_reflect():
                            self.improver.run_reflection()

                        last_row = merged.iloc[-1]
                        candle_time = last_row['Date']
                        current_price = last_row['Close']
                        current_rsi = last_row.get('rsi', 0)

                        new_bars_added = len(merged) - prev_count

                        if new_bars_added > 0:
                            print(f"[{now_str}] New K-line recorded | Time: {candle_time}  |  Current Price: {current_price}  |  RSI: {current_rsi:.2f}  (Key_{key_id})")
                        else:
                            print(f"[{now_str}] Updating new K-line | Time: {candle_time}  |  Current Price: {current_price}  |  RSI: {current_rsi:.2f}  (Key_{key_id})")

                    else:
                        print(f"[{now_str}] No data found by (Key_{key_id})")

                except Exception as req_err:
                    print(f"[{now_str}] Network timeout or error by (Key_{key_id}): {req_err}")

        except KeyboardInterrupt:
            print("One minute data acquisition stopped.")

    def start(self):
        self.auto_heal_and_sync()

        self.resampler.generate_all_timeframes(quiet=True)

        self.start_minute_crawler()

if __name__ == "__main__":
    pipeline = DataAcquisition()
    pipeline.start()