# ALPHA-D: Autonomous Quantitative Intelligence Terminal for XAU/USD

ALPHA-D is a production-grade algorithmic trading system designed for XAU/USD (Spot Gold). It combines an academic **Hierarchical Multi-Timeframe Dempster-Shafer (D-S) Evidence Fusion Framework** with an autonomous, LLM-driven scientific self-improvement engine.

---

## 🌟 Key Architecture & Highlights

1. **Hierarchical Evidence Fusion Engine (`algo_layer.py`)**:
   - **Tier 1 (Intra-frame Fuzzification)**: Fuzzifies 5 core indicators (RSI, CCI, MACD, Stochastic Oscillator, Momentum) into Basic Belief Assignments ($BBA = [m(BUY), m(SELL), m(HOLD)]$) adjusted by Dynamic Relative Volume (RVol).
   - **Tier 2 (Inter-frame Dynamic Aggregation)**: Synthesizes evidence across multiple timeframes (1H, M30, M15, M10, M5, M1) with information certainty weighting ($\Omega_\tau$).
   - **Risk Management**: Includes macro-trend veto checks and cross-timeframe conflict evaluation ($\Psi_{global}$). Dynamic stop-loss (SL) and take-profit (TP) are calculated using ATR(14).

2. **Autonomous Strategy Evolution Engine (`self_improve.py`)**:
   - Analyzes trade outcomes every $N$ trades (set by `reflection_cadence`).
   - Evaluates losing trades against entry snapshots ($\Psi_{global}$, confidence scores).
   - Enforces **Controlled Single-Variable Mutation**: Only modifies one parameter at a time.
   - Archives previous configurations in `state/history/` and logs hypotheses in `state/hypotheses.jsonl`.

3. **High-Frequency Real-time Pipeline (`data_agent.py`)**:
   - Features an API Key Rotator supporting distributed backfills across multiple keys.
   - Includes automatic data gap healing and real-time 1-minute data ingestion.
   - Updates virtual order execution and tracks active positions via `trade_tracker.py`.

4. **Institutional Bloomberg-Style Terminal (`app.py`)**:
   - Dark-mode analytics UI built with Streamlit.
   - Embedded native TradingView multi-indicator chart (RSI, CCI, MACD, Stoch, Momentum).
   - Built-in Quant Copilot powered by Google Gemini.

---

## 📂 Project Structure

```text
├── algo_layer.py           # Core D-S Evidence Theory Fusion Mathematical Engine
├── app.py                  # Streamlit Institutional Monitoring Terminal
├── data_agent.py           # Real-time Crawler, API Key Rotator & Sync Pipeline
├── self_improve.py         # Autonomous Reflection & Parameter Mutation Engine
├── timeframe_resampler.py  # Local High-Precision Multi-Timeframe Resampler
├── trade_tracker.py        # Virtual Order Lifecycle & Execution Tracker
├── test_evolution.py       # Mock Environment for Evolution Testing
├── data/                   # Data Directory (XAU_USD 1m and resampled CSVs)
└── state/                  # Strategy Configurations, Order Ledgers & Logs
    ├── goal.yaml           # Strategic Objectives & Optimization Boundaries
    ├── strategy.yaml       # Current Active Hyperparameter Configuration
    ├── trades.jsonl        # Completed Trade Ledgers
    ├── hypotheses.jsonl    # Log of AI-Generated Hypotheses & Mutations
    └── history/            # Backups of Historical Strategy Versions
```

---

## 🚀 Quick Start

### 1. Installation

```Bash
git clone https://github.com/your-username/alpha-d-terminal.git
cd alpha-d-terminal
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Setup
Create a .env file in the root directory:
```Env
# Gemini API Key (for Quant Copilot and Strategy Evolution)
GEMINI_API_KEY=your_google_gemini_api_key

# Twelve Data API Keys (Supports up to 8 keys for round-robin rotation)
TWELVE_DATA_API_KEY_1=your_twelve_data_key_1
TWELVE_DATA_API_KEY_2=your_twelve_data_key_2
```

### 3. Run the Data Pipeline
Start the data ingestion engine to backfill historical bars and begin live minute-by-minute streaming:
```Bash
python data_agent.py
```

### 4. Launch the Trading Terminal
Launch the dashboard in another terminal:
```Bash
streamlit run app.py
```

### 5. Verify the Evolution Engine
Simulate trade batches and trigger autonomous strategy mutation:
```Bash
python test_evolution.py
```


## Strategy Configuration (state/strategy.yaml)
| Parameter Category | Key Parameter              | Description                                                       |
|--------------------|----------------------------|-------------------------------------------------------------------|
| fuzzification      | k_rsi, k_cci               | Sigmoid steepness coefficients for extreme market bounds          |
| voter_weights      | rsi, cci, stoch, macd, mom | Intra-frame authority discounts                                   |
| macro_veto         | veto_trigger_threshold     | Conviction threshold on higher timeframes that triggers penalties |
| decision_rules     | delta_signal               | Global confidence threshold required for market entry             |
| decision_rules     | psi_max                    | Maximum allowable cross-timeframe conflict score                  |


## Risk Disclaimer
This software is developed for research and educational purposes only. Past performance and simulated evolution do not guarantee future returns. Always exercise risk management when deploying algorithmic models.