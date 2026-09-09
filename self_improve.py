import os
import json
import yaml
import shutil
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from datetime import datetime

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

class StrategySelfImprove:
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        self.strategy_file = os.path.join(state_dir, "strategy.yaml")
        self.goal_file = os.path.join(state_dir, "goal.yaml")
        self.trades_file = os.path.join(state_dir, "trades.jsonl")
        self.hypotheses_file = os.path.join(state_dir, "hypotheses.jsonl")
        self.history_dir = os.path.join(state_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found, please check .env file")

        self.client = genai.Client(api_key=api_key)


    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _read_closed_trades(self) -> list:
        if not os.path.exists(self.trades_file):
            return []
        trades = []
        with open(self.trades_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line.strip()))
        return trades

    def should_reflect(self) -> bool:
        # Check the number of closed trade is sufficient or not since last reflection
        trades = self._read_closed_trades()
        if not trades:
            return False

        goal = self._load_yaml(self.goal_file)
        cadence = goal.get("reflection_cadence", 5)

        num_hypotheses = 0
        if os.path.exists(self.hypotheses_file):
            with open(self.hypotheses_file, "r", encoding="utf-8") as f:
                num_hypotheses = sum(1 for line in f if line.strip())

        trades_since_last_reflection = len(trades) - (num_hypotheses * cadence)
        return trades_since_last_reflection >= cadence


    def run_reflection(self):
        trades = self._read_closed_trades()
        goal = self._load_yaml(self.goal_file)
        current_strategy = self._load_yaml(self.strategy_file)
        current_version = current_strategy.get("version", "v01")

        cadence = goal.get("reflection_cadence", 5)
        recent_trades = trades[-cadence:]

        wins = sum(1 for t in recent_trades if t.get("outcome") == "WIN")
        losses = len(recent_trades) - wins
        win_rate = round(wins / len(recent_trades), 2)
        total_pnl = round(sum(t.get("pnl", 0.0) for t in recent_trades), 2)

        print("="*80)
        print(f"[AI Self Reflection Engine]  Reflected the recent {len(recent_trades)} trades")
        print(f"Model Performance:  Win Rate: {win_rate:.2f}% ({wins} win / {losses} Losses)  |  Net P&L: {total_pnl:.2f}%")
        print("="*80)

        prompt = f"""
        You are the quantitative research brain of a self-improving algorithmic trading agent for XAU/USD.
        We use a Hierarchical Multi-Timeframe (1H, M30, M15, M10, M5, M1) Dempster-Shafer (D-S) Evidence Fusion Model.

        Goal Specification:
        {json.dumps(goal, indent=2)}

        Current Strategy Parameters ({current_version}):
        {json.dumps(current_strategy, indent=2)}

        Recent {len(recent_trades)} Closed Trades Outcome & Feature Snapshots:
        {json.dumps(recent_trades, indent=2)}

        CRITICAL SCIENTIFIC METHOD CONSTRAINTS:
        1. You MUST change EXACTLY ONE variable from the Current Strategy parameters. Changing more than one variable is STRICTLY FORBIDDEN.
        2. The candidate variables you can optimize are:
           - decision_rules.delta_signal (range 0.45 to 0.70)
           - decision_rules.delta_margin (range 0.10 to 0.25)
           - decision_rules.psi_max (range 0.40 to 0.75)
           - macro_veto.veto_trigger_threshold (range 0.40 to 0.65)
           - macro_veto.veto_pass_threshold (range 0.50 to 0.75)
           - fuzzification.k_rsi (range 0.10 to 0.25)
           - fuzzification.k_cci (range 0.01 to 0.04)
        3. Analyze why the losing trades failed based on their 'entry_snapshot' (e.g., was psi_global too high indicating cross-timeframe conflict? Was delta_signal too loose? Was macro_veto not strict enough?).
        4. Provide a clear scientific hypothesis explaining why changing this ONE variable will improve future risk-adjusted returns.

        Respond STRICTLY in valid JSON format with no markdown wrappers or additional text:
        {{
          "diagnosis": "Brief explanation of what went wrong in the losing trades",
          "target_section": "decision_rules or macro_veto or fuzzification",
          "target_parameter": "parameter_name_here",
          "old_value": 0.0,
          "new_value": 0.0,
          "hypothesis": "Clear statement: By changing X from old to new, we expect Y"
        }}
        """

        print("Sending Information to Google Gemini, generating the order hypothesis.")
        try:
            response = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents = prompt
            )

            raw_text = response.text.strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            reflection_result = json.loads(raw_text.strip())

        except Exception as e:
            print(f"Model Interpretation Error: {e}")
            return

        section = reflection_result.get("target_section")
        param = reflection_result.get("target_parameter")
        new_val = reflection_result.get("new_value")
        old_val = reflection_result.get("old_value")

        if section not in current_strategy or param not in current_strategy[section]:
            print(f"Current parameter {param} not in current strategy {section}")
            return

        # 1. Archive current old strategy version
        next_ver_num = int(current_version.replace("v", "")) + 1
        next_version = f"v{next_ver_num:02d}"
        archive_path = os.path.join(self.history_dir, f"{current_version}.yaml")
        shutil.copyfile(self.strategy_file, archive_path)
        print(f"Current Strategy Version: {archive_path}")

        # 2. Update parameter and generate new version
        current_strategy[section][param] = new_val
        current_strategy["version"] = next_version

        with open(self.strategy_file, "w", encoding="utf-8") as f:
            yaml.dump(current_strategy, f, sort_keys=False)


        # 3. Record scientific hypothesis to hypothesis.jsonl
        hypothesis_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "from_version": current_version,
            "to_version": next_version,
            "diagnosis": reflection_result.get("diagnosis"),
            "mutation": f"{section}.{param}: {old_val} -> {new_val}",
            "hypothesis": reflection_result.get("hypothesis")
        }

        with open(self.hypotheses_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(hypothesis_record) + "\n")

        print("="*80)
        print(f"Strategy Updated: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 【{current_version} -> {next_version}】")
        print(f"Updated Parameter: {section}.{param}: {old_val} -> {new_val}")
        print(f"Scientific Hypothesis: {reflection_result.get('hypothesis')}")
        print("="*80 + "\n")

if __name__ == "__main__":
    improver = StrategySelfImprove()
    improver.run_reflection()

