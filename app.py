"""
Economic Simulation Dashboard
A simple interface to run and visualize macroeconomic simulations.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Model checkpoint path - try multiple locations
MODEL_PATHS = [
    "checkpoints/run_20251206_021307/best_model.pt",
    "checkpoints/run_20251205_125911/best_model.pt",
    "checkpoints/best_model.pt",
    "best_model.pt",
]

def find_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None

MODEL_PATH = find_model_path()

# Store simulation results
sim_result = {"df": None, "loaded": False}

# Import simulation modules
try:
    from config import DEFAULT_CONFIG
    from environment import MacroEconEnvironment
    from training import MultiAgentPPO, PPOConfig
    READY = True
except ImportError as e:
    READY = False
    IMPORT_ERR = str(e)


def get_model(config):
    """Load trained model."""
    env = MacroEconEnvironment(config)
    ppo = MultiAgentPPO(agent_configs=env.get_agent_configs(), device="cpu")
    
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        ppo.load(MODEL_PATH)
        return ppo, True
    return ppo, False


def run_sim(duration, households, firms, banks):
    """Run the economic simulation."""
    config = DEFAULT_CONFIG
    config.economic.simulation_length = duration
    config.economic.num_households = households
    config.economic.num_firms = firms
    config.economic.num_commercial_banks = banks
    
    env = MacroEconEnvironment(config)
    ppo, loaded = get_model(config)
    
    obs = env.reset()
    for _ in range(duration):
        if loaded:
            actions, _, _ = ppo.get_actions(obs, deterministic=True)
        else:
            actions = {
                "central_bank": {0: 10},
                "banks": {i: 18 for i in range(banks)},
                "households": {i: 12 for i in range(households)},
                "firms": {i: 50 for i in range(firms)},
            }
        
        result = env.step(actions)
        obs = env._get_observations()
        if result.dones.get("central_bank", False):
            break
    
    return env.get_history_dataframe(), loaded


@app.route("/")
def index():
    return render_template("index.html", ready=READY)


@app.route("/run", methods=["POST"])
def run():
    if not READY:
        return jsonify({"error": "System not ready"}), 500
    
    try:
        duration = int(request.form.get("duration", 120))
        households = int(request.form.get("households", 100))
        firms = int(request.form.get("firms", 20))
        banks = int(request.form.get("banks", 3))
        
        df, loaded = run_sim(duration, households, firms, banks)
        sim_result["df"] = df
        sim_result["loaded"] = loaded
        
        # Extract key data
        last = df.iloc[-1]
        first = df.iloc[0]
        
        # Format numbers properly
        inflation = last['inflation'] * 100 if last['inflation'] < 1 else last['inflation']
        unemployment = last['unemployment'] * 100 if last['unemployment'] < 1 else last['unemployment']
        policy_rate = last['policy_rate'] * 12 * 100 if last['policy_rate'] < 0.02 else last['policy_rate'] * 100
        gdp_change = ((last['gdp'] - first['gdp']) / first['gdp'] * 100) if first['gdp'] > 0 else 0
        
        return jsonify({
            "ok": True,
            "loaded": loaded,
            "stats": {
                "gdp": round(last['gdp'], 0),
                "gdp_pct": round(gdp_change, 1),
                "inflation": round(inflation, 1),
                "unemployment": round(unemployment, 1),
                "rate": round(policy_rate, 1)
            },
            "series": {
                "months": df["step"].tolist(),
                "gdp": df["gdp"].tolist(),
                "inflation": (df["inflation"] * 100).tolist() if df["inflation"].max() < 1 else df["inflation"].tolist(),
                "unemployment": (df["unemployment"] * 100).tolist() if df["unemployment"].max() < 1 else df["unemployment"].tolist(),
                "rate": (df["policy_rate"] * 12 * 100).tolist() if df["policy_rate"].max() < 0.02 else (df["policy_rate"] * 100).tolist()
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download")
def download():
    if sim_result["df"] is None:
        return "No data", 404
    
    buf = io.StringIO()
    sim_result["df"].to_csv(buf, index=False)
    buf.seek(0)
    
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='simulation.csv'
    )


if __name__ == "__main__":
    print(f"Economic Simulation Dashboard")
    print(f"Model: {MODEL_PATH}")
    print(f"URL: http://localhost:5000")
    app.run(debug=True, port=5000)