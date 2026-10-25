"""Interactive Web Dashboard for Q-learning Taxi-v4 Agent.

Run with:
    python -m src.dashboard
or:
    uvicorn src.dashboard:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import QLearningAgent
from src.env_utils import decode_state, make_env

app = FastAPI(title="Taxi-v4 Q-Learning Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount figures static directory if available
figures_path = Path("results/figures")
if figures_path.exists():
    app.mount("/figures", StaticFiles(directory=str(figures_path)), name="figures")

# Load models into memory
MODELS: dict[str, np.ndarray | None] = {}
tuned_path = Path("results/q_table_final.npy")
if tuned_path.exists():
    MODELS["tuned"] = np.load(tuned_path)

baseline_path = Path("results/q_table_baseline.npy")
if baseline_path.exists():
    MODELS["baseline"] = np.load(baseline_path)

MODELS["untrained"] = np.zeros((500, 6), dtype=np.float64)
MODELS["random"] = None


class SimulationRequest(BaseModel):
    policy: str = "tuned"  # "tuned", "baseline", "untrained", "random"
    seed: int = 10042
    max_steps: int = 30


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Return saved evaluation test metrics and comparison table."""
    results: dict[str, Any] = {"metrics": None, "comparison": None}

    test_metrics_path = Path("results/test_metrics.json")
    if test_metrics_path.exists():
        with open(test_metrics_path, encoding="utf-8") as f:
            results["metrics"] = json.load(f)

    comparison_path = Path("results/comparison_table.csv")
    if comparison_path.exists():
        import pandas as pd

        df = pd.read_csv(comparison_path)
        results["comparison"] = df.to_dict(orient="records")

    return results


@app.post("/api/simulate")
def simulate(req: SimulationRequest) -> dict[str, Any]:
    """Run rollout simulation on the environment and return step-by-step trace."""
    env = make_env("Taxi-v4")
    obs, _ = env.reset(seed=req.seed)
    rng = np.random.default_rng(req.seed)

    q_table = MODELS.get(req.policy)
    if q_table is not None:
        agent = QLearningAgent()
        agent.q_table = q_table.copy()
    else:
        agent = None

    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    action_symbols = ["↓", "↑", "→", "←", "P", "D"]

    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    step_num = 0

    # Initial state record
    r, c, p, d = decode_state(int(obs))
    init_q = q_table[int(obs)].tolist() if q_table is not None else [0.0] * 6
    trace.append(
        {
            "step": 0,
            "action": None,
            "action_name": "Initial State",
            "action_symbol": "•",
            "state": int(obs),
            "taxi_row": r,
            "taxi_col": c,
            "pass_loc": p,
            "dest_loc": d,
            "reward": 0,
            "total_reward": 0,
            "q_values": [round(float(v), 2) for v in init_q],
            "terminated": False,
            "truncated": False,
        }
    )

    while not done and step_num < req.max_steps:
        state_idx = int(obs)
        if req.policy == "random" or agent is None:
            action = int(rng.integers(0, 6))
            current_q = [0.0] * 6
        else:
            action = agent.greedy_action(state_idx, rng)
            current_q = agent.q_table[state_idx].tolist()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        reward_val = int(reward)
        total_reward += reward_val
        step_num += 1

        next_r, next_c, next_p, next_d = decode_state(int(next_obs))
        done = bool(terminated or truncated)

        trace.append(
            {
                "step": step_num,
                "action": action,
                "action_name": action_names[action],
                "action_symbol": action_symbols[action],
                "state": int(next_obs),
                "taxi_row": next_r,
                "taxi_col": next_c,
                "pass_loc": next_p,
                "dest_loc": next_d,
                "reward": reward_val,
                "total_reward": int(total_reward),
                "q_values": [round(float(v), 2) for v in current_q],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        obs = next_obs

    env.close()

    return {
        "policy": req.policy,
        "seed": req.seed,
        "total_steps": step_num,
        "total_reward": total_reward,
        "success": trace[-1]["terminated"],
        "trace": trace,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Taxi-v4 Q-Learning Agent Dashboard</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --primary-maroon: #800000;
      --primary-maroon-dark: #600000;
      --primary-maroon-light: #991b1b;
      --primary-maroon-soft: rgba(128, 0, 0, 0.08);
      --accent-gold: #f59e0b;
      --accent-gold-dark: #d97706;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --green: #16a34a;
      --red: #dc2626;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
      --shadow-lg: 0 10px 15px -3px rgba(128,0,0,0.08), 0 4px 6px -4px rgba(128,0,0,0.04);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text-main);
      line-height: 1.5;
      padding: 24px;
    }
    .container {
      max-width: 1240px;
      margin: 0 auto;
    }

    /* HEADER */
    header {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-top: 5px solid var(--primary-maroon);
      border-radius: 12px;
      padding: 18px 24px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header-titles h1 {
      font-size: 1.55rem;
      font-weight: 800;
      color: var(--primary-maroon);
      display: flex;
      align-items: center;
      gap: 10px;
      letter-spacing: -0.02em;
    }
    .header-titles p {
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-top: 3px;
    }
    .header-badges {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      font-size: 0.78rem;
      padding: 5px 12px;
      border-radius: 9999px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .badge-green {
      background: rgba(22, 163, 74, 0.12);
      color: var(--green);
      border: 1px solid rgba(22, 163, 74, 0.25);
    }
    .badge-maroon {
      background: var(--primary-maroon-soft);
      color: var(--primary-maroon);
      border: 1px solid rgba(128, 0, 0, 0.2);
    }
    .canva-link-btn {
      background: var(--primary-maroon);
      color: #ffffff;
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 700;
      padding: 8px 16px;
      border-radius: 8px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
      box-shadow: 0 2px 6px rgba(128,0,0,0.2);
    }
    .canva-link-btn:hover {
      background: var(--primary-maroon-dark);
      transform: translateY(-1px);
    }

    /* NAVIGATION TABS */
    .tabs-nav {
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      border-bottom: 2px solid var(--border);
      padding-bottom: 4px;
    }
    .tab-btn {
      background: transparent;
      color: var(--text-muted);
      border: none;
      padding: 10px 22px;
      border-radius: 8px 8px 0 0;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }
    .tab-btn:hover { color: var(--primary-maroon); }
    .tab-btn.active {
      color: var(--primary-maroon);
      background: rgba(128, 0, 0, 0.06);
    }
    .tab-btn.active::after {
      content: '';
      position: absolute;
      bottom: -6px;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--primary-maroon);
      border-radius: 3px 3px 0 0;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.25s ease-out; }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* MAIN SIMULATION GRID LAYOUT */
    .sim-layout {
      display: grid;
      grid-template-columns: 480px 1fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 1024px) {
      .sim-layout { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      box-shadow: var(--shadow-sm);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--primary-maroon);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* 5x5 WORLD ARENA WITH COORDINATE LABELS */
    .arena-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-bottom: 20px;
    }
    .col-labels {
      display: grid;
      grid-template-columns: repeat(5, 76px);
      margin-left: 28px;
      text-align: center;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .arena-middle-row {
      display: flex;
      align-items: center;
    }
    .row-labels {
      display: grid;
      grid-template-rows: repeat(5, 76px);
      width: 28px;
      text-align: right;
      padding-right: 8px;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
    }
    .taxi-world-grid {
      position: relative;
      width: 380px;
      height: 380px;
      background: #ffffff;
      border: 3px solid #334155;
      border-radius: 8px;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      grid-template-rows: repeat(5, 1fr);
      box-shadow: var(--shadow-md);
      overflow: hidden;
    }
    .cell {
      position: relative;
      border: 1px dashed #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    /* Accurate Taxi Grid Walls */
    .cell.wall-right { border-right: 5px solid #1e293b !important; }
    .cell.wall-left { border-left: 5px solid #1e293b !important; }

    /* Landmark Badges */
    .landmark-box {
      position: absolute;
      width: 20%;
      height: 20%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      font-weight: 900;
      z-index: 2;
      opacity: 0.85;
      border-radius: 4px;
    }
    .landmark-box.R { top: 0; left: 0; background: rgba(239, 68, 68, 0.18); color: #dc2626; border: 2px dashed rgba(239, 68, 68, 0.6); }
    .landmark-box.G { top: 0; left: 80%; background: rgba(22, 163, 74, 0.18); color: #16a34a; border: 2px dashed rgba(22, 163, 74, 0.6); }
    .landmark-box.Y { top: 80%; left: 0; background: rgba(234, 179, 8, 0.18); color: #ca8a04; border: 2px dashed rgba(234, 179, 8, 0.6); }
    .landmark-box.B { top: 80%; left: 60%; background: rgba(37, 99, 235, 0.18); color: #2563eb; border: 2px dashed rgba(37, 99, 235, 0.6); }

    /* Destination Pulsing Halo */
    .destination-halo {
      position: absolute;
      width: 20%;
      height: 20%;
      border-radius: 8px;
      border: 4px dashed var(--primary-maroon);
      background: rgba(128, 0, 0, 0.12);
      z-index: 3;
      transition: all 0.3s ease;
      animation: pulseMaroon 1.8s infinite;
    }
    @keyframes pulseMaroon {
      0% { transform: scale(0.92); opacity: 0.7; }
      50% { transform: scale(1.04); opacity: 1; }
      100% { transform: scale(0.92); opacity: 0.7; }
    }

    /* Taxi and Passenger Avatars */
    .taxi-token {
      position: absolute;
      width: 17%;
      height: 17%;
      background: var(--accent-gold);
      border: 2px solid #ffffff;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-size: 1.6rem;
      z-index: 10;
      transition: top 0.22s cubic-bezier(0.34, 1.56, 0.64, 1), left 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.45);
    }
    .passenger-token {
      position: absolute;
      width: 15%;
      height: 15%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      z-index: 8;
      transition: all 0.2s ease;
    }

    /* CONTROLS PANEL */
    .sim-controls {
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .control-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .control-label {
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-muted);
      min-width: 65px;
    }
    select, input[type="number"] {
      background: #ffffff;
      border: 1px solid var(--border-dark);
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 600;
      outline: none;
      transition: border-color 0.15s;
    }
    select:focus, input[type="number"]:focus {
      border-color: var(--primary-maroon);
    }
    .btn {
      background: var(--primary-maroon);
      color: #ffffff;
      border: none;
      padding: 9px 18px;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn:hover { background: var(--primary-maroon-dark); }
    .btn-secondary {
      background: #ffffff;
      color: var(--text-main);
      border: 1px solid var(--border-dark);
    }
    .btn-secondary:hover {
      background: #f1f5f9;
      border-color: #94a3b8;
    }

    /* SCRUBBER SLIDER */
    .scrubber-container {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 4px;
    }
    .scrubber-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-muted);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--primary-maroon);
      cursor: pointer;
    }

    /* RIGHT COLUMN: TELEMETRY & Q-VALUES */
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric-card {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 10px;
      text-align: center;
    }
    .metric-value {
      font-size: 1.45rem;
      font-weight: 800;
      color: var(--primary-maroon);
      letter-spacing: -0.02em;
    }
    .metric-label {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .journey-info-box {
      background: #ffffff;
      border: 1px solid var(--border);
      border-left: 4px solid var(--primary-maroon);
      border-radius: 6px;
      padding: 12px 16px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.9rem;
    }

    /* ACTION Q-VALUE BARS */
    .q-value-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .q-value-row {
      display: grid;
      grid-template-columns: 110px 1fr 65px;
      align-items: center;
      gap: 12px;
    }
    .q-action-title {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-main);
    }
    .q-action-title.selected-action {
      color: var(--primary-maroon);
      font-weight: 800;
    }
    .q-bar-bg {
      height: 20px;
      background: #f1f5f9;
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }
    .q-bar-fill {
      height: 100%;
      background: #cbd5e1;
      transition: width 0.22s ease;
      border-radius: 3px;
    }
    .q-bar-fill.highlight {
      background: var(--primary-maroon);
    }
    .q-num-label {
      font-size: 0.85rem;
      font-weight: 700;
      text-align: right;
    }
    .q-num-label.pos { color: var(--green); }
    .q-num-label.neg { color: var(--red); }

    /* BENCHMARKS TABLE */
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }
    table.data-table th, table.data-table td {
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 0.92rem;
    }
    table.data-table th {
      background: #f8fafc;
      color: var(--text-muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    table.data-table tr:hover {
      background: rgba(128, 0, 0, 0.02);
    }

    /* FIGURES GALLERY */
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 24px;
    }
    .gallery-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: var(--shadow-sm);
    }
    .gallery-card-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--primary-maroon);
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
    }
    .gallery-card img {
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: block;
    }
  </style>
</head>
<body>

  <div class="container">

    <!-- HEADER -->
    <header>
      <div class="header-titles">
        <h1>🚖 Taxi-v4 Q-Learning Agent Dashboard</h1>
        <p>Tabular Reinforcement Learning with Unbiased Vectorized Tie-Breaking & Optimal Spatial Planning</p>
      </div>
      <div class="header-badges">
        <span class="badge badge-green">100% Success Rate</span>
        <span class="badge badge-maroon">0 Illegal Moves</span>
        <a href="https://canva.link/vog4wcqpdx5qgqx" target="_blank" class="canva-link-btn">
          🎨 View Canva Design
        </a>
      </div>
    </header>

    <!-- NAVIGATION TABS -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab(event, 'sim')">🎮 Interactive Simulation</button>
      <button class="tab-btn" onclick="switchTab(event, 'bench')">📊 Comparative Benchmarks</button>
      <button class="tab-btn" onclick="switchTab(event, 'gallery')">🖼️ Publication Figures (V1–V9)</button>
    </div>

    <!-- TAB 1: INTERACTIVE SIMULATION -->
    <div id="tab-sim" class="tab-content active">
      <div class="sim-layout">
        
        <!-- LEFT COLUMN: 5x5 TAXI WORLD ARENA -->
        <div class="card">
          <div class="card-header">
            <div class="card-title">5x5 Grid World Environment</div>
            <span id="delivery-status-badge" class="badge badge-green">Ready</span>
          </div>

          <div class="arena-container">
            <!-- Column Coordinate Numbers (0..4) -->
            <div class="col-labels">
              <div>0</div><div>1</div><div>2</div><div>3</div><div>4</div>
            </div>
            <div class="arena-middle-row">
              <!-- Row Coordinate Numbers (0..4) -->
              <div class="row-labels">
                <div>0</div><div>1</div><div>2</div><div>3</div><div>4</div>
              </div>
              <!-- Grid Matrix -->
              <div class="taxi-world-grid" id="arena-grid">
                <!-- Landmark Areas R, G, Y, B -->
                <div class="landmark-box R">R</div>
                <div class="landmark-box G">G</div>
                <div class="landmark-box Y">Y</div>
                <div class="landmark-box B">B</div>

                <!-- Destination Target Halo -->
                <div id="dest-ring" class="destination-halo"></div>

                <!-- Passenger and Taxi Tokens -->
                <div id="pass-token" class="passenger-token">🧍</div>
                <div id="taxi-token" class="taxi-token">🚖</div>

                <!-- 25 Grid Cells with Walls -->
                <!-- Row 0 -->
                <div class="cell"></div><div class="cell wall-right"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div>
                <!-- Row 1 -->
                <div class="cell"></div><div class="cell wall-right"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div>
                <!-- Row 2 -->
                <div class="cell"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div>
                <!-- Row 3 -->
                <div class="cell wall-right"></div><div class="cell"></div><div class="cell wall-right"></div><div class="cell"></div><div class="cell"></div>
                <!-- Row 4 -->
                <div class="cell wall-right"></div><div class="cell"></div><div class="cell wall-right"></div><div class="cell"></div><div class="cell"></div>
              </div>
            </div>
          </div>

          <!-- CONTROLS -->
          <div class="sim-controls">
            <div class="control-row">
              <span class="control-label">Policy:</span>
              <select id="policy-picker" style="flex: 1;">
                <option value="tuned" selected>Tuned Model (α=0.7, γ=0.99) — 800 ep fast</option>
                <option value="baseline">Baseline Model (α=0.1, γ=0.99) — 2,300 ep</option>
                <option value="untrained">Untrained Agent (All Zeros)</option>
                <option value="random">Random Policy</option>
              </select>
            </div>

            <div class="control-row">
              <span class="control-label">Seed:</span>
              <input type="number" id="seed-input" value="10042" style="width: 110px;">
              <button class="btn btn-secondary" onclick="randomizeSeed()">🎲 Random</button>
              <button class="btn" onclick="executeSimulation()">⚡ Run Rollout</button>
            </div>

            <!-- SCRUBBER SLIDER -->
            <div class="scrubber-container">
              <div class="scrubber-header">
                <span>Trajectory Timeline</span>
                <span id="scrubber-text">Step 0 / 0</span>
              </div>
              <input type="range" id="step-slider" min="0" max="0" value="0" oninput="onSliderMove(this.value)">
            </div>

            <!-- PLAYBACK BUTTONS -->
            <div style="display: flex; gap: 8px; justify-content: center; margin-top: 6px;">
              <button class="btn btn-secondary" onclick="jumpToStep(0)">⏮ Reset</button>
              <button class="btn btn-secondary" onclick="stepDelta(-1)">◀ Prev</button>
              <button class="btn" id="play-pause-btn" onclick="togglePlayPause()" style="min-width: 100px;">▶ Play</button>
              <button class="btn btn-secondary" onclick="stepDelta(1)">Next ▶</button>
              <button class="btn btn-secondary" onclick="jumpToStep(currentTrace.length - 1)">End ⏭</button>
            </div>
          </div>
        </div>

        <!-- RIGHT COLUMN: TELEMETRY & Q-VALUES -->
        <div style="display: flex; flex-direction: column; gap: 20px;">
          
          <!-- TELEMETRY STATS -->
          <div class="card">
            <div class="card-header">
              <div class="card-title">Live Episode Telemetry</div>
              <span id="route-indicator" style="font-size: 0.85rem; font-weight: 700; color: var(--primary-maroon);">Seed: 10042</span>
            </div>

            <div class="telemetry-grid">
              <div class="metric-card">
                <div class="metric-value" id="disp-step">0 / 0</div>
                <div class="metric-label">Step Count</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" id="disp-action" style="font-size: 1.15rem;">-</div>
                <div class="metric-label">Last Action</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" id="disp-reward">0</div>
                <div class="metric-label">Step Reward</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" id="disp-total-reward">0</div>
                <div class="metric-label">Total Reward</div>
              </div>
            </div>

            <div class="journey-info-box">
              <div>
                Passenger Status: <strong id="disp-passenger-status" style="color: var(--primary-maroon);">Waiting at Y</strong>
              </div>
              <div>
                Target Destination: <strong id="disp-dest-name" style="color: var(--green);">G (0, 4)</strong>
              </div>
            </div>
          </div>

          <!-- Q-VALUES BAR CHART -->
          <div class="card">
            <div class="card-header">
              <div class="card-title">Action Q-Values for Current State: Q(s, a)</div>
              <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">Greedy Action Highlighted</span>
            </div>

            <div class="q-value-list" id="q-value-bars">
              <!-- Dynamically populated -->
            </div>
          </div>

        </div>

      </div>
    </div>

    <!-- TAB 2: BENCHMARKS -->
    <div id="tab-bench" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div class="card-title">1,000 Unseen Test Episodes Comparative Benchmarks</div>
          <span class="badge badge-maroon">Seeds 10000–10999</span>
        </div>

        <table class="data-table">
          <thead>
            <tr>
              <th>Policy Model</th>
              <th>Episodes</th>
              <th>Mean Reward</th>
              <th>Std Reward</th>
              <th>95% Bootstrap CI</th>
              <th>Mean Steps</th>
              <th>Success Rate</th>
              <th>Illegal Actions</th>
            </tr>
          </thead>
          <tbody id="bench-table-body">
            <!-- Dynamically populated via /api/metrics -->
          </tbody>
        </table>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">Reinforcement Learning Empirical Takeaways</div>
        </div>
        <p style="color: var(--text-muted); margin-bottom: 12px; font-size: 0.95rem;">
          • <strong>100% Deterministic Success</strong>: The tuned tabular Q-learning model successfully delivers the passenger across all 1,000 unseen test configurations with a mean path length of <strong>13.13 steps</strong> (beating the target limit of $\\le 14.0$).
        </p>
        <p style="color: var(--text-muted); margin-bottom: 12px; font-size: 0.95rem;">
          • <strong>Zero Illegal Moves</strong>: Across 13,130 test environment interactions, the agent attempted exactly <strong>0 illegal pickups or dropoffs</strong>.
        </p>
        <p style="color: var(--text-muted); font-size: 0.95rem;">
          • <strong>2.8x Faster Convergence with High Alpha</strong>: In deterministic transition MDPs like Taxi, high learning rates ($\alpha=0.7$) propagate Bellman targets rapidly without oscillation, achieving convergence in <strong>800 episodes</strong> compared to 2,300 for baseline ($\alpha=0.1$).
        </p>
      </div>
    </div>

    <!-- TAB 3: PUBLICATION FIGURES -->
    <div id="tab-gallery" class="tab-content">
      <div class="gallery-grid">
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V1: Multi-Seed Learning Curve</span>
            <span class="badge badge-maroon">5 Seeds ±1σ</span>
          </div>
          <img src="/figures/V1_learning_curve.png" alt="V1">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V2: Episode Length Compression</span>
            <span class="badge badge-maroon">Optimal ~13 Steps</span>
          </div>
          <img src="/figures/V2_steps_per_episode.png" alt="V2">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V3: Epsilon Exploration Decay</span>
            <span class="badge badge-maroon">Floor: 0.01</span>
          </div>
          <img src="/figures/V3_epsilon_schedule.png" alt="V3">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V4: Greedy Validation & Convergence Marker</span>
            <span class="badge badge-green">Threshold: +7.0</span>
          </div>
          <img src="/figures/V4_validation_curve.png" alt="V4">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V5: Alpha vs Gamma Convergence Heatmap</span>
            <span class="badge badge-maroon">63 Trials</span>
          </div>
          <img src="/figures/V5_tuning_heatmap.png" alt="V5">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V6: Test Reward Distribution Histogram</span>
            <span class="badge badge-green">Mean: +7.87</span>
          </div>
          <img src="/figures/V6_test_reward_distribution.png" alt="V6">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V7: Test Steps Distribution Histogram</span>
            <span class="badge badge-maroon">Median: 13</span>
          </div>
          <img src="/figures/V7_test_steps_distribution.png" alt="V7">
        </div>
        <div class="gallery-card">
          <div class="gallery-card-title">
            <span>V8: Spatial Policy Action Map</span>
            <span class="badge badge-maroon">5x5 Arrow Vectors</span>
          </div>
          <img src="/figures/V8_policy_grid.png" alt="V8">
        </div>
        <div class="gallery-card" style="grid-column: 1 / -1; max-width: 500px; margin: 0 auto;">
          <div class="gallery-card-title">
            <span>V9: Animated Replay Rollout (GIF)</span>
            <span class="badge badge-green">Greedy Agent</span>
          </div>
          <img src="/figures/V9_replay.gif" alt="V9">
        </div>
      </div>
    </div>

  </div>

  <script>
    const landmarkPositions = [
      { r: 0, c: 0, name: 'R' },
      { r: 0, c: 4, name: 'G' },
      { r: 4, c: 0, name: 'Y' },
      { r: 4, c: 3, name: 'B' }
    ];
    const actionTitles = [
      "South (↓)", "North (↑)", "East (→)", "West (←)", "Pickup (P)", "Dropoff (D)"
    ];

    let currentTrace = [];
    let currentIdx = 0;
    let isPlaying = false;
    let playInterval = null;

    function switchTab(e, tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      e.currentTarget.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
      if (tabId === 'bench') fetchBenchmarks();
    }

    function randomizeSeed() {
      document.getElementById('seed-input').value = Math.floor(10000 + Math.random() * 999);
      executeSimulation();
    }

    async function executeSimulation() {
      stopPlayback();
      const policy = document.getElementById('policy-picker').value;
      const seed = parseInt(document.getElementById('seed-input').value, 10);
      document.getElementById('route-indicator').innerText = `Seed: ${seed}`;

      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy, seed, max_steps: 30 })
      });
      const data = await res.json();
      currentTrace = data.trace;

      const slider = document.getElementById('step-slider');
      slider.max = currentTrace.length - 1;
      slider.value = 0;
      currentIdx = 0;

      updateVisualization(0);
    }

    function updateVisualization(idx) {
      if (!currentTrace || currentTrace.length === 0) return;
      const step = currentTrace[idx];
      currentIdx = idx;

      // Update Slider & Text
      document.getElementById('step-slider').value = idx;
      document.getElementById('scrubber-text').innerText = `Step ${idx} / ${currentTrace.length - 1}`;

      // Accurate Percentage Positioning on 5x5 Grid (each cell = 20%)
      const taxi = document.getElementById('taxi-token');
      taxi.style.top = `calc(${step.taxi_row * 20}% + 1.5%)`;
      taxi.style.left = `calc(${step.taxi_col * 20}% + 1.5%)`;

      // Destination Ring
      const dest = landmarkPositions[step.dest_loc];
      const destRing = document.getElementById('dest-ring');
      destRing.style.top = `${dest.r * 20}%`;
      destRing.style.left = `${dest.c * 20}%`;

      // Passenger Token
      const pass = document.getElementById('pass-token');
      if (step.pass_loc === 4) {
        // Passenger is inside taxi
        pass.style.display = 'none';
        taxi.innerHTML = '🚖<span style="font-size: 0.6rem; font-weight: 800; color: #fff; background: #600000; padding: 1px 3px; border-radius: 3px; margin-top: -4px;">PASS</span>';
        document.getElementById('disp-passenger-status').innerText = '🚖 Inside Taxi';
      } else {
        pass.style.display = 'flex';
        const pLoc = landmarkPositions[step.pass_loc];
        pass.style.top = `calc(${pLoc.r * 20}% + 2.5%)`;
        pass.style.left = `calc(${pLoc.c * 20}% + 2.5%)`;
        taxi.innerHTML = '🚖';
        document.getElementById('disp-passenger-status').innerText = `Waiting at ${pLoc.name} (${pLoc.r}, ${pLoc.c})`;
      }

      // Telemetry Cards
      document.getElementById('disp-step').innerText = `${step.step} / ${currentTrace.length - 1}`;
      document.getElementById('disp-action').innerText = `${step.action_symbol} ${step.action_name}`;
      document.getElementById('disp-reward').innerText = (step.reward > 0 ? '+' : '') + step.reward;
      document.getElementById('disp-total-reward').innerText = (step.total_reward > 0 ? '+' : '') + step.total_reward;
      document.getElementById('disp-dest-name').innerText = `${dest.name} (${dest.r}, ${dest.c})`;

      // Status Badge
      const statusBadge = document.getElementById('delivery-status-badge');
      if (step.terminated) {
        statusBadge.innerText = 'Delivered! 🎉';
        statusBadge.className = 'badge badge-green';
      } else if (step.truncated) {
        statusBadge.innerText = 'Truncated ⚠️';
        statusBadge.className = 'badge badge-maroon';
      } else {
        statusBadge.innerText = step.pass_loc === 4 ? 'En Route to Destination' : 'En Route to Passenger';
        statusBadge.className = 'badge badge-maroon';
      }

      // Render Q-Value Bars
      renderQValues(step.q_values, step.action);
    }

    function renderQValues(qVals, selectedAction) {
      const container = document.getElementById('q-value-bars');
      container.innerHTML = '';

      const minVal = Math.min(...qVals, -10);
      const maxVal = Math.max(...qVals, 15);
      const span = maxVal - minVal || 1;

      actionTitles.forEach((label, actionIdx) => {
        const val = qVals[actionIdx] !== undefined ? qVals[actionIdx] : 0;
        const pct = Math.max(4, Math.min(100, ((val - minVal) / span) * 100));
        const isChosen = (actionIdx === selectedAction);

        container.innerHTML += `
          <div class="q-value-row">
            <span class="q-action-title ${isChosen ? 'selected-action' : ''}">
              ${isChosen ? '★ ' : ''}${label}
            </span>
            <div class="q-bar-bg">
              <div class="q-bar-fill ${isChosen ? 'highlight' : ''}" style="width: ${pct}%;"></div>
            </div>
            <span class="q-num-label ${val >= 0 ? 'pos' : 'neg'}">
              ${val >= 0 ? '+' : ''}${val.toFixed(1)}
            </span>
          </div>
        `;
      });
    }

    function onSliderMove(val) {
      stopPlayback();
      updateVisualization(parseInt(val, 10));
    }

    function togglePlayPause() {
      if (isPlaying) stopPlayback();
      else startPlayback();
    }

    function startPlayback() {
      isPlaying = true;
      document.getElementById('play-pause-btn').innerText = '⏸ Pause';
      playInterval = setInterval(() => {
        if (currentIdx < currentTrace.length - 1) {
          updateVisualization(currentIdx + 1);
        } else {
          stopPlayback();
        }
      }, 350);
    }

    function stopPlayback() {
      isPlaying = false;
      document.getElementById('play-pause-btn').innerText = '▶ Play';
      if (playInterval) clearInterval(playInterval);
    }

    function stepDelta(delta) {
      stopPlayback();
      const next = currentIdx + delta;
      if (next >= 0 && next < currentTrace.length) {
        updateVisualization(next);
      }
    }

    function jumpToStep(target) {
      stopPlayback();
      if (target >= 0 && target < currentTrace.length) {
        updateVisualization(target);
      }
    }

    async function fetchBenchmarks() {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      const tbody = document.getElementById('bench-table-body');
      tbody.innerHTML = '';
      if (data.comparison) {
        data.comparison.forEach(row => {
          const isTuned = row.Policy.includes('Tuned');
          tbody.innerHTML += `
            <tr style="${isTuned ? 'background: rgba(128,0,0,0.04); font-weight: 700;' : ''}">
              <td><strong style="${isTuned ? 'color: var(--primary-maroon);' : ''}">${row.Policy}</strong></td>
              <td>${row.Episodes}</td>
              <td style="color: ${parseFloat(row['Mean Reward']) > 0 ? 'var(--green)' : 'var(--red)'}; font-weight: 700;">
                ${row['Mean Reward']}
              </td>
              <td>${row['Std Reward']}</td>
              <td><code>${row['95% CI']}</code></td>
              <td>${row['Mean Steps']}</td>
              <td><span class="badge ${parseFloat(row['Success Rate']) > 50 ? 'badge-green' : 'badge-maroon'}">${row['Success Rate']}</span></td>
              <td>${row['Illegal Actions']}</td>
            </tr>
          `;
        });
      }
    }

    // Auto-initialize simulation on load
    window.addEventListener('DOMContentLoaded', () => {
      executeSimulation();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page dashboard HTML."""
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    import uvicorn

    print("Launching Taxi-v4 Q-Learning Web Dashboard on http://127.0.0.1:8000 ...")
    uvicorn.run("src.dashboard:app", host="127.0.0.1", port=8000, reload=False)
