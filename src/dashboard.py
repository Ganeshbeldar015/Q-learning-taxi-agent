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
            "action_name": "Initial",
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
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --border: #334155;
      --green: #22c55e;
      --red: #ef4444;
      --yellow: #eab308;
      --gold: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 1.6rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
    .badge {
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(34, 197, 94, 0.15);
      color: var(--green);
      border: 1px solid rgba(34, 197, 94, 0.3);
      font-weight: 600;
    }
    .tabs {
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
    }
    .tab-btn {
      background: var(--card-bg);
      color: var(--text-muted);
      border: 1px solid var(--border);
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .tab-btn:hover { color: #fff; border-color: var(--accent); }
    .tab-btn.active {
      background: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
      box-shadow: 0 0 16px var(--accent-glow);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* SIMULATION VIEW */
    .sim-grid-layout {
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 16px;
      color: #fff;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* 5x5 WORLD GRID */
    .taxi-world {
      position: relative;
      width: 400px;
      height: 400px;
      margin: 0 auto 16px auto;
      background: #090d16;
      border: 3px solid #64748b;
      border-radius: 8px;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      grid-template-rows: repeat(5, 1fr);
    }
    .cell {
      position: relative;
      border: 1px dashed rgba(100, 116, 139, 0.25);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      color: #475569;
    }
    /* Taxi Walls */
    .cell.wall-r { border-right: 4px solid #cbd5e1 !important; }
    .cell.wall-l { border-left: 4px solid #cbd5e1 !important; }

    .landmark {
      position: absolute;
      width: 60px;
      height: 60px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      font-weight: bold;
      opacity: 0.35;
      z-index: 1;
    }
    .landmark.R { top: 10px; left: 10px; background: rgba(239, 68, 68, 0.4); color: #f87171; border: 2px solid #ef4444; }
    .landmark.G { top: 10px; right: 10px; background: rgba(34, 197, 94, 0.4); color: #4ade80; border: 2px solid #22c55e; }
    .landmark.Y { bottom: 10px; left: 10px; background: rgba(234, 179, 8, 0.4); color: #fde047; border: 2px solid #eab308; }
    .landmark.B { bottom: 10px; right: 90px; background: rgba(59, 130, 246, 0.4); color: #60a5fa; border: 2px solid #3b82f6; }

    .taxi-avatar {
      position: absolute;
      width: 64px;
      height: 64px;
      background: var(--gold);
      border: 2px solid #fff;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2rem;
      z-index: 10;
      transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
      box-shadow: 0 4px 16px rgba(245, 158, 11, 0.5);
    }
    .passenger-avatar {
      position: absolute;
      font-size: 1.6rem;
      z-index: 8;
      transition: all 0.2s ease;
    }
    .dest-halo {
      position: absolute;
      width: 72px;
      height: 72px;
      border-radius: 12px;
      border: 3px dashed var(--accent);
      z-index: 2;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.6; }
      50% { transform: scale(1.08); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.6; }
    }

    /* CONTROLS */
    .controls-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .form-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    label { font-size: 0.9rem; font-weight: 600; color: var(--text-muted); min-width: 80px; }
    select, input {
      background: #0f172a;
      border: 1px solid var(--border);
      color: #fff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    .btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 9px 16px;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { opacity: 0.9; }
    .btn-secondary {
      background: #334155;
      color: #fff;
    }
    .btn-secondary:hover { background: #475569; }

    /* STATS & TELEMETRY */
    .stats-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat-box {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }
    .stat-val { font-size: 1.4rem; font-weight: 700; color: #fff; }
    .stat-lbl { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }

    /* Q-VALUE BARS */
    .q-bar-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 12px;
    }
    .q-bar-row {
      display: grid;
      grid-template-columns: 80px 1fr 60px;
      align-items: center;
      gap: 10px;
    }
    .q-track {
      height: 16px;
      background: #0f172a;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid var(--border);
      position: relative;
    }
    .q-fill {
      height: 100%;
      background: #475569;
      transition: width 0.2s ease;
    }
    .q-fill.best { background: var(--accent); }

    /* TABLE */
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }
    th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }

    /* FIGURE GALLERY */
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
    }
    .figure-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .figure-card img {
      width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
      margin-top: 8px;
      border: 1px solid var(--border);
    }
  </style>
</head>
<body>

  <header>
    <h1>🚖 Taxi-v4 Q-Learning Agent Dashboard <span class="badge">100% Success Rate</span></h1>
    <div style="font-size: 0.85rem; color: var(--text-muted);">
      Policy: <strong>Tabular Q-Learning (α=0.7, γ=0.99)</strong> | Target Destination: <strong>Optimal Routing</strong>
    </div>
  </header>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('sim')">Interactive Simulation</button>
    <button class="tab-btn" onclick="switchTab('bench')">Comparative Benchmarks</button>
    <button class="tab-btn" onclick="switchTab('figures')">Publication Figures (V1-V9)</button>
  </div>

  <!-- TAB 1: SIMULATION -->
  <div id="tab-sim" class="tab-content active">
    <div class="sim-grid-layout">
      
      <!-- LEFT: TAXI WORLD -->
      <div class="card">
        <div class="card-title">
          <span>5x5 Grid World Environment</span>
          <span id="env-status-badge" class="badge">Ready</span>
        </div>

        <div class="taxi-world" id="world">
          <!-- Landmarks -->
          <div class="landmark R">R</div>
          <div class="landmark G">G</div>
          <div class="landmark Y">Y</div>
          <div class="landmark B">B</div>

          <!-- Dynamic Elements -->
          <div id="dest-halo" class="dest-halo"></div>
          <div id="passenger" class="passenger-avatar">🧍</div>
          <div id="taxi" class="taxi-avatar">🚖</div>

          <!-- 25 Grid Cells -->
          <!-- Row 0 -->
          <div class="cell"></div><div class="cell wall-r"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div>
          <!-- Row 1 -->
          <div class="cell"></div><div class="cell wall-r"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div>
          <!-- Row 2 -->
          <div class="cell"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div>
          <!-- Row 3 -->
          <div class="cell wall-r"></div><div class="cell"></div><div class="cell wall-r"></div><div class="cell"></div><div class="cell"></div>
          <!-- Row 4 -->
          <div class="cell wall-r"></div><div class="cell"></div><div class="cell wall-r"></div><div class="cell"></div><div class="cell"></div>
        </div>

        <div class="controls-panel">
          <div class="form-group">
            <label>Policy:</label>
            <select id="policy-select">
              <option value="tuned" selected>Tuned Model (α=0.7, γ=0.99) - Fast 800 ep</option>
              <option value="baseline">Baseline Model (α=0.1, γ=0.99)</option>
              <option value="untrained">Untrained Agent (All Zeros)</option>
              <option value="random">Random Policy</option>
            </select>
          </div>
          <div class="form-group">
            <label>Seed:</label>
            <input type="number" id="seed-input" value="10042" style="width: 100px;">
            <button class="btn btn-secondary" onclick="randomSeed()">🎲 Random</button>
            <button class="btn" onclick="startSimulation()">⚡ Run Rollout</button>
          </div>
          <div style="display: flex; gap: 8px; justify-content: center; margin-top: 8px;">
            <button class="btn btn-secondary" onclick="stepBackward()">⏮ Prev</button>
            <button class="btn btn-secondary" id="play-btn" onclick="togglePlay()">▶ Play</button>
            <button class="btn btn-secondary" onclick="stepForward()">⏭ Next</button>
            <button class="btn btn-secondary" onclick="resetToStart()">🔄 Reset</button>
          </div>
        </div>
      </div>

      <!-- RIGHT: TELEMETRY & Q-VALUES -->
      <div style="display: flex; flex-direction: column; gap: 24px;">
        <div class="card">
          <div class="card-title">Live Episode Telemetry</div>
          <div class="stats-row">
            <div class="stat-box">
              <div class="stat-val" id="tele-step">0 / 0</div>
              <div class="stat-lbl">Step Count</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" id="tele-action">-</div>
              <div class="stat-lbl">Last Action</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" id="tele-reward">0</div>
              <div class="stat-lbl">Step Reward</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" id="tele-cum-reward" style="color: var(--accent);">0</div>
              <div class="stat-lbl">Total Reward</div>
            </div>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.9rem;">
            <div>Passenger: <strong id="tele-pass-status">Waiting at landmark</strong></div>
            <div>Destination: <strong id="tele-dest">G (0,4)</strong></div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Current State Action Q-Values: Q(s, a)</div>
          <div class="q-bar-group" id="q-bars">
            <!-- Dynamic Q-value bars -->
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- TAB 2: BENCHMARKS -->
  <div id="tab-bench" class="tab-content">
    <div class="card" style="margin-bottom: 24px;">
      <div class="card-title">1,000 Unseen Test Episodes Benchmark (Seeds 10000-10999)</div>
      <table id="benchmarks-table">
        <thead>
          <tr>
            <th>Policy</th>
            <th>Episodes</th>
            <th>Mean Reward</th>
            <th>Std Reward</th>
            <th>95% Bootstrap CI</th>
            <th>Mean Steps</th>
            <th>Success Rate</th>
            <th>Illegal Actions</th>
          </tr>
        </thead>
        <tbody id="benchmarks-tbody">
          <!-- Loaded via API -->
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-title">Analytical Performance Summary</div>
      <p style="color: var(--text-muted); margin-bottom: 12px;">
        The tuned agent achieves <strong>100.0% delivery success</strong> across 1,000 independent test seeds with exactly <strong>0 illegal actions</strong> and a mean path length of <strong>13.13 steps</strong> (beating the requirement of ≤14 steps).
      </p>
      <p style="color: var(--text-muted);">
        By setting learning rate α = 0.7 appropriate for deterministic MDPs, training convergence was accelerated from 2,300 episodes down to <strong>800 episodes (2.8x speedup)</strong>.
      </p>
    </div>
  </div>

  <!-- TAB 3: FIGURES -->
  <div id="tab-figures" class="tab-content">
    <div class="gallery-grid">
      <div class="figure-card">
        <strong>V1: Learning Curve (5 Seeds ±1σ Band)</strong>
        <img src="/figures/V1_learning_curve.png" alt="V1">
      </div>
      <div class="figure-card">
        <strong>V2: Episode Length Compression</strong>
        <img src="/figures/V2_steps_per_episode.png" alt="V2">
      </div>
      <div class="figure-card">
        <strong>V3: Epsilon Exploration Schedule</strong>
        <img src="/figures/V3_epsilon_schedule.png" alt="V3">
      </div>
      <div class="figure-card">
        <strong>V4: Periodic Validation & Convergence (+7.0)</strong>
        <img src="/figures/V4_validation_curve.png" alt="V4">
      </div>
      <div class="figure-card">
        <strong>V5: Hyperparameter Tuning Sweeps</strong>
        <img src="/figures/V5_tuning_heatmap.png" alt="V5">
      </div>
      <div class="figure-card">
        <strong>V6: Test Reward Distribution Histogram</strong>
        <img src="/figures/V6_test_reward_distribution.png" alt="V6">
      </div>
      <div class="figure-card">
        <strong>V7: Test Steps Distribution</strong>
        <img src="/figures/V7_test_steps_distribution.png" alt="V7">
      </div>
      <div class="figure-card">
        <strong>V8: 5x5 Spatial Policy Action Map</strong>
        <img src="/figures/V8_policy_grid.png" alt="V8">
      </div>
      <div class="figure-card" style="grid-column: span 2;">
        <strong>V9: Animated Replay Rollout Demonstration (GIF)</strong>
        <img src="/figures/V9_replay.gif" alt="V9" style="max-width: 400px; margin: 8px auto;">
      </div>
    </div>
  </div>

  <script>
    const landmarkCoords = [
      { r: 0, c: 0, name: 'R' },
      { r: 0, c: 4, name: 'G' },
      { r: 4, c: 0, name: 'Y' },
      { r: 4, c: 3, name: 'B' }
    ];
    const actionLabels = ["South (v)", "North (^)", "East (>)", "West (<)", "Pickup (P)", "Dropoff (D)"];

    let currentTrace = [];
    let currentIndex = 0;
    let isPlaying = false;
    let timer = null;

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
      if (tabId === 'bench') loadBenchmarks();
    }

    function randomSeed() {
      document.getElementById('seed-input').value = Math.floor(10000 + Math.random() * 999);
      startSimulation();
    }

    async function startSimulation() {
      pause();
      const policy = document.getElementById('policy-select').value;
      const seed = parseInt(document.getElementById('seed-input').value, 10);

      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy, seed, max_steps: 30 })
      });
      const data = await res.json();
      currentTrace = data.trace;
      currentIndex = 0;
      renderStep(0);
    }

    function renderStep(idx) {
      if (!currentTrace || currentTrace.length === 0) return;
      const step = currentTrace[idx];

      // Update Taxi Position
      const taxiEl = document.getElementById('taxi');
      taxiEl.style.top = (step.taxi_row * 80 + 8) + 'px';
      taxiEl.style.left = (step.taxi_col * 80 + 8) + 'px';

      // Update Destination Halo
      const dest = landmarkCoords[step.dest_loc];
      const halo = document.getElementById('dest-halo');
      halo.style.top = (dest.r * 80 + 4) + 'px';
      halo.style.left = (dest.c * 80 + 4) + 'px';

      // Update Passenger
      const passEl = document.getElementById('passenger');
      if (step.pass_loc === 4) {
        passEl.style.display = 'none'; // Inside Taxi
      } else {
        passEl.style.display = 'block';
        const pCoord = landmarkCoords[step.pass_loc];
        passEl.style.top = (pCoord.r * 80 + 26) + 'px';
        passEl.style.left = (pCoord.c * 80 + 26) + 'px';
      }

      // Telemetry
      document.getElementById('tele-step').innerText = `${step.step} / ${currentTrace.length - 1}`;
      document.getElementById('tele-action').innerText = step.action_name;
      document.getElementById('tele-reward').innerText = (step.reward > 0 ? '+' : '') + step.reward;
      document.getElementById('tele-cum-reward').innerText = step.total_reward;
      document.getElementById('tele-pass-status').innerText = step.pass_loc === 4 ? '🚖 Inside Taxi' : `Waiting at ${landmarkCoords[step.pass_loc].name}`;
      document.getElementById('tele-dest').innerText = `${dest.name} (${dest.r}, ${dest.c})`;

      // Badge
      const badge = document.getElementById('env-status-badge');
      if (step.terminated) {
        badge.innerText = 'Delivered! 🎉';
        badge.style.color = '#22c55e';
      } else if (step.truncated) {
        badge.innerText = 'Truncated';
        badge.style.color = '#ef4444';
      } else {
        badge.innerText = 'In Transit';
        badge.style.color = '#38bdf8';
      }

      // Q-Bars
      renderQBars(step.q_values, step.action);
    }

    function renderQBars(qVals, bestAction) {
      const container = document.getElementById('q-bars');
      container.innerHTML = '';
      const minVal = Math.min(...qVals, -10);
      const maxVal = Math.max(...qVals, 15);
      const range = maxVal - minVal || 1;

      actionLabels.forEach((name, i) => {
        const val = qVals[i] !== undefined ? qVals[i] : 0;
        const pct = Math.max(5, Math.min(100, ((val - minVal) / range) * 100));
        const isBest = (i === bestAction);

        container.innerHTML += `
          <div class="q-bar-row">
            <span style="font-size: 0.8rem; font-weight: ${isBest ? 'bold' : 'normal'}; color: ${isBest ? 'var(--accent)' : 'inherit'}">${name}</span>
            <div class="q-track">
              <div class="q-fill ${isBest ? 'best' : ''}" style="width: ${pct}%;"></div>
            </div>
            <span style="font-size: 0.8rem; text-align: right; color: ${val < 0 ? '#ef4444' : '#22c55e'}">${val.toFixed(1)}</span>
          </div>
        `;
      });
    }

    function togglePlay() {
      if (isPlaying) pause();
      else play();
    }
    function play() {
      isPlaying = true;
      document.getElementById('play-btn').innerText = '⏸ Pause';
      timer = setInterval(() => {
        if (currentIndex < currentTrace.length - 1) {
          currentIndex++;
          renderStep(currentIndex);
        } else {
          pause();
        }
      }, 350);
    }
    function pause() {
      isPlaying = false;
      document.getElementById('play-btn').innerText = '▶ Play';
      if (timer) clearInterval(timer);
    }
    function stepForward() {
      pause();
      if (currentIndex < currentTrace.length - 1) {
        currentIndex++;
        renderStep(currentIndex);
      }
    }
    function stepBackward() {
      pause();
      if (currentIndex > 0) {
        currentIndex--;
        renderStep(currentIndex);
      }
    }
    function resetToStart() {
      pause();
      currentIndex = 0;
      renderStep(0);
    }

    async function loadBenchmarks() {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      const tbody = document.getElementById('benchmarks-tbody');
      tbody.innerHTML = '';
      if (data.comparison) {
        data.comparison.forEach(r => {
          tbody.innerHTML += `
            <tr>
              <td><strong>${r.Policy}</strong></td>
              <td>${r.Episodes}</td>
              <td style="color: ${parseFloat(r['Mean Reward']) > 0 ? '#22c55e' : '#ef4444'}">${r['Mean Reward']}</td>
              <td>${r['Std Reward']}</td>
              <td>${r['95% CI']}</td>
              <td>${r['Mean Steps']}</td>
              <td><strong>${r['Success Rate']}</strong></td>
              <td>${r['Illegal Actions']}</td>
            </tr>
          `;
        });
      }
    }

    // Auto-load on boot
    window.addEventListener('load', () => {
      startSimulation();
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
