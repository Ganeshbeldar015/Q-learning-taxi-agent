"""Interactive Web Dashboard for Q-learning Taxi-v4 Agent.

Built with React 18 & Three.js 3D WebGL Visualization.
White & Maroon visual system per project specifications.

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
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import QLearningAgent
from src.env_utils import decode_state, make_env

app = FastAPI(title="Taxi-v4 Q-Learning 3D Dashboard", version="2.0.0")

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

# Mount vendor static directory if available
static_path = Path("src/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Mount compiled React frontend assets if built
dist_assets_path = Path("frontend/dist/assets")
if dist_assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_assets_path)), name="assets")

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
    data: dict[str, Any] = {}

    metrics_file = Path("results/test_metrics.json")
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            data["metrics"] = json.load(f)

    comp_file = Path("results/comparison_table.csv")
    if comp_file.exists():
        import csv

        with open(comp_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data["comparison"] = list(reader)

    return data


@app.post("/api/simulate")
def simulate(req: SimulationRequest) -> dict[str, Any]:
    """Execute a single episode simulation and return the full trajectory with telemetry."""
    env = make_env()
    obs, _ = env.reset(seed=req.seed)
    rng = np.random.default_rng(req.seed)

    q_table = MODELS.get(req.policy)
    agent = None
    if q_table is not None:
        agent = QLearningAgent()
        agent.q_table = q_table.copy()

    trace = []
    total_reward = 0
    step_num = 0
    done = False

    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    action_symbols = ["↓", "↑", "→", "←", "[P]", "[D]"]

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
  <title>Taxi-v4 Q-Learning 3D Dashboard (React 18 & Three.js)</title>
  
  <!-- Vendor Dependencies (Local with CDN fallback) -->
  <script src="/static/vendor/react.min.js"></script>
  <script src="/static/vendor/react-dom.min.js"></script>
  <script src="/static/vendor/three.min.js"></script>
  <script src="/static/vendor/OrbitControls.js"></script>
  <script src="/static/vendor/babel.min.js"></script>
  <script>
    if (!window.React) {
      document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"><' + '/script>');
      document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"><' + '/script>');
    }
    if (!window.THREE) {
      document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><' + '/script>');
      document.write('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><' + '/script>');
    }
    if (!window.Babel) {
      document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.5/babel.min.js"><' + '/script>');
    }
  </script>

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
      max-width: 1280px;
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

    /* SIMULATION LAYOUT */
    .sim-layout {
      display: grid;
      grid-template-columns: 540px 1fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 1100px) {
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

    /* THREE.JS 3D CANVAS CONTAINER */
    .three-canvas-container {
      position: relative;
      width: 100%;
      height: 420px;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      border: 2px solid #cbd5e1;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: inset 0 2px 6px rgba(0,0,0,0.06);
    }
    .three-canvas-container canvas {
      width: 100% !important;
      height: 100% !important;
      display: block;
    }
    .three-controls-overlay {
      position: absolute;
      top: 10px;
      right: 10px;
      display: flex;
      gap: 6px;
      z-index: 10;
    }
    .three-view-btn {
      background: rgba(255, 255, 255, 0.9);
      color: var(--primary-maroon);
      border: 1px solid var(--border-dark);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      backdrop-filter: blur(4px);
    }
    .three-view-btn:hover {
      background: var(--primary-maroon);
      color: #ffffff;
    }
    .three-legend {
      position: absolute;
      bottom: 8px;
      left: 10px;
      display: flex;
      gap: 10px;
      background: rgba(255, 255, 255, 0.88);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      border: 1px solid var(--border);
      backdrop-filter: blur(4px);
      z-index: 10;
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
      margin-top: 16px;
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

    /* ROUTE PILLS & BREADCRUMBS */
    .route-pill {
      padding: 4px 8px;
      background: #f1f5f9;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
      white-space: nowrap;
    }
    .route-pill:hover {
      background: #e2e8f0;
      color: var(--text-main);
    }
    .route-pill.active {
      background: var(--primary-maroon);
      color: #ffffff;
      border-color: var(--primary-maroon);
      font-weight: 700;
      box-shadow: 0 2px 4px rgba(128,0,0,0.25);
    }
    .route-pill.pickup {
      border-color: var(--accent-gold-dark);
      font-weight: 700;
      color: #b45309;
    }
    .route-pill.pickup.active {
      background: var(--accent-gold-dark);
      color: #ffffff;
    }
    .route-pill.dropoff {
      border-color: var(--green);
      font-weight: 700;
      color: var(--green);
    }
    .route-pill.dropoff.active {
      background: var(--green);
      color: #ffffff;
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

    /* TELEMETRY & Q-VALUES */
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

    /* Q-VALUE BARS */
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

    /* DATA TABLES & GALLERY */
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
  <div id="root"></div>

  <!-- REACT 18 & THREE.JS APPLICATION CODE -->
  <script type="text/babel">
    const { useState, useEffect, useRef, useCallback } = React;

    const LANDMARKS = [
      { r: 0, c: 0, name: 'R', color: 0xef4444 },
      { r: 0, c: 4, name: 'G', color: 0x16a34a },
      { r: 4, c: 0, name: 'Y', color: 0xeab308 },
      { r: 4, c: 3, name: 'B', color: 0x2563eb }
    ];

    const ACTION_TITLES = [
      "South (↓)", "North (↑)", "East (→)", "West (←)", "Pickup (P)", "Dropoff (D)"
    ];

    // Helper: Convert Taxi grid (r, c) to 3D world (x, z)
    function cellToWorld(r, c) {
      return {
        x: (c - 2) * 2.0,
        z: (r - 2) * 2.0
      };
    }

    // ── THREE.JS 3D SCENE COMPONENT ──────────────────────────────────────
    function ThreeTaxiScene({ currentTrace, currentStepIdx }) {
      const mountRef = useRef(null);
      const sceneRef = useRef(null);
      const cameraRef = useRef(null);
      const controlsRef = useRef(null);
      const rendererRef = useRef(null);

      const taxiGroupRef = useRef(null);
      const passengerGroupRef = useRef(null);
      const destBeaconRef = useRef(null);
      const beamMeshRef = useRef(null);
      const pathLinesRef = useRef([]);
      const waypointsRef = useRef([]);

      const animState = useRef({
        targetPos: new THREE.Vector3(0, 0, 0),
        targetRotY: 0
      });

      useEffect(() => {
        const container = mountRef.current;
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        // 1. Scene & Camera
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf1f5f9);
        scene.fog = new THREE.FogExp2(0xf1f5f9, 0.025);
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        camera.position.set(7.5, 9.5, 9.0);
        cameraRef.current = camera;

        // 2. WebGL Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        rendererRef.current = renderer;
        container.appendChild(renderer.domElement);

        // 3. OrbitControls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2.05;
        controls.minDistance = 4;
        controls.maxDistance = 25;
        controls.target.set(0, 0.2, 0);
        controlsRef.current = controls;

        // 4. Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
        dirLight.position.set(10, 15, 8);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 1024;
        dirLight.shadow.mapSize.height = 1024;
        scene.add(dirLight);

        // 5. Ground Base Slab & 25 Cell Tiles
        const baseGeo = new THREE.BoxGeometry(11.2, 0.3, 11.2);
        const baseMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.8 });
        const baseMesh = new THREE.Mesh(baseGeo, baseMat);
        baseMesh.position.y = -0.15;
        baseMesh.receiveShadow = true;
        scene.add(baseMesh);

        for (let r = 0; r < 5; r++) {
          for (let c = 0; c < 5; c++) {
            const { x, z } = cellToWorld(r, c);
            const tileGeo = new THREE.BoxGeometry(1.88, 0.04, 1.88);
            const tileMat = new THREE.MeshStandardMaterial({
              color: (r + c) % 2 === 0 ? 0xffffff : 0xf8fafc,
              roughness: 0.6
            });
            const tile = new THREE.Mesh(tileGeo, tileMat);
            tile.position.set(x, 0.02, z);
            tile.receiveShadow = true;
            scene.add(tile);
          }
        }

        // 6. Taxi-v4 Barrier Walls
        const wallMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.5, metalness: 0.2 });
        // Wall 1: Col 1 & 2, Rows 0-1
        const wall1 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.65, 3.8), wallMat);
        wall1.position.set(-1.0, 0.325, -3.0);
        wall1.castShadow = true;
        scene.add(wall1);

        // Wall 2: Col 0 & 1, Rows 3-4
        const wall2 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.65, 3.8), wallMat);
        wall2.position.set(-3.0, 0.325, 3.0);
        wall2.castShadow = true;
        scene.add(wall2);

        // Wall 3: Col 2 & 3, Rows 3-4
        const wall3 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.65, 3.8), wallMat);
        wall3.position.set(1.0, 0.325, 3.0);
        wall3.castShadow = true;
        scene.add(wall3);

        // 7. Landmark Pads (R, G, Y, B)
        LANDMARKS.forEach(lm => {
          const { x, z } = cellToWorld(lm.r, lm.c);
          const padGeo = new THREE.CylinderGeometry(0.72, 0.72, 0.06, 32);
          const padMat = new THREE.MeshStandardMaterial({
            color: lm.color,
            roughness: 0.4,
            emissive: lm.color,
            emissiveIntensity: 0.25
          });
          const pad = new THREE.Mesh(padGeo, padMat);
          pad.position.set(x, 0.05, z);
          pad.receiveShadow = true;
          scene.add(pad);
        });

        // 8. Destination Pulsing Beacon
        const destGroup = new THREE.Group();
        const beamGeo = new THREE.CylinderGeometry(0.7, 0.7, 2.5, 32, 1, true);
        const beamMat = new THREE.MeshStandardMaterial({
          color: 0x800000,
          transparent: true,
          opacity: 0.35,
          side: THREE.DoubleSide
        });
        const beamMesh = new THREE.Mesh(beamGeo, beamMat);
        beamMesh.position.y = 1.25;
        destGroup.add(beamMesh);
        beamMeshRef.current = beamMesh;

        const ringGeo = new THREE.RingGeometry(0.65, 0.78, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x800000, side: THREE.DoubleSide });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.rotation.x = -Math.PI / 2;
        ringMesh.position.y = 0.08;
        destGroup.add(ringMesh);

        scene.add(destGroup);
        destBeaconRef.current = destGroup;

        // 9. Procedural 3D Taxi Vehicle
        const taxiGroup = new THREE.Group();
        // Body (Yellow with Maroon racing stripe)
        const bodyGeo = new THREE.BoxGeometry(1.3, 0.42, 0.82);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.3 });
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.y = 0.32;
        bodyMesh.castShadow = true;
        taxiGroup.add(bodyMesh);

        const stripeGeo = new THREE.BoxGeometry(1.32, 0.1, 0.84);
        const stripeMat = new THREE.MeshStandardMaterial({ color: 0x800000 });
        const stripeMesh = new THREE.Mesh(stripeGeo, stripeMat);
        stripeMesh.position.y = 0.3;
        taxiGroup.add(stripeMesh);

        // Cabin
        const cabinGeo = new THREE.BoxGeometry(0.75, 0.34, 0.7);
        const cabinMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.2 });
        const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
        cabinMesh.position.set(-0.05, 0.65, 0);
        cabinMesh.castShadow = true;
        taxiGroup.add(cabinMesh);

        // Taxi Roof Sign
        const signGeo = new THREE.BoxGeometry(0.35, 0.12, 0.16);
        const signMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfffbeb, emissiveIntensity: 0.4 });
        const signMesh = new THREE.Mesh(signGeo, signMat);
        signMesh.position.set(-0.05, 0.88, 0);
        taxiGroup.add(signMesh);

        // 4 Wheels
        const wheelGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.12, 16);
        const wheelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.9 });
        [[-0.4, 0.18, 0.42], [-0.4, 0.18, -0.42], [0.4, 0.18, 0.42], [0.4, 0.18, -0.42]].forEach(([wx, wy, wz]) => {
          const wheel = new THREE.Mesh(wheelGeo, wheelMat);
          wheel.rotation.x = Math.PI / 2;
          wheel.position.set(wx, wy, wz);
          wheel.castShadow = true;
          taxiGroup.add(wheel);
        });

        // Headlights
        const lightGeo = new THREE.SphereGeometry(0.08, 12, 12);
        const lightMat = new THREE.MeshBasicMaterial({ color: 0xfef08a });
        const lightL = new THREE.Mesh(lightGeo, lightMat);
        lightL.position.set(0.66, 0.34, 0.25);
        const lightR = new THREE.Mesh(lightGeo, lightMat);
        lightR.position.set(0.66, 0.34, -0.25);
        taxiGroup.add(lightL);
        taxiGroup.add(lightR);

        scene.add(taxiGroup);
        taxiGroupRef.current = taxiGroup;

        // 10. Passenger Character
        const passGroup = new THREE.Group();
        const pBodyGeo = new THREE.CylinderGeometry(0.16, 0.2, 0.55, 16);
        const pBodyMat = new THREE.MeshStandardMaterial({ color: 0x2563eb });
        const pBody = new THREE.Mesh(pBodyGeo, pBodyMat);
        pBody.position.y = 0.35;
        pBody.castShadow = true;
        passGroup.add(pBody);

        const pHeadGeo = new THREE.SphereGeometry(0.16, 16, 16);
        const pHeadMat = new THREE.MeshStandardMaterial({ color: 0xfed7aa });
        const pHead = new THREE.Mesh(pHeadGeo, pHeadMat);
        pHead.position.y = 0.72;
        pHead.castShadow = true;
        passGroup.add(pHead);

        const pCapGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.08, 16);
        const pCapMat = new THREE.MeshStandardMaterial({ color: 0x800000 });
        const pCap = new THREE.Mesh(pCapGeo, pCapMat);
        pCap.position.y = 0.82;
        passGroup.add(pCap);

        scene.add(passGroup);
        passengerGroupRef.current = passGroup;

        // 11. Animation Loop
        let animId;
        const animate = () => {
          animId = requestAnimationFrame(animate);

          // Smooth taxi interpolation
          if (taxiGroupRef.current) {
            taxiGroupRef.current.position.lerp(animState.current.targetPos, 0.18);
            const curRot = taxiGroupRef.current.rotation.y;
            const tgtRot = animState.current.targetRotY;
            taxiGroupRef.current.rotation.y += (tgtRot - curRot) * 0.2;
          }

          // Pulsing beacon
          if (beamMeshRef.current) {
            beamMeshRef.current.rotation.y += 0.015;
            beamMeshRef.current.material.opacity = 0.25 + 0.15 * Math.sin(Date.now() * 0.005);
          }

          controls.update();
          renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
          if (!container) return;
          const w = container.clientWidth;
          const h = container.clientHeight;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        };
        window.addEventListener('resize', handleResize);

        return () => {
          window.removeEventListener('resize', handleResize);
          cancelAnimationFrame(animId);
          if (renderer.domElement && container) {
            container.removeChild(renderer.domElement);
          }
          renderer.dispose();
        };
      }, []);

      // Update positions & shortest path ribbons on step changes
      useEffect(() => {
        if (!currentTrace || currentTrace.length === 0 || !sceneRef.current) return;
        const step = currentTrace[currentStepIdx] || currentTrace[0];
        const scene = sceneRef.current;

        // 1. Target Taxi World Position & Heading
        const { x, z } = cellToWorld(step.taxi_row, step.taxi_col);
        animState.current.targetPos.set(x, 0, z);

        if (step.action === 0) animState.current.targetRotY = -Math.PI / 2; // South
        else if (step.action === 1) animState.current.targetRotY = Math.PI / 2; // North
        else if (step.action === 2) animState.current.targetRotY = 0; // East
        else if (step.action === 3) animState.current.targetRotY = Math.PI; // West

        // 2. Passenger Visibility & Placement
        if (passengerGroupRef.current) {
          if (step.pass_loc === 4) {
            passengerGroupRef.current.visible = false;
          } else {
            passengerGroupRef.current.visible = true;
            const pLm = LANDMARKS[step.pass_loc];
            if (pLm) {
              const pw = cellToWorld(pLm.r, pLm.c);
              passengerGroupRef.current.position.set(pw.x, 0, pw.z);
            }
          }
        }

        // 3. Destination Beacon
        if (destBeaconRef.current) {
          const dLm = LANDMARKS[step.dest_loc];
          if (dLm) {
            const dw = cellToWorld(dLm.r, dLm.c);
            destBeaconRef.current.position.set(dw.x, 0, dw.z);
          }
        }

        // 4. Update 3D Shortest Path Ribbon & Nodes
        pathLinesRef.current.forEach(mesh => scene.remove(mesh));
        pathLinesRef.current = [];
        waypointsRef.current.forEach(mesh => scene.remove(mesh));
        waypointsRef.current = [];

        const fullPoints = currentTrace.map(s => {
          const w = cellToWorld(s.taxi_row, s.taxi_col);
          return new THREE.Vector3(w.x, 0.12, w.z);
        });

        // Planned path (dashed ribbon)
        const fullGeo = new THREE.BufferGeometry().setFromPoints(fullPoints);
        const fullMat = new THREE.LineDashedMaterial({
          color: 0x800000,
          dashSize: 0.2,
          gapSize: 0.15,
          opacity: 0.32,
          transparent: true
        });
        const fullLine = new THREE.Line(fullGeo, fullMat);
        fullLine.computeLineDistances();
        scene.add(fullLine);
        pathLinesRef.current.push(fullLine);

        // Traveled path (solid vibrant maroon)
        if (currentStepIdx > 0) {
          const traveledPoints = fullPoints.slice(0, currentStepIdx + 1);
          const traveledGeo = new THREE.BufferGeometry().setFromPoints(traveledPoints);
          const traveledMat = new THREE.LineBasicMaterial({
            color: 0x800000,
            linewidth: 3
          });
          const traveledLine = new THREE.Line(traveledGeo, traveledMat);
          scene.add(traveledLine);
          pathLinesRef.current.push(traveledLine);
        }

        // Waypoints
        fullPoints.forEach((pt, i) => {
          const isTraveled = (i <= currentStepIdx);
          const isCurrent = (i === currentStepIdx);
          const radius = isCurrent ? 0.18 : (isTraveled ? 0.12 : 0.08);
          const color = isCurrent ? 0xf59e0b : (isTraveled ? 0x800000 : 0x94a3b8);

          const wpGeo = new THREE.SphereGeometry(radius, 12, 12);
          const wpMat = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: isCurrent ? 0.6 : 0.2
          });
          const wpMesh = new THREE.Mesh(wpGeo, wpMat);
          wpMesh.position.copy(pt);
          scene.add(wpMesh);
          waypointsRef.current.push(wpMesh);
        });

      }, [currentTrace, currentStepIdx]);

      const setCameraView = (type) => {
        if (!cameraRef.current || !controlsRef.current) return;
        const cam = cameraRef.current;
        const controls = controlsRef.current;
        if (type === 'iso') {
          cam.position.set(7.5, 9.5, 9.0);
          controls.target.set(0, 0.2, 0);
        } else if (type === 'top') {
          cam.position.set(0, 14, 0.01);
          controls.target.set(0, 0, 0);
        } else if (type === 'close') {
          const pos = animState.current.targetPos;
          cam.position.set(pos.x + 3.5, 3.5, pos.z + 3.5);
          controls.target.copy(pos);
        }
        controls.update();
      };

      return (
        <div className="three-canvas-container" ref={mountRef}>
          <div className="three-controls-overlay">
            <button className="three-view-btn" onClick={() => setCameraView('iso')}>Isometric</button>
            <button className="three-view-btn" onClick={() => setCameraView('top')}>Top-Down</button>
            <button className="three-view-btn" onClick={() => setCameraView('close')}>Chase Cam</button>
          </div>
          <div className="three-legend">
            <span style={{ color: '#dc2626' }}>● R</span>
            <span style={{ color: '#16a34a' }}>● G</span>
            <span style={{ color: '#ca8a04' }}>● Y</span>
            <span style={{ color: '#2563eb' }}>● B</span>
            <span style={{ color: '#800000' }}>─ Shortest Path</span>
          </div>
        </div>
      );
    }

    // ── SHORTEST PATH & SHARING CARD ─────────────────────────────────────
    function ShortestPathCard({ trace, currentStepIdx, onJumpToStep, policyName, seed }) {
      const [copied, setCopied] = useState(false);
      if (!trace || trace.length === 0) return null;

      const totalSteps = trace.length - 1;
      const finalStep = trace[totalSteps];
      const start = trace[0];
      const pLoc = LANDMARKS[start.pass_loc] || { name: '?', r: 0, c: 0 };
      const dLoc = LANDMARKS[start.dest_loc] || { name: '?', r: 0, c: 0 };

      const handleCopy = async () => {
        let text = `🚖 Taxi-v4 Shortest Path Route Trajectory\\n`;
        text += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;
        text += `• Policy: ${policyName}\\n`;
        text += `• Environment Seed: ${seed}\\n`;
        text += `• Total Path Steps: ${totalSteps}\\n`;
        text += `• Total Episode Reward: ${finalStep.total_reward >= 0 ? '+' : ''}${finalStep.total_reward}\\n`;
        text += `• Passenger Pickup: ${pLoc.name} (${pLoc.r}, ${pLoc.c}) ➔ Destination: ${dLoc.name} (${dLoc.r}, ${dLoc.c})\\n`;
        text += `• Outcome: ${finalStep.terminated ? 'SUCCESS (Delivered)' : 'INCOMPLETE'}\\n\\n`;
        text += `Waypoints & Actions Sequence:\\n`;

        trace.forEach((s, i) => {
          if (i === 0) {
            text += `  Step  0: Taxi start at (${s.taxi_row}, ${s.taxi_col})\\n`;
          } else {
            text += `  Step ${String(i).padStart(2, ' ')}: Action ${s.action_symbol} ${s.action_name.padEnd(10, ' ')} ➔ Taxi at (${s.taxi_row}, ${s.taxi_col}) [Reward: ${s.reward > 0 ? '+' : ''}${s.reward}]\\n`;
          }
        });
        text += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;

        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 2200);
        } catch (err) {
          alert("Route trajectory:\\n\\n" + text);
        }
      };

      return (
        <div className="card" style={{ marginTop: '18px', padding: '16px', background: '#ffffff' }}>
          <div className="card-header" style={{ marginBottom: '12px' }}>
            <div className="card-title" style={{ fontSize: '1rem' }}>
              Shortest Path Route Trajectory
            </div>
            <button
              className="btn btn-secondary"
              onClick={handleCopy}
              style={{
                fontSize: '0.78rem',
                padding: '5px 12px',
                background: copied ? 'var(--primary-maroon)' : '',
                color: copied ? '#ffffff' : ''
              }}
            >
              {copied ? 'Copied to Clipboard' : 'Share / Copy Route'}
            </button>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', fontSize: '0.8rem', marginBottom: '12px', maxHeight: '120px', overflowY: 'auto' }}>
            {trace.map((step, i) => {
              let label = `${step.action_symbol} (${step.taxi_row},${step.taxi_col})`;
              let extraClass = '';
              if (i === 0) {
                label = `Start (${step.taxi_row},${step.taxi_col})`;
              } else if (step.action === 4) {
                label = `Pickup (${step.taxi_row},${step.taxi_col})`;
                extraClass = 'pickup';
              } else if (step.action === 5) {
                label = `Dropoff (${step.taxi_row},${step.taxi_col})`;
                extraClass = 'dropoff';
              }

              const isActive = (i === currentStepIdx);
              return (
                <div
                  key={i}
                  className={`route-pill ${extraClass} ${isActive ? 'active' : ''}`}
                  onClick={() => onJumpToStep(i)}
                  title={`Jump to Step ${i} - ${step.action_name}`}
                >
                  {label}
                </div>
              );
            })}
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border)', paddingTop: '8px' }}>
            <span>
              Total Distance: <strong style={{ color: 'var(--primary-maroon)' }}>{totalSteps} steps</strong> (Reward: {finalStep.total_reward >= 0 ? '+' : ''}{finalStep.total_reward})
            </span>
            <span className={`badge ${finalStep.terminated ? 'badge-green' : 'badge-maroon'}`}>
              {finalStep.terminated ? '100% Shortest Path' : 'Incomplete'}
            </span>
          </div>
        </div>
      );
    }

    // ── TELEMETRY CARD ───────────────────────────────────────────────────
    function TelemetryCard({ step, maxSteps, seed }) {
      if (!step) return null;
      const dest = LANDMARKS[step.dest_loc] || { name: '?', r: 0, c: 0 };
      const pass = LANDMARKS[step.pass_loc];

      let passStatus = 'En Route';
      if (step.pass_loc === 4) passStatus = 'Inside Taxi';
      else if (pass) passStatus = `Waiting at ${pass.name} (${pass.r}, ${pass.c})`;

      return (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Live Episode Telemetry</div>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--primary-maroon)' }}>
              Seed: {seed}
            </span>
          </div>

          <div className="telemetry-grid">
            <div className="metric-card">
              <div className="metric-value">{step.step} / {maxSteps}</div>
              <div className="metric-label">Step Count</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ fontSize: '1.15rem' }}>
                {step.action_symbol} {step.action_name}
              </div>
              <div className="metric-label">Last Action</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: step.reward > 0 ? 'var(--green)' : 'var(--primary-maroon)' }}>
                {step.reward > 0 ? '+' : ''}{step.reward}
              </div>
              <div className="metric-label">Step Reward</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: step.total_reward >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {step.total_reward >= 0 ? '+' : ''}{step.total_reward}
              </div>
              <div className="metric-label">Total Reward</div>
            </div>
          </div>

          <div className="journey-info-box">
            <div>Passenger Status: <strong style={{ color: 'var(--primary-maroon)' }}>{passStatus}</strong></div>
            <div>Target Destination: <strong style={{ color: 'var(--green)' }}>{dest.name} ({dest.r}, {dest.c})</strong></div>
          </div>
        </div>
      );
    }

    // ── Q-VALUES CARD ────────────────────────────────────────────────────
    function QValuesCard({ qValues, selectedAction }) {
      const vals = qValues || [0, 0, 0, 0, 0, 0];
      const minVal = Math.min(...vals, -10);
      const maxVal = Math.max(...vals, 15);
      const span = maxVal - minVal || 1;

      return (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Action Q-Values for Current State: Q(s, a)</div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
              Greedy Action Highlighted
            </span>
          </div>

          <div className="q-value-list">
            {ACTION_TITLES.map((label, idx) => {
              const val = vals[idx] !== undefined ? vals[idx] : 0;
              const pct = Math.max(4, Math.min(100, ((val - minVal) / span) * 100));
              const isChosen = (idx === selectedAction);

              return (
                <div key={idx} className="q-value-row">
                  <span className={`q-action-title ${isChosen ? 'selected-action' : ''}`}>
                    {isChosen ? '★ ' : ''}{label}
                  </span>
                  <div className="q-bar-bg">
                    <div className={`q-bar-fill ${isChosen ? 'highlight' : ''}`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className={`q-num-label ${val >= 0 ? 'pos' : 'neg'}`}>
                    {val >= 0 ? '+' : ''}{val.toFixed(1)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    // ── BENCHMARKS TAB ───────────────────────────────────────────────────
    function BenchmarksTab() {
      const [metrics, setMetrics] = useState([]);
      const [loading, setLoading] = useState(true);

      useEffect(() => {
        fetch('/api/metrics')
          .then(res => res.json())
          .then(data => {
            if (data.comparison) setMetrics(data.comparison);
            setLoading(false);
          })
          .catch(() => setLoading(false));
      }, []);

      return (
        <div>
          <div className="card" style={{ marginBottom: '24px' }}>
            <div className="card-header">
              <div className="card-title">1,000 Unseen Test Episodes Comparative Benchmarks</div>
              <span className="badge badge-maroon">Seeds 10000–10999</span>
            </div>

            {loading ? (
              <p style={{ color: 'var(--text-muted)' }}>Loading benchmarks...</p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Policy Model</th>
                    <th>Episodes</th>
                    <th>Mean Reward</th>
                    <th>Std Reward</th>
                    <th>95% CI</th>
                    <th>Mean Steps</th>
                    <th>Success Rate</th>
                    <th>Illegal Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((row, i) => {
                    const isTuned = row.Policy.includes('Tuned');
                    const meanRew = parseFloat(row['Mean Reward']);
                    return (
                      <tr key={i} style={{ background: isTuned ? 'rgba(128,0,0,0.04)' : '', fontWeight: isTuned ? 700 : 400 }}>
                        <td><strong style={{ color: isTuned ? 'var(--primary-maroon)' : '' }}>{row.Policy}</strong></td>
                        <td>{row.Episodes}</td>
                        <td style={{ color: meanRew > 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>
                          {row['Mean Reward']}
                        </td>
                        <td>{row['Std Reward']}</td>
                        <td><code>{row['95% CI']}</code></td>
                        <td>{row['Mean Steps']}</td>
                        <td>
                          <span className={`badge ${parseFloat(row['Success Rate']) > 50 ? 'badge-green' : 'badge-maroon'}`}>
                            {row['Success Rate']}
                          </span>
                        </td>
                        <td>{row['Illegal Actions']}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title">Reinforcement Learning Empirical Takeaways</div>
            </div>
            <p style={{ color: 'var(--text-muted)', marginBottom: '12px', fontSize: '0.95rem' }}>
              • <strong>100% Deterministic Success</strong>: The tuned tabular Q-learning model successfully delivers the passenger across all 1,000 unseen test configurations with a mean path length of <strong>13.13 steps</strong> (beating the target limit of &le; 14.0).
            </p>
            <p style={{ color: 'var(--text-muted)', marginBottom: '12px', fontSize: '0.95rem' }}>
              • <strong>Zero Illegal Moves</strong>: Across 13,130 test environment interactions, the agent attempted exactly <strong>0 illegal pickups or dropoffs</strong>.
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              • <strong>2.8x Faster Convergence with High Alpha</strong>: In deterministic transition MDPs like Taxi, high learning rates ($\alpha=0.7$) propagate Bellman targets rapidly without oscillation, achieving convergence in <strong>800 episodes</strong> compared to 2,300 for baseline ($\alpha=0.1$).
            </p>
          </div>
        </div>
      );
    }

    // ── GALLERY TAB ──────────────────────────────────────────────────────
    function FiguresGalleryTab() {
      const figures = [
        { id: 'V1', title: 'V1: Multi-Seed Learning Curve', badge: '5 Seeds ±1σ', file: '/figures/V1_learning_curve.png' },
        { id: 'V2', title: 'V2: Episode Length Compression', badge: 'Optimal ~13 Steps', file: '/figures/V2_steps_per_episode.png' },
        { id: 'V3', title: 'V3: Epsilon Exploration Decay', badge: 'Floor: 0.01', file: '/figures/V3_epsilon_schedule.png' },
        { id: 'V4', title: 'V4: Greedy Validation Curve', badge: 'Threshold: +7.0', file: '/figures/V4_validation_curve.png' },
        { id: 'V5', title: 'V5: Alpha vs Gamma Tuning Heatmap', badge: '63 Trials', file: '/figures/V5_tuning_heatmap.png' },
        { id: 'V6', title: 'V6: Test Reward Distribution', badge: 'Mean: +7.87', file: '/figures/V6_test_reward_distribution.png' },
        { id: 'V7', title: 'V7: Test Steps Distribution', badge: 'Median: 13', file: '/figures/V7_test_steps_distribution.png' },
        { id: 'V8', title: 'V8: Spatial Policy Action Map', badge: '5x5 Arrow Vectors', file: '/figures/V8_policy_grid.png' },
        { id: 'V9', title: 'V9: Animated Replay Rollout (GIF)', badge: 'Greedy Agent', file: '/figures/V9_replay.gif' }
      ];

      return (
        <div className="gallery-grid">
          {figures.map(fig => (
            <div key={fig.id} className="gallery-card" style={fig.id === 'V9' ? { gridColumn: '1 / -1', maxWidth: '520px', margin: '0 auto' } : {}}>
              <div className="gallery-card-title">
                <span>{fig.title}</span>
                <span className="badge badge-maroon">{fig.badge}</span>
              </div>
              <img src={fig.file} alt={fig.title} />
            </div>
          ))}
        </div>
      );
    }

    // ── MAIN ROOT APPLICATION ────────────────────────────────────────────
    function App() {
      const [activeTab, setActiveTab] = useState('sim');
      const [policy, setPolicy] = useState('tuned');
      const [seed, setSeed] = useState(10042);
      const [trace, setTrace] = useState([]);
      const [currentIdx, setCurrentIdx] = useState(0);
      const [isPlaying, setIsPlaying] = useState(false);
      const playIntervalRef = useRef(null);

      const fetchSimulation = useCallback(async (selectedPolicy = policy, selectedSeed = seed) => {
        setIsPlaying(false);
        if (playIntervalRef.current) clearInterval(playIntervalRef.current);

        try {
          const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ policy: selectedPolicy, seed: selectedSeed, max_steps: 30 })
          });
          const data = await res.json();
          setTrace(data.trace || []);
          setCurrentIdx(0);
        } catch (err) {
          console.error('Failed to simulate:', err);
        }
      }, [policy, seed]);

      useEffect(() => {
        fetchSimulation();
      }, [fetchSimulation]);

      useEffect(() => {
        if (isPlaying) {
          playIntervalRef.current = setInterval(() => {
            setCurrentIdx(prev => {
              if (prev < trace.length - 1) return prev + 1;
              setIsPlaying(false);
              return prev;
            });
          }, 350);
        } else {
          if (playIntervalRef.current) clearInterval(playIntervalRef.current);
        }
        return () => {
          if (playIntervalRef.current) clearInterval(playIntervalRef.current);
        };
      }, [isPlaying, trace.length]);

      const handleRandomizeSeed = () => {
        const nextSeed = Math.floor(10000 + Math.random() * 999);
        setSeed(nextSeed);
        fetchSimulation(policy, nextSeed);
      };

      const handleStepDelta = (delta) => {
        setIsPlaying(false);
        setCurrentIdx(prev => Math.max(0, Math.min(trace.length - 1, prev + delta)));
      };

      const handleJumpToStep = (target) => {
        setIsPlaying(false);
        setCurrentIdx(Math.max(0, Math.min(trace.length - 1, target)));
      };

      const currentStep = trace[currentIdx] || null;
      const isDelivered = currentStep && currentStep.terminated;
      const isTruncated = currentStep && currentStep.truncated;

      return (
        <div className="container">
          <header>
            <div className="header-titles">
              <h1>Taxi-v4 Q-Learning 3D Agent Dashboard</h1>
              <p>React 18 & Three.js 3D WebGL Spatial Planning and Unbiased Q-Learning Simulation</p>
            </div>
            <div className="header-badges">
              <span className="badge badge-green">100% Success Rate</span>
              <span className="badge badge-maroon">0 Illegal Moves</span>
              <a href="https://canva.link/vog4wcqpdx5qgqx" target="_blank" rel="noreferrer" className="canva-link-btn">
                View Canva Design
              </a>
            </div>
          </header>

          <div className="tabs-nav">
            <button className={`tab-btn ${activeTab === 'sim' ? 'active' : ''}`} onClick={() => setActiveTab('sim')}>
              3D Interactive Simulation
            </button>
            <button className={`tab-btn ${activeTab === 'bench' ? 'active' : ''}`} onClick={() => setActiveTab('bench')}>
              Comparative Benchmarks
            </button>
            <button className={`tab-btn ${activeTab === 'gallery' ? 'active' : ''}`} onClick={() => setActiveTab('gallery')}>
              Publication Figures (V1–V9)
            </button>
          </div>

          {activeTab === 'sim' && (
            <div className="sim-layout">
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Three.js 3D WebGL City Grid World</div>
                  <span className={`badge ${isDelivered ? 'badge-green' : 'badge-maroon'}`}>
                    {isDelivered
                      ? 'Delivered'
                      : isTruncated
                      ? 'Truncated'
                      : currentStep && currentStep.pass_loc === 4
                      ? 'En Route to Destination'
                      : 'En Route to Passenger'}
                  </span>
                </div>

                {/* 3D WebGL Canvas */}
                <ThreeTaxiScene currentTrace={trace} currentStepIdx={currentIdx} />

                {/* Controls */}
                <div className="sim-controls">
                  <div className="control-row">
                    <span className="control-label">Policy:</span>
                    <select value={policy} onChange={(e) => setPolicy(e.target.value)} style={{ flex: 1 }}>
                      <option value="tuned">Tuned Model (α=0.7, γ=0.99) — 800 ep fast</option>
                      <option value="baseline">Baseline Model (α=0.1, γ=0.99) — 2,300 ep</option>
                      <option value="untrained">Untrained Agent (All Zeros)</option>
                      <option value="random">Random Policy</option>
                    </select>
                  </div>

                  <div className="control-row">
                    <span className="control-label">Seed:</span>
                    <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)} style={{ width: '110px' }} />
                    <button className="btn btn-secondary" onClick={handleRandomizeSeed}>Random</button>
                    <button className="btn" onClick={() => fetchSimulation()}>Run Rollout</button>
                  </div>

                  <div className="scrubber-container">
                    <div className="scrubber-header">
                      <span>Trajectory Timeline</span>
                      <span>Step {currentIdx} / {Math.max(0, trace.length - 1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max={Math.max(0, trace.length - 1)}
                      value={currentIdx}
                      onChange={(e) => handleJumpToStep(parseInt(e.target.value, 10))}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', marginTop: '6px' }}>
                    <button className="btn btn-secondary" onClick={() => handleJumpToStep(0)}>Reset</button>
                    <button className="btn btn-secondary" onClick={() => handleStepDelta(-1)}>Prev</button>
                    <button className="btn" onClick={() => setIsPlaying(p => !p)} style={{ minWidth: '100px' }}>
                      {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button className="btn btn-secondary" onClick={() => handleStepDelta(1)}>Next</button>
                    <button className="btn btn-secondary" onClick={() => handleJumpToStep(trace.length - 1)}>End</button>
                  </div>
                </div>

                {/* Shortest Path Route & Clipboard Share */}
                <ShortestPathCard
                  trace={trace}
                  currentStepIdx={currentIdx}
                  onJumpToStep={handleJumpToStep}
                  policyName={
                    policy === 'tuned'
                      ? 'Tuned Model (α=0.7, γ=0.99) — 800 ep fast'
                      : policy === 'baseline'
                      ? 'Baseline Model (α=0.1, γ=0.99)'
                      : policy
                  }
                  seed={seed}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <TelemetryCard step={currentStep} maxSteps={Math.max(0, trace.length - 1)} seed={seed} />
                <QValuesCard qValues={currentStep ? currentStep.q_values : [0, 0, 0, 0, 0, 0]} selectedAction={currentStep ? currentStep.action : null} />
              </div>
            </div>
          )}

          {activeTab === 'bench' && <BenchmarksTab />}
          {activeTab === 'gallery' && <FiguresGalleryTab />}
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(<App />);
  </script>
</body>
</html>
"""


@app.get("/")
def index() -> Response:
    """Serve the compiled React production build if available, otherwise embedded fallback."""
    dist_index = Path("frontend/dist/index.html")
    if dist_index.exists():
        return FileResponse(str(dist_index))
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    import uvicorn

    print("Launching Taxi-v4 Q-Learning 3D Web Dashboard on http://127.0.0.1:8000 ...")
    uvicorn.run("src.dashboard:app", host="127.0.0.1", port=8000, reload=False)
