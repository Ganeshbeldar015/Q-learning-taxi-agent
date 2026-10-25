# Taxi-v4 Q-Learning 3D Web Dashboard (React 18 + Three.js)

An interactive 3D WebGL dashboard built with **React 18** and **Three.js** to visualize, inspect, and benchmark the optimal tabular Q-learning policy on the Gymnasium `Taxi-v4` reinforcement learning environment.

## Key Features
- **Three.js 3D WebGL City Arena**: Full 3D isometric simulation with interactive `OrbitControls` (360° rotate, zoom, pan).
- **Procedural 3D Taxi Vehicle**: Animated 3D car with headlights, wheels, roof sign, and direction-oriented steering.
- **Shortest Path Trajectory**: Real-time 3D ribbon and waypoint spheres showing the mathematically optimal route around walls.
- **Live Q-Value Telemetry**: Dynamic bar chart showing action values $Q(s, a)$ with greedy action highlighted.
- **One-Click Trajectory Sharing**: `📋 Share / Copy Route` button copies formatted step-by-step path details directly to the clipboard.
- **White & Maroon UI Design**: Styled per the project design system with Canva MCP integration.

## Standalone Development

To run with Vite dev server:
```bash
cd frontend
npm install
npm run dev
```

Alternatively, the Python backend server (`python -m src.dashboard`) hosts the complete React + Three.js dashboard at `http://127.0.0.1:8000` with zero Node.js installation required.
