import React, { useState, useEffect, useRef, useCallback } from 'react';
import ThreeTaxiScene from './components/ThreeTaxiScene';
import ControlsPanel from './components/ControlsPanel';
import ShortestPathCard from './components/ShortestPathCard';
import TelemetryCard from './components/TelemetryCard';
import QValuesCard from './components/QValuesCard';
import BenchmarksTab from './components/BenchmarksTab';
import FiguresGalleryTab from './components/FiguresGalleryTab';
import './styles.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('sim');
  const [policy, setPolicy] = useState('tuned');
  const [seed, setSeed] = useState(10042);
  const [trace, setTrace] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playIntervalRef = useRef(null);

  // Fetch rollout simulation
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

  // Initial load
  useEffect(() => {
    fetchSimulation();
  }, [fetchSimulation]);

  // Playback timer
  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = setInterval(() => {
        setCurrentIdx(prev => {
          if (prev < trace.length - 1) {
            return prev + 1;
          } else {
            setIsPlaying(false);
            return prev;
          }
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
      {/* HEADER */}
      <header>
        <div className="header-titles">
          <h1>🚖 Taxi-v4 Q-Learning 3D Agent Dashboard</h1>
          <p>React 18 & Three.js 3D WebGL Spatial Planning and Unbiased Q-Learning Simulation</p>
        </div>
        <div className="header-badges">
          <span className="badge badge-green">100% Success Rate</span>
          <span className="badge badge-maroon">0 Illegal Moves</span>
          <a
            href="https://canva.link/vog4wcqpdx5qgqx"
            target="_blank"
            rel="noreferrer"
            className="canva-link-btn"
          >
            🎨 View Canva Design
          </a>
        </div>
      </header>

      {/* NAVIGATION TABS */}
      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'sim' ? 'active' : ''}`}
          onClick={() => setActiveTab('sim')}
        >
          🎮 3D Interactive Simulation
        </button>
        <button
          className={`tab-btn ${activeTab === 'bench' ? 'active' : ''}`}
          onClick={() => setActiveTab('bench')}
        >
          📊 Comparative Benchmarks
        </button>
        <button
          className={`tab-btn ${activeTab === 'gallery' ? 'active' : ''}`}
          onClick={() => setActiveTab('gallery')}
        >
          🖼️ Publication Figures (V1–V9)
        </button>
      </div>

      {/* TAB CONTENT */}
      {activeTab === 'sim' && (
        <div className="sim-layout">
          {/* LEFT COLUMN: THREE.JS 3D SCENE & CONTROLS */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Three.js 3D WebGL City Grid World</div>
              <span className={`badge ${isDelivered ? 'badge-green' : 'badge-maroon'}`}>
                {isDelivered
                  ? 'Delivered! 🎉'
                  : isTruncated
                  ? 'Truncated ⚠️'
                  : currentStep && currentStep.pass_loc === 4
                  ? 'En Route to Destination'
                  : 'En Route to Passenger'}
              </span>
            </div>

            {/* 3D WebGL Arena Canvas */}
            <ThreeTaxiScene
              currentTrace={trace}
              currentStepIdx={currentIdx}
              onJumpToStep={handleJumpToStep}
            />

            {/* Simulation Controls */}
            <ControlsPanel
              policy={policy}
              setPolicy={setPolicy}
              seed={seed}
              setSeed={setSeed}
              onRandomizeSeed={handleRandomizeSeed}
              onRunRollout={() => fetchSimulation()}
              currentStep={currentIdx}
              maxSteps={Math.max(0, trace.length - 1)}
              onSliderChange={handleJumpToStep}
              isPlaying={isPlaying}
              onTogglePlayPause={() => setIsPlaying(p => !p)}
              onStepDelta={handleStepDelta}
              onJumpToStep={handleJumpToStep}
            />

            {/* Shortest Path Trajectory & Clipboard Share */}
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

          {/* RIGHT COLUMN: LIVE TELEMETRY & ACTION Q-VALUES */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <TelemetryCard
              step={currentStep}
              maxSteps={Math.max(0, trace.length - 1)}
              seed={seed}
            />

            <QValuesCard
              qValues={currentStep ? currentStep.q_values : [0, 0, 0, 0, 0, 0]}
              selectedAction={currentStep ? currentStep.action : null}
            />
          </div>
        </div>
      )}

      {activeTab === 'bench' && <BenchmarksTab />}

      {activeTab === 'gallery' && <FiguresGalleryTab />}
    </div>
  );
}
