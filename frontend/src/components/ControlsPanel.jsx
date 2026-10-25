import React from 'react';

export default function ControlsPanel({
  policy,
  setPolicy,
  seed,
  setSeed,
  onRandomizeSeed,
  onRunRollout,
  currentStep,
  maxSteps,
  onSliderChange,
  isPlaying,
  onTogglePlayPause,
  onStepDelta,
  onJumpToStep
}) {
  return (
    <div className="sim-controls">
      {/* Policy Selection */}
      <div className="control-row">
        <span className="control-label">Policy:</span>
        <select
          value={policy}
          onChange={(e) => setPolicy(e.target.value)}
          style={{ flex: 1 }}
        >
          <option value="tuned">Tuned Model (α=0.7, γ=0.99) — 800 ep fast</option>
          <option value="baseline">Baseline Model (α=0.1, γ=0.99) — 2,300 ep</option>
          <option value="untrained">Untrained Agent (All Zeros)</option>
          <option value="random">Random Policy</option>
        </select>
      </div>

      {/* Seed Selection */}
      <div className="control-row">
        <span className="control-label">Seed:</span>
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
          style={{ width: '110px' }}
        />
        <button className="btn btn-secondary" onClick={onRandomizeSeed}>
          🎲 Random
        </button>
        <button className="btn" onClick={onRunRollout}>
          ⚡ Run Rollout
        </button>
      </div>

      {/* Scrubber Slider */}
      <div className="scrubber-container">
        <div className="scrubber-header">
          <span>Trajectory Timeline</span>
          <span>Step {currentStep} / {maxSteps}</span>
        </div>
        <input
          type="range"
          min="0"
          max={maxSteps}
          value={currentStep}
          onChange={(e) => onSliderChange(parseInt(e.target.value, 10))}
        />
      </div>

      {/* Playback Controls */}
      <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', marginTop: '6px' }}>
        <button className="btn btn-secondary" onClick={() => onJumpToStep(0)}>
          ⏮ Reset
        </button>
        <button className="btn btn-secondary" onClick={() => onStepDelta(-1)}>
          ◀ Prev
        </button>
        <button className="btn" onClick={onTogglePlayPause} style={{ minWidth: '100px' }}>
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <button className="btn btn-secondary" onClick={() => onStepDelta(1)}>
          Next ▶
        </button>
        <button className="btn btn-secondary" onClick={() => onJumpToStep(maxSteps)}>
          End ⏭
        </button>
      </div>
    </div>
  );
}
