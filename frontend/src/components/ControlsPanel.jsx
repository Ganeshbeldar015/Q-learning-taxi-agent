import React from 'react';
import {
  FaDice,
  FaBolt,
  FaRotateLeft,
  FaBackwardStep,
  FaPlay,
  FaPause,
  FaForwardStep,
  FaForwardFast
} from 'react-icons/fa6';

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
  onJumpToStep,
  speedMs = 550,
  setSpeedMs
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
          <FaDice style={{ marginRight: '5px' }} /> Random
        </button>
        <button className="btn" onClick={onRunRollout}>
          <FaBolt style={{ marginRight: '5px' }} /> Run Rollout
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
        <button className="btn btn-secondary" onClick={() => onJumpToStep(0)} title="Reset to start">
          <FaRotateLeft style={{ marginRight: '4px' }} /> Reset
        </button>
        <button className="btn btn-secondary" onClick={() => onStepDelta(-1)} title="Previous step">
          <FaBackwardStep style={{ marginRight: '4px' }} /> Prev
        </button>
        <button className="btn" onClick={onTogglePlayPause} style={{ minWidth: '100px' }}>
          {isPlaying ? (
            <>
              <FaPause style={{ marginRight: '5px' }} /> Pause
            </>
          ) : (
            <>
              <FaPlay style={{ marginRight: '5px' }} /> Play
            </>
          )}
        </button>
        <button className="btn btn-secondary" onClick={() => onStepDelta(1)} title="Next step">
          Next <FaForwardStep style={{ marginLeft: '4px' }} />
        </button>
        <button className="btn btn-secondary" onClick={() => onJumpToStep(maxSteps)} title="Jump to end">
          End <FaForwardFast style={{ marginLeft: '4px' }} />
        </button>
      </div>

      {/* Travel Pace Selector */}
      {setSpeedMs && (
        <div className="control-row" style={{ marginTop: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <span className="control-label" style={{ fontSize: '0.78rem' }}>Pace:</span>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {[
              { label: 'Cinematic (0.75x)', ms: 750 },
              { label: 'Smooth (1x)', ms: 550 },
              { label: 'Fast (1.5x)', ms: 350 },
              { label: 'Rapid (2x)', ms: 200 }
            ].map((item) => (
              <button
                key={item.ms}
                type="button"
                className="btn btn-secondary"
                onClick={() => setSpeedMs(item.ms)}
                style={{
                  fontSize: '0.72rem',
                  padding: '3px 9px',
                  background: speedMs === item.ms ? 'var(--primary-maroon)' : '#ffffff',
                  color: speedMs === item.ms ? '#ffffff' : 'var(--text-main)',
                  borderColor: speedMs === item.ms ? 'var(--primary-maroon)' : 'var(--border-dark)',
                  fontWeight: speedMs === item.ms ? 700 : 500
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
