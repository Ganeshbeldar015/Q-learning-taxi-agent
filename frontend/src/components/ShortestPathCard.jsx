import React, { useState } from 'react';
import {
  FaRoute,
  FaCopy,
  FaCheck,
  FaFlagCheckered,
  FaBoxOpen,
  FaBullseye,
  FaStar,
  FaTriangleExclamation
} from 'react-icons/fa6';

const LANDMARK_NAMES = [
  { name: 'R', r: 0, c: 0 },
  { name: 'G', r: 0, c: 4 },
  { name: 'Y', r: 4, c: 0 },
  { name: 'B', r: 4, c: 3 }
];

export default function ShortestPathCard({
  trace,
  currentStepIdx,
  onJumpToStep,
  policyName,
  seed
}) {
  const [copySuccess, setCopySuccess] = useState(false);

  if (!trace || trace.length === 0) return null;

  const totalSteps = trace.length - 1;
  const finalStep = trace[totalSteps];
  const startStep = trace[0];
  const pLoc = LANDMARK_NAMES[startStep.pass_loc] || { name: '?', r: 0, c: 0 };
  const dLoc = LANDMARK_NAMES[startStep.dest_loc] || { name: '?', r: 0, c: 0 };

  const handleCopyRoute = async () => {
    let text = `Taxi-v4 Shortest Path Route Trajectory\n`;
    text += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
    text += `• Policy: ${policyName}\n`;
    text += `• Environment Seed: ${seed}\n`;
    text += `• Total Path Steps: ${totalSteps}\n`;
    text += `• Total Episode Reward: ${finalStep.total_reward >= 0 ? '+' : ''}${finalStep.total_reward}\n`;
    text += `• Passenger Pickup: ${pLoc.name} (${pLoc.r}, ${pLoc.c}) -> Destination: ${dLoc.name} (${dLoc.r}, ${dLoc.c})\n`;
    text += `• Outcome: ${finalStep.terminated ? 'SUCCESS (Delivered)' : 'INCOMPLETE'}\n\n`;
    text += `Waypoints & Actions Sequence:\n`;

    trace.forEach((s, i) => {
      if (i === 0) {
        text += `  Step  0: Taxi start at (${s.taxi_row}, ${s.taxi_col})\n`;
      } else {
        text += `  Step ${String(i).padStart(2, ' ')}: Action ${s.action_symbol} ${s.action_name.padEnd(10, ' ')} -> Taxi at (${s.taxi_row}, ${s.taxi_col}) [Reward: ${s.reward > 0 ? '+' : ''}${s.reward}]\n`;
      }
    });
    text += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2200);
    } catch (err) {
      alert("Route Trajectory:\n\n" + text);
    }
  };

  return (
    <div className="card" style={{ marginTop: '18px', padding: '16px', background: '#ffffff' }}>
      <div className="card-header" style={{ marginBottom: '12px' }}>
        <div className="card-title" style={{ fontSize: '1rem' }}>
          <FaRoute style={{ marginRight: '6px' }} /> Shortest Path Route Trajectory
        </div>
        <button
          className="btn btn-secondary"
          onClick={handleCopyRoute}
          style={{
            fontSize: '0.78rem',
            padding: '5px 12px',
            background: copySuccess ? 'var(--primary-maroon)' : '',
            color: copySuccess ? '#ffffff' : ''
          }}
        >
          {copySuccess ? (
            <>
              <FaCheck style={{ marginRight: '4px' }} /> Copied to Clipboard!
            </>
          ) : (
            <>
              <FaCopy style={{ marginRight: '4px' }} /> Share / Copy Route
            </>
          )}
        </button>
      </div>

      {/* Waypoint Pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', fontSize: '0.8rem', marginBottom: '12px', maxHeight: '120px', overflowY: 'auto' }}>
        {trace.map((step, i) => {
          let label = `${step.action_symbol} (${step.taxi_row},${step.taxi_col})`;
          let icon = null;
          let extraClass = '';
          if (i === 0) {
            label = `Start (${step.taxi_row},${step.taxi_col})`;
            icon = <FaFlagCheckered style={{ marginRight: '4px' }} />;
          } else if (step.action === 4) {
            label = `Pickup (${step.taxi_row},${step.taxi_col})`;
            icon = <FaBoxOpen style={{ marginRight: '4px' }} />;
            extraClass = 'pickup';
          } else if (step.action === 5) {
            label = `Dropoff (${step.taxi_row},${step.taxi_col})`;
            icon = <FaBullseye style={{ marginRight: '4px' }} />;
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
              {icon}
              {label}
            </div>
          );
        })}
      </div>

      {/* Path Stats */}
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border)', paddingTop: '8px' }}>
        <span>
          Total Distance: <strong style={{ color: 'var(--primary-maroon)' }}>{totalSteps} steps</strong> (Reward: {finalStep.total_reward >= 0 ? '+' : ''}{finalStep.total_reward})
        </span>
        <span className={`badge ${finalStep.terminated ? 'badge-green' : 'badge-maroon'}`}>
          {finalStep.terminated ? (
            <>
              <FaStar style={{ marginRight: '4px' }} /> 100% Shortest Path
            </>
          ) : (
            <>
              <FaTriangleExclamation style={{ marginRight: '4px' }} /> Incomplete
            </>
          )}
        </span>
      </div>
    </div>
  );
}
