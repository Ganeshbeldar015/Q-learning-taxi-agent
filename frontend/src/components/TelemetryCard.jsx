import React from 'react';
import { FaTaxi, FaBullseye, FaUserCheck, FaLocationDot } from 'react-icons/fa6';

const LANDMARKS = [
  { name: 'R', r: 0, c: 0 },
  { name: 'G', r: 0, c: 4 },
  { name: 'Y', r: 4, c: 0 },
  { name: 'B', r: 4, c: 3 }
];

export default function TelemetryCard({ step, maxSteps, seed }) {
  if (!step) return null;

  const dest = LANDMARKS[step.dest_loc] || { name: '?', r: 0, c: 0 };
  const pass = LANDMARKS[step.pass_loc];

  let passStatus = <span>En Route</span>;
  if (step.pass_loc === 4) {
    passStatus = (
      <span>
        <FaTaxi style={{ marginRight: '5px', verticalAlign: 'middle' }} /> Inside Taxi
      </span>
    );
  } else if (pass) {
    passStatus = (
      <span>
        <FaUserCheck style={{ marginRight: '5px', verticalAlign: 'middle' }} />
        Waiting at {pass.name} ({pass.r}, {pass.c})
      </span>
    );
  }

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
        <div>
          Passenger Status: <strong style={{ color: 'var(--primary-maroon)' }}>{passStatus}</strong>
        </div>
        <div>
          Target Destination: <strong style={{ color: 'var(--green)' }}>
            <FaBullseye style={{ marginRight: '4px', verticalAlign: 'middle' }} />
            {dest.name} ({dest.r}, {dest.c})
          </strong>
        </div>
      </div>
    </div>
  );
}
