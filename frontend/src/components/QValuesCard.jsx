import React from 'react';

const ACTION_TITLES = [
  "South (↓)", "North (↑)", "East (→)", "West (←)", "Pickup (P)", "Dropoff (D)"
];

export default function QValuesCard({ qValues, selectedAction }) {
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
                <div
                  className={`q-bar-fill ${isChosen ? 'highlight' : ''}`}
                  style={{ width: `${pct}%` }}
                />
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
