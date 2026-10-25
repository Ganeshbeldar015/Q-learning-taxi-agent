import React from 'react';

const FIGURES = [
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

export default function FiguresGalleryTab() {
  return (
    <div className="gallery-grid">
      {FIGURES.map(fig => (
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
