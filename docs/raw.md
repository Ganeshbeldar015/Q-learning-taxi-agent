PRD: Taxi Route Optimization with Reinforcement Learning
Problem Statement 12 · Q-learning on Gymnasium Taxi-v3 · Version 1.0 · October 25, 2026
1. Overview
Product: An end-to-end reinforcement learning project in which a tabular Q-learning agent learns to solve the Taxi-v3 environment from Gymnasium: drive to the passenger, pick them up, drive to the destination, and drop them off in as few steps as possible without making illegal moves.
Why it matters: Taxi-v3 is a compact, fully observable problem that still exercises the complete RL workflow: environment study, agent design, exploration vs exploitation, hyperparameter tuning, rigorous evaluation, and clear communication of results.
Deliverables: a reproducible Python package, a Jupyter notebook, a trained Q-table, evaluation results, learning-curve and policy visualizations, and a README plus short report.
Executor: a coding agent works through Phases 0 to 9 in Section 7 in order. Every task has an output and an acceptance check so the work can be verified without guessing.
2. Goals, Non-Goals and Success Metrics
Goals
• Implement Q-learning from scratch with NumPy (no RL library for the algorithm itself).
• Use an epsilon-greedy policy with a decay schedule.
• Tune learning rate (alpha), discount factor (gamma) and epsilon decay to speed up convergence, with evidence.
• Evaluate with average reward per episode and steps to completion, and plot the learning curve.
• Test the trained greedy policy on unseen episodes (reset seeds never used in training).
Non-goals
• Deep RL (DQN, PPO), function approximation, or multi-agent settings (stretch goals only).
• Custom environments, real maps, or GPS data.
• Production deployment.
Success metrics (targets, configurable)
Metric
Target
Greedy success rate on 1,000 unseen test episodes
100%
Average reward (greedy, unseen)
7.5 or higher (a perfect episode scores 21 minus steps)
Average steps to completion
14 or fewer
Illegal pickup/drop-off actions (greedy)
0
Convergence speed
Tuned config reaches the greedy-reward threshold in measurably fewer episodes than the baseline config
Reproducibility
Same seed gives an identical Q-table and metrics
Random-policy baseline
Reported and clearly beaten
3. Environment Specification (Taxi-v3)
Item
Value
Grid
5x5 with internal walls; four stops R, G, Y, B
States
500 discrete (25 taxi cells x 5 passenger locations x 4 destinations); about 400 are reachable
State encoding
((taxi_row x 5 + taxi_col) x 5 + passenger_location) x 4 + destination
Actions
0 south, 1 north, 2 east, 3 west, 4 pickup, 5 drop-off
Rewards
-1 per step; +20 for a successful drop-off; -10 for an illegal pickup or drop-off
Episode end
terminated on successful drop-off; truncated after 200 steps
API
env.reset(seed=...) returns (obs, info); env.step(a) returns (obs, reward, terminated, truncated, info)
The agent must verify these against the installed Gymnasium version in Phase 1 and record any difference.
4. Algorithm Specification
Q-learning update
Q[s, a] = Q[s, a] + alpha * (r + gamma * max_a' Q[s', a'] * (1 - terminated) - Q[s, a])
Rules
• Q is a NumPy array of shape (500, 6), initialised to zeros.
• Cut off bootstrapping with terminated, not truncated. A 200-step timeout is not a true terminal state, so the update should still bootstrap from s'.
• Epsilon-greedy: with probability epsilon pick a random action, otherwise the greedy action. Break ties between equal Q-values randomly; with an all-zero table a plain argmax always returns action 0 and biases early learning.
• Epsilon schedule: start at 1.0 and decay per episode to a floor. Support multiplicative (epsilon x decay) and linear schedules, selected by config.
• Use one seeded NumPy Generator for all exploration. Seed the environment on the first reset only; later resets are unseeded so the sequence stays reproducible but varied.
• Evaluation always uses the pure greedy policy (epsilon = 0) and never updates the Q-table.
Baseline configuration (config/default.yaml)
seed: 42
env: Taxi-v3
train:
  episodes: 10000
  alpha: 0.1
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.999      # multiplicative, per episode
  schedule: multiplicative  # or linear
  max_steps: 200
eval:
  every: 100                # greedy validation every N training episodes
  episodes: 50
  validation_seed_offset: 5000
  test_episodes: 1000
  test_seed_offset: 10000
  convergence_threshold: 7.0
  convergence_patience: 3   # consecutive evals above threshold
5. Tech Stack and Repository Structure
• Python 3.10+, gymnasium, numpy, pandas, matplotlib, pyyaml, pytest, jupyter. Optional: seaborn, imageio (GIFs), tqdm.
• Pin versions in requirements.txt.
taxi-rl/
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml
│   └── best.yaml            # written by the tuning phase
├── src/
│   ├── env_utils.py         # make_env, decode helpers, seeding
│   ├── agent.py             # QLearningAgent
│   ├── schedules.py         # epsilon schedules
│   ├── train.py             # training loop and logging
│   ├── evaluate.py          # greedy evaluation and baselines
│   ├── tune.py              # hyperparameter search
│   └── plots.py             # all figures
├── notebooks/taxi_q_learning.ipynb
├── tests/
├── results/                 # csv/json metrics, q_tables, figures, gifs
└── report/REPORT.md
6. Agent Operating Rules
1. Work in phase order. Do not start a phase until the previous phase's acceptance checks pass.
2. Keep logic in src/ modules. The notebook imports them and only orchestrates and displays results.
3. Seed every random source. Save the seed and config next to every result file.
4. Never hard-code results. Every number in the README or report must come from a saved results file.
5. Write artifacts to results/ with descriptive names, and never overwrite the final Q-table by accident.
6. After each phase, run the tests and append a short note (done, pending, assumptions) to CHANGELOG.md.
7. If the installed Gymnasium differs from Section 3, record the difference and adapt instead of silently assuming.
8. Do not change this spec. If something is ambiguous or blocked, state the assumption in the README and continue.
7. Section-wise Agent Tasks
Phase 0: Project setup
• T0.1 Create the repository structure from Section 5.
• T0.2 Write requirements.txt with pinned versions and a short setup guide in the README.
• T0.3 Create config/default.yaml and a loader that validates required keys and types.
• T0.4 Add a seeding helper that returns a seeded NumPy Generator.
Done when: a fresh virtual environment installs cleanly and Taxi-v3 can be created with gymnasium.make.
Phase 1: Environment exploration
• T1.1 Instantiate Taxi-v3 and print the observation space, action space, reward structure and step limit.
• T1.2 Write decode_state(s) returning (taxi_row, taxi_col, passenger_loc, destination) using env.unwrapped.decode, plus an encode round-trip check.
• T1.3 Render a few ansi frames in the notebook and explain the stops, walls and passenger-in-taxi state.
• T1.4 Run a seeded random-policy baseline for 1,000 episodes and record average reward, average steps, success rate, truncation rate and illegal-action count.
• T1.5 Verify the sanity invariant: for a successful episode with no illegal actions, total reward = 21 minus steps.
Done when: baseline metrics are saved to results/baseline_random.json and displayed in the notebook.
Phase 2: Agent implementation
• T2.1 Implement QLearningAgent in agent.py with: Q-table, select_action(state, epsilon, rng), greedy_action(state) with random tie-breaking, update(s, a, r, s_next, terminated), and save/load of the Q-table as .npy.
• T2.2 Implement multiplicative and linear epsilon schedules in schedules.py, each returning epsilon for a given episode index and respecting the floor.
• T2.3 Unit-test the update rule on a hand-computed transition, the terminal case (no bootstrap), tie-breaking, and schedule values.
Done when: all Phase 2 tests pass.
Phase 3: Training loop
• T3.1 Implement train(config) that runs N episodes and logs per episode: total reward, steps, epsilon, illegal-action count, success flag.
• T3.2 Every eval.every episodes, run a greedy validation over eval.episodes episodes on fixed validation seeds and log the average reward and steps.
• T3.3 Save the training log (CSV), the final Q-table and the exact config used.
• T3.4 Add a convergence detector: the first episode at which greedy validation reward stays at or above the threshold for eval.convergence_patience consecutive evaluations. Log it, or log that it never converged.
• T3.5 Print concise progress every 500 episodes and log wall-clock time per run.
Done when: a baseline run completes with logs saved, and the greedy validation reward visibly improves from strongly negative to near the target.
Phase 4: Hyperparameter tuning
• T4.1 Implement tune.py with a config-driven search (Section 8). Run configurations in parallel with multiprocessing.
• T4.2 Record per configuration, across seeds: mean and std of episodes-to-convergence, final greedy validation reward, area under the validation-reward curve, and the fraction of seeds that never converge.
• T4.3 Run one-factor-at-a-time sweeps for alpha, gamma and epsilon decay, and save a learning-curve plot per factor.
• T4.4 Run the focused grid, apply the selection rule in Section 8, and save the winner as config/best.yaml.
• T4.5 Write a short evidence-based paragraph per hyperparameter on how it affected convergence speed and stability.
Done when: results/tuning_results.csv and config/best.yaml exist, and the best configuration converges faster than the baseline configuration on average across seeds.
Phase 5: Final training
• T5.1 Retrain with config/best.yaml on 5 seeds, all distinct from validation and test seeds. Choose the primary Q-table by validation reward, never by test reward.
• T5.2 Save the primary Q-table to results/q_table_final.npy and all seed runs under results/final_runs/.
Done when: the files exist and rerunning the same seed reproduces an identical Q-table.
Phase 6: Evaluation on unseen episodes
• T6.1 Use test seeds disjoint from training and validation seeds (for example 10,000 to 10,999). Touch the test set only after the configuration is frozen.
• T6.2 Run the greedy policy for 1,000 test episodes and log per-episode reward, steps, success and illegal actions.
• T6.3 Compute mean, std and median reward; mean and median steps; success rate; truncation rate; illegal-action count.
• T6.4 Produce one comparison table: random policy, untrained all-zero Q-table, baseline-config Q-table, tuned Q-table.
• T6.5 Optimal-policy check (recommended): compute the optimal policy with value iteration over env.unwrapped.P, then report the share of states where the learned greedy action is an optimal action, and the gap between learned and optimal average steps.
• T6.6 Add a note to the report: tabular Q-learning visits essentially every state during training, so unseen here means unseen episodes (new reset seeds and start configurations), not unseen states.
Done when: results/test_metrics.json and results/comparison_table.csv exist.
Phase 7: Visualization
• T7.1 Implement every figure in Section 10 in plots.py as a function that takes saved results files and writes a PNG.
• T7.2 Save figures to results/figures/ at 150 dpi or higher and embed them in the notebook.
Done when: all figures V1 to V8 exist, and V9 (replay GIF) exists if imageio is installed.
Phase 8: Notebook, README and report
• T8.1 Build the notebook with sections mirroring the phases: introduction, environment, algorithm, training, tuning, evaluation, conclusions.
• T8.2 Write the README: problem, approach, install and run commands (train, tune, evaluate, plot), results summary table, key figures, repo structure, limitations.
• T8.3 Write report/REPORT.md (2 to 4 pages): problem, method, hyperparameter findings, results, discussion, limitations, future work.
• T8.4 Check that every number in the README and report matches a saved results file.
Done when: a reviewer can reproduce all results by following the README alone.
Phase 9: Testing and QA
• T9.1 Tests for: update rule, terminal vs truncated handling, schedules, same-seed determinism, evaluation not mutating the Q-table, state encode/decode round trip, the reward invariant (21 minus steps), and metric functions on synthetic data.
• T9.2 Run the complete pipeline from a clean clone (setup, train, tune with a reduced budget, evaluate, plot) and fix any failure.
• T9.3 Run a lint or format pass and remove dead code and unused files.
Done when: pytest passes and the end-to-end run finishes without manual intervention.
8. Hyperparameter Search Spec
Parameter
Values
alpha (learning rate)
0.05, 0.1, 0.3, 0.5, 0.7
gamma (discount factor)
0.8, 0.9, 0.95, 0.99
epsilon_decay (multiplicative, per episode)
0.99, 0.995, 0.999, 0.9995
epsilon_min
0.01, 0.05
episodes
5,000 during search, 10,000 for final training
Stage A (one factor at a time): vary one parameter around the baseline with the others fixed, 5 seeds each.
Stage B (focused grid): take the best 2 to 3 values per parameter from Stage A and run the full grid on them. Size the grid to the time budget in config (default 30 minutes), using the per-run wall-clock time logged in Phase 3 and parallel workers.
Selection rule: among configurations where at least 80% of seeds converge and the mean final greedy validation reward meets the target, choose the lowest mean episodes-to-convergence. Break ties by higher final reward, then lower std across seeds.
Hypotheses to test, not assume:
• A very small alpha converges slowly; a large alpha is usually safe here because Taxi-v3 transitions are deterministic.
• A low gamma shrinks the planning horizon, so the +20 drop-off reward is weakly felt from far-away start states.
• A fast epsilon decay can lock in a poor policy; a very slow decay wastes episodes.
9. Evaluation Protocol and Metrics
Metric
Definition
Average reward per episode
Mean total reward over episodes, greedy policy
Steps to completion
Mean and median steps per successful episode
Success rate
Share of episodes ending in terminated (successful drop-off)
Truncation rate
Share of episodes hitting the 200-step limit
Illegal actions
Count of steps with reward -10
Episodes to convergence
First episode meeting the convergence rule in T3.4
Reward std
Spread across episodes and across training seeds
Seed protocol: training seeds are independent of validation seeds (offset 5000) and test seeds (offset 10000). Report mean and std across the 5 final training seeds.
Training vs greedy curves: training reward is depressed by exploration, so always show the greedy validation curve beside it.
10. Visualization Spec
ID
Figure
V1
Learning curve: reward per episode (faint) with a 100-episode moving average; baseline vs tuned; mean with std band across seeds
V2
Steps per episode (moving average) over training, with the 200-step cap marked
V3
Epsilon schedule over episodes, overlaid on a secondary axis or as its own panel
V4
Greedy validation reward vs episode with the convergence episode marked
V5
Tuning plots: one-factor sweeps for alpha, gamma and epsilon decay; heatmaps of episodes-to-convergence (alpha x gamma, alpha x epsilon decay)
V6
Test results: histograms of reward and steps; bar chart comparing random, untrained, baseline and tuned
V7
Policy view: for a fixed passenger location and destination, best-action arrows and max-Q heatmap over the 5x5 grid, before pickup and after pickup
V8
Q-table heatmap (states x actions), optional
V9
GIF replay of one greedy test episode using rgb_array frames
Style: titled figures, labeled axes with units, legends, a colorblind-safe palette, and a consistent figure size.
11. Deliverables Checklist
Deliverable
Path
Phase
Source package
src/
2 to 7
Baseline and tuned configs
config/default.yaml, config/best.yaml
0, 4
Random baseline metrics
results/baseline_random.json
1
Training logs
results/ (CSV per run)
3, 5
Tuning results
results/tuning_results.csv
4
Final Q-table
results/q_table_final.npy
5
Test metrics and comparison
results/test_metrics.json, results/comparison_table.csv
6
Figures
results/figures/
7
Notebook
notebooks/taxi_q_learning.ipynb
8
README and report
README.md, report/REPORT.md
8
Tests
tests/
9
12. Risks and Mitigations
Risk
Mitigation
Treating timeout as terminal biases values
Bootstrap on truncation; zero the next value only on terminated
Argmax always picking action 0 on ties
Random tie-breaking in greedy_action
Learning curve looks worse than the policy really is
Show the greedy validation curve beside the training curve
High variance between seeds
5 seeds; report mean and std; judge configs by the fraction of seeds that converge
Overfitting hyperparameters to the test set
Select on validation seeds only; run the test set once after freezing
Gymnasium version differences
Pin versions and verify Section 3 in Phase 1
Claiming generalization to unseen states
Use the wording and note from T6.6
13. Stretch Goals (only after the Definition of Done)
• Compare against SARSA and Double Q-learning on the same seeds and metrics.
• Ablations: optimistic Q initialisation and alpha decay.
• Stochastic variant: a wrapper that makes moves slip with some probability, then re-tune and compare.
• Small Streamlit or Gradio demo where the user picks a start state and watches the agent solve it.
14. Definition of Done
• All Phase 0 to 9 acceptance checks pass.
• The tuned agent meets the success metrics in Section 2, or the report explains the gap with evidence.
• The learning curve, tuning plots, test comparison and policy view are in the notebook and README.
• Results are reproducible from a clean clone using only the README.
• No number in the README or report is missing a source file in results/.