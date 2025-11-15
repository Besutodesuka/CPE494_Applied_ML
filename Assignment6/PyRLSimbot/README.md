# Assignment 6 – PyRLSimbot

This project contains the solution for AML Assignment 06 (Reinforcement Learning). A PySimbot robot learns to navigate a 2D map, avoid obstacles and reach food using a Q-learning policy that is trained online while the simulation is running.

## Getting Started

1. Install Python 3.9+ along with `pip`.
2. Install dependencies for your platform:
   - Windows: `pip install -r requirements_windows.txt`
   - macOS: `pip3 install -r requirements_macos.txt`
   - Linux / others: `pip install -r requirements_basic.txt`
3. Launch the simulator: `python Assignment_6.py`
   - The Kivy window opens and the robot immediately starts exploring.
   - The simulation runs indefinitely (`simulation_forever=True`). Stop it with `Ctrl+C` or by closing the window.

## Learning Algorithm Overview

The implementation in `Assignment_6.py` uses tabular Q-learning with the following design decisions:

- **State representation** – We discretise five distance sensors (front, front-right, front-left, far-left, far-right) into three buckets: *close* (<25), *near* (25–60) and *far* (>60). A sixth feature is the smell angle, bucketed into *left*, *center*, *right*, or *not-detected*. The resulting tuple forms the key for our Q-table.
- **Actions** – Three actions are available: `forward` (move 6 px), `turn_left` (−18°) and `turn_right` (+18°).
- **Exploration strategy** – Starts with ε-greedy (ε decays 0.45 → 0.05) but blends in a Boltzmann softmax controlled by a temperature parameter that decays 1.2 → 0.2. This keeps exploration stochastic while gradually emphasising the best Q-values.
- **Learning rate** – The Q-table is backed by `defaultdict(lambda: np.zeros(3))`. The learning rate `alpha` decays from 0.35 to 0.05. Discount factor γ is 0.95.
- **Potential-based reward shaping** – Besides the handcrafted terms below, we add a potential function based on the Euclidean distance to the closest food. The shaped reward includes `W_p * (γΦ(s') − Φ(s))`, which encourages progress while keeping the optimal policy invariant.
- **Reward function** – Rewards encourage useful behaviours:
  - Time penalty (−0.05) to motivate faster food acquisition.
  - Moving forward yields +0.3 unless the robot is stuck (−2.5).
  - Turning costs −0.02.
  - Aligning with the food source adds `0.02 * (|prev_angle| - |next_angle|)`; losing the scent gives −0.5.
  - Eating food grants a large bonus (+8).
  - Potential term rewards moving closer to food based on map-wide Euclidean distance.
- **Map control** – `Assignment_6.py` explicitly loads `maps/default_map.kv` and randomises objective positions at the start via a `customfn_before_simulation` hook. After each eat, PySimbot automatically respawns food in a new random valid location.
- **Update loop** – Each `update()` call observes the state, selects an action using ε/softmax hybrid exploration, executes it, recomputes the next state and smell angle, computes the shaped reward (including potentials), performs the TD update `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`, and decays α/ε/temperature.

These additions give the robot a mathematically richer signal: it balances stochastic exploration with softmax sampling and receives potential-based shaping rewards tied to the precise map geometry, leading to faster convergence toward reliable food-seeking behaviours.
