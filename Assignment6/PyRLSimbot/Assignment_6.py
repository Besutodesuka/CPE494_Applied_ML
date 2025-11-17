#!/usr/bin/python3

from pathlib import Path
from collections import defaultdict
from typing import Tuple
import math
import random

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for window display
import matplotlib.pyplot as plt
from pysimbotlib.core import PySimbotApp, Robot
from pysimbotlib.core.config import SIMBOTMAP_SIZE
from kivy.config import Config

# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set('kivy', 'log_level', 'info')

# Tracking variables for eat and collision events
event_counts = {'eat': 0, 'collide': 0}
step_counter = 0
plot_history = []  # Initialize with starting point
plot_window = None  # Persistent plot window
plot_axes = None  # Persistent axes
# For diagnostics
cumulative_reward = 0.0
cumulative_rewards = []

# Learning hyper-parameters
PLOT_INTERVAL = 2000
# Q-learning parameters (reference values)
ALPHA_START = 0.5
ALPHA_MIN = 0.3
ALPHA_DECAY = 0.999
GAMMA = 0.9

EPSILON_START = 0.3
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999

TEMP_START = 1.2
TEMP_MIN = 0.2
TEMP_DECAY = 0.9995

FORWARD_STEP = 6
TURN_DEGREE = 18

# Distance buckets for discretising state space
CLOSE_DISTANCE = 15

MAP_PATH = Path(__file__).with_name("maps").joinpath("default_map.kv")
MAX_FOOD_DISTANCE = math.hypot(SIMBOTMAP_SIZE[0], SIMBOTMAP_SIZE[1])
POTENTIAL_WEIGHT = 3



ACTIONS = ("forward", "turn_left", "turn_right")
q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

prev_action = 0
def euclidian_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def plot_event_statistics(step, eat_count, collide_count):
    """Plot eat and collision counts as line graphs (stacked vertically)"""
    global plot_history, plot_window, plot_axes
    
    # Store data point
    plot_history.append({
        'step': step,
        'eat': eat_count,
        'collide': collide_count
    })
    
    # Create figure with two subplots stacked vertically if not exists
    if plot_window is None:
        plot_window = plt.figure(figsize=(10, 8))
        plot_window.suptitle('Robot Event Statistics (Real-time)', fontsize=14, weight='bold')
        plot_axes = [
            plot_window.add_subplot(2, 1, 1),
            plot_window.add_subplot(2, 1, 2)
        ]
        plt.subplots_adjust(hspace=0.3)
        plt.show(block=False)
    
    # Extract data for plotting
    steps = [p['step'] for p in plot_history]
    eats = [p['eat'] for p in plot_history]
    collides = [p['collide'] for p in plot_history]
    
    # Clear and plot eat events (line graph)
    plot_axes[0].clear()
    plot_axes[0].plot(steps, eats, marker='o', color='#2ecc71', linewidth=2, markersize=6, alpha=0.8)
    plot_axes[0].fill_between(steps, eats, alpha=0.3, color='#2ecc71')
    plot_axes[0].set_xlabel('Step', fontsize=11, weight='bold')
    plot_axes[0].set_ylabel('Eat Rate', fontsize=11, weight='bold')
    plot_axes[0].set_title('Robot Eating Events', fontsize=12, weight='bold')
    plot_axes[0].grid(True, alpha=0.3)
    
    # Clear and plot collision events (line graph)
    plot_axes[1].clear()
    plot_axes[1].plot(steps, collides, marker='s', color='#e74c3c', linewidth=2, markersize=6, alpha=0.8)
    plot_axes[1].fill_between(steps, collides, alpha=0.3, color='#e74c3c')
    plot_axes[1].set_xlabel('Step', fontsize=11, weight='bold')
    plot_axes[1].set_ylabel('Collision Rate', fontsize=11, weight='bold')
    plot_axes[1].set_title('Robot Collision Events', fontsize=12, weight='bold')
    plot_axes[1].grid(True, alpha=0.3)
    
    # Update window
    plot_window.canvas.draw()
    plot_window.canvas.flush_events()
    
    # Also save the plot
    if step == 100000:
        plot_window.savefig('realtime_events_100000.png', dpi=100, bbox_inches='tight')
    else:
        plot_window.savefig('realtime_events.png', dpi=100, bbox_inches='tight')
    print(f"Step {step}: Event plot updated (Eats: {eat_count}, Collisions: {collide_count})")

class RL_Robot(Robot):
    """Q-learning based robot with discretised state space and epsilon-greedy exploration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = ALPHA_START
        self.epsilon = EPSILON_START
        self.temperature = TEMP_START
        self.iteration = 0
        self.total_rotation = 0.0  # Track cumulative rotation for spin detection

    # -------------------------- Helpers for state/action --------------------------

    def _discretize_distance(self, distance: float, interval: int = 5) -> int:
        if distance < CLOSE_DISTANCE:
            return 0
        return 1

    def _discretize_smell(self, angle: float) -> int:
        if angle == -1:
            return 3
        if angle < -20:
            return 0
        if angle > 20:
            return 2
        return 1

    def _observe(self) -> Tuple[Tuple[int, ...], float]:
        ir = self.distance()
        target = self.smell()
        FIR = self._discretize_distance(ir[0])
        LIR = self._discretize_distance(ir[7])
        RIR = self._discretize_distance(ir[1])
        LLIR = self._discretize_distance(ir[6])
        RRIR = self._discretize_distance(ir[2])
        SM = self._discretize_smell(target)
        state = (FIR, LIR, RIR, LLIR, RRIR, SM)
        return state, target

    def _select_action(self, state: Tuple[int, ...]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        return self._max_q_action(state)

    def _softmax_action(self, state: Tuple[int, ...]) -> int:
        if random.random() < 0.95:
            q_values = q_table[state]
            # normalize q values to be probability weight for selecting action
            max_q = np.max(q_values)
            scaled = (q_values - max_q) / max(self.temperature, 1e-3)
            exp_values = np.exp(scaled)
            probabilities = exp_values / np.sum(exp_values)
            return int(np.random.choice(len(ACTIONS), p=probabilities))
    
    def _max_q_action(self, state: Tuple[int, ...]) -> int:
        q_values = q_table[state]
        return int(np.argmax(q_values))

    def _apply_action(self, action_idx: int) -> None:
        if action_idx == 0:
            self.move(FORWARD_STEP)
        elif action_idx == 1:
            self.turn(-TURN_DEGREE)
        else:
            self.turn(TURN_DEGREE)

    # -------------------------- Learning utilities --------------------------

    def _calculate_reward(
        self,
        prev_action,
        action_idx: int,
        prev_angle: float,
        next_angle: float,
        prev_distance: float,
        next_distance: float,
    ) -> float:
        reward = -0.05  # time penalty to encourage shorter solutions

        if action_idx == 0:
            if self.stuck:
                reward -= 3 # move forward to wall is no good
            elif abs(prev_angle) < 15:
                reward += 1 # move while facing toward food is good
            else:
                reward += 0.05 # small reward for able to move
        else: # base penalty for unnecessary  turn (will be zero sum by below term)
            reward -= 0.02
        
        # Track rotation for full-circle spin detection
        if action_idx == 1:  # Turn left
            self.total_rotation += TURN_DEGREE
        elif action_idx == 2:  # Turn right
            self.total_rotation -= TURN_DEGREE
        
        # Penalize full-circle rotations (360 degrees in either direction)
        if abs(self.total_rotation) >= 360:
            reward -= 5.0  # Heavy penalty for spinning in circles
            print(f"[SPIN] Step {step_counter}: Full circle detected! total_rotation={self.total_rotation:.1f}°")
            self.total_rotation = self.total_rotation % 360  # Reset to remaining angle
        
        if (prev_action == 1 and action_idx == 2) or (prev_action == 2 and action_idx == 1): # alternating spin penalty
            reward -= 3

        if prev_angle != -1 and next_angle != -1:
            # Reward being closer to the food (smaller absolute angle)
            reward += 0.02 * (abs(prev_angle) - abs(next_angle))

        if self.just_eat:
            # huge reward for eating
            reward += 8

        if prev_distance is not None and next_distance is not None:
            # reward for closer in distance
            phi_prev = self._potential(prev_distance)
            phi_next = self._potential(next_distance)
            reward += POTENTIAL_WEIGHT * (phi_next - phi_prev)
        
        return reward

    def _update_q_values(self, state: Tuple[int, ...], action_idx: int, reward: float, next_state: Tuple[int, ...]) -> None:
        current_q = q_table[state][action_idx]
        best_future = np.max(q_table[next_state])
        q_table[state][action_idx] = current_q + self.alpha * (reward + GAMMA * best_future - current_q)

    def _decay_hyperparameters(self) -> None:
        self.iteration += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)
        self.temperature = max(TEMP_MIN, self.temperature * TEMP_DECAY)

    def _food_distance(self) -> float:
        objectives = getattr(self._sm, "objectives", None)
        if not objectives:
            return None
        return min(
            math.hypot(self.center_x - obj.center_x, self.center_y - obj.center_y)
            for obj in objectives
        )

    def _potential(self, distance: float) -> float:
        normalized = 1.0 - min(distance / MAX_FOOD_DISTANCE, 1.0)
        return normalized

    # -------------------------- Main update loop --------------------------

    def update(self):
        global event_counts, step_counter, cumulative_reward, cumulative_rewards, prev_action
        
        self.just_eat = False
        current_state, prev_angle = self._observe()
        prev_distance = self._food_distance()

        action_idx = self._select_action(current_state)
        self._apply_action(action_idx)
        
        next_state, next_angle = self._observe()
        next_distance = self._food_distance()
        reward = self._calculate_reward(prev_action ,action_idx, prev_angle, next_angle, prev_distance, next_distance)
        prev_action = action_idx
        self._update_q_values(current_state, action_idx, reward, next_state)
        self._decay_hyperparameters()
        # accumulate reward for diagnostics
        cumulative_reward += reward
        # debug prints for signal events
        if self.just_eat:
            print(f"[DEBUG] Step {step_counter}: robot just ate; reward={reward:.2f}")
        if getattr(self, 'just_hit', False):
            print(f"[DEBUG] Step {step_counter}: robot just hit; reward={reward:.2f}")
        
        # # Track events
        step_counter += 1
        # Plot and reset every PLOT_INTERVAL steps
        if step_counter % PLOT_INTERVAL == 0:
            event_counts['eat'] = self.eat_count
            event_counts['collide'] = self.collision_count # update temp
            avg_reward = cumulative_reward / PLOT_INTERVAL if cumulative_reward != 0 else 0.0
            cumulative_rewards.append(avg_reward)
            print(f"[DIAG] Steps {step_counter-999}-{step_counter}: avg reward={avg_reward:.4f}")
            cumulative_reward = 0.0
            plot_event_statistics(step_counter, self.eat_count / step_counter, self.collision_count / step_counter)


def randomize_objectives(simbot):
    """Randomise all objective positions at simulation start."""
    for obj in simbot.objectives:
        simbot.change_objective_pos(obj)


if __name__ == '__main__':
    app = PySimbotApp(
        robot_cls=RL_Robot,
        num_robots=1,
        max_tick=110000,
        simulation_forever=True,
        map_path=str(MAP_PATH),
        customfn_before_simulation=randomize_objectives,
        food_move_after_eat=True,
    )
    app.run()
