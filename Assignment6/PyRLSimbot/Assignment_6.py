#!/usr/bin/python3

from pathlib import Path
from collections import defaultdict
from typing import Tuple
import math
import random

import numpy as np
from pysimbotlib.core import PySimbotApp, Robot
from pysimbotlib.core.config import SIMBOTMAP_SIZE
from kivy.config import Config

# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set('kivy', 'log_level', 'info')

# Learning hyper-parameters
ALPHA_START = 0.35
ALPHA_MIN = 0.05
ALPHA_DECAY = 0.999
GAMMA = 0.95

EPSILON_START = 0.45
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

TEMP_START = 1.2
TEMP_MIN = 0.2
TEMP_DECAY = 0.9995

FORWARD_STEP = 6
TURN_DEGREE = 18

# Distance buckets for discretising state space
CLOSE_DISTANCE = 25
NEAR_DISTANCE = 60

MAP_PATH = Path(__file__).with_name("maps").joinpath("default_map.kv")
MAX_FOOD_DISTANCE = math.hypot(SIMBOTMAP_SIZE[0], SIMBOTMAP_SIZE[1])
POTENTIAL_WEIGHT = 1.8

ACTIONS = ("forward", "turn_left", "turn_right")
q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))


class RL_Robot(Robot):
    """Q-learning based robot with discretised state space and epsilon-greedy exploration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = ALPHA_START
        self.epsilon = EPSILON_START
        self.temperature = TEMP_START
        self.iteration = 0

    # -------------------------- Helpers for state/action --------------------------

    def _discretize_distance(self, distance: float) -> int:
        if distance < CLOSE_DISTANCE:
            return 0
        if distance < NEAR_DISTANCE:
            return 1
        return 2

    def _discretize_smell(self, angle: float) -> int:
        if angle == -1:
            return 3
        if angle < -20:
            return 0
        if angle > 20:
            return 2
        return 1

    def _observe(self) -> Tuple[Tuple[int, ...], float]:
        smell_angle = self.smell()
        distances = self.distance()
        state = (
            self._discretize_distance(distances[0]),  # front
            self._discretize_distance(distances[1]),  # front-right
            self._discretize_distance(distances[7]),  # front-left
            self._discretize_distance(distances[6]),  # far-left
            self._discretize_distance(distances[2]),  # far-right
            self._discretize_smell(smell_angle),
        )
        return state, smell_angle

    def _select_action(self, state: Tuple[int, ...]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        return self._softmax_action(state)

    def _softmax_action(self, state: Tuple[int, ...]) -> int:
        q_values = q_table[state]
        max_q = np.max(q_values)
        scaled = (q_values - max_q) / max(self.temperature, 1e-3)
        exp_values = np.exp(scaled)
        probabilities = exp_values / np.sum(exp_values)
        return int(np.random.choice(len(ACTIONS), p=probabilities))
        q_values = q_table[state]
        max_q = np.max(q_values)
        best_actions = [i for i, value in enumerate(q_values) if value == max_q]
        return random.choice(best_actions)

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
        action_idx: int,
        prev_angle: float,
        next_angle: float,
        prev_distance: float,
        next_distance: float,
    ) -> float:
        reward = -0.05  # time penalty to encourage shorter solutions

        if action_idx == 0:
            if self.stuck:
                reward -= 2.5
            else:
                reward += 0.3
        else:
            reward -= 0.02

        if prev_angle != -1 and next_angle != -1:
            # Reward being closer to the food (smaller absolute angle)
            reward += 0.02 * (abs(prev_angle) - abs(next_angle))
        elif next_angle == -1:
            reward -= 0.5

        if self.just_eat:
            reward += 8

        if prev_distance is not None and next_distance is not None:
            phi_prev = self._potential(prev_distance)
            phi_next = self._potential(next_distance)
            reward += POTENTIAL_WEIGHT * (GAMMA * phi_next - phi_prev)

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
        self.just_eat = False
        current_state, prev_angle = self._observe()
        prev_distance = self._food_distance()

        action_idx = self._select_action(current_state)
        self._apply_action(action_idx)

        next_state, next_angle = self._observe()
        next_distance = self._food_distance()
        reward = self._calculate_reward(action_idx, prev_angle, next_angle, prev_distance, next_distance)
        self._update_q_values(current_state, action_idx, reward, next_state)
        self._decay_hyperparameters()


def randomize_objectives(simbot):
    """Randomise all objective positions at simulation start."""
    for obj in simbot.objectives:
        simbot.change_objective_pos(obj)


if __name__ == '__main__':
    app = PySimbotApp(
        robot_cls=RL_Robot,
        num_robots=1,
        max_tick=5000,
        simulation_forever=True,
        map_path=str(MAP_PATH),
        customfn_before_simulation=randomize_objectives,
        food_move_after_eat=True,
    )
    app.run()
