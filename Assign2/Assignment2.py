#!/usr/bin/python3

import os, sys
import random

from pysimbotlib.Window import PySimbotApp
from pysimbotlib.Robot import Robot
from kivy.core.window import Window
from kivy.logger import Logger

# Number of robot that will be run
ROBOT_NUM = 1

# Delay between update (default: 1/60 (or 60 frame per sec))
TIME_INTERVAL = 1.0/60 #10frame per second 

# Max tick
MAX_TICK = 5000

# START POINT
START_POINT = (20, 560)

# Map file
MAP_FILE = 'maps/default_map.kv'

class FuzzyRobot(Robot):

    def __init__(self):
        super(FuzzyRobot, self).__init__()
        self.pos = START_POINT

    def update(self):
        ''' Update method which will be called each frame
        '''        
        self.ir_values = self.distance()
        self.target = self.smell()

        # Add oscillation detection
        if not hasattr(self, 'prev_positions'):
            self.prev_positions = []
            self.stuck_counter = 0
            
        current_pos = (round(self.pos[0]/10), round(self.pos[1]/10))  # Discretize position
        self.prev_positions.append(current_pos)
        if len(self.prev_positions) > 10:
            self.prev_positions.pop(0)
            
        # Check if stuck (same position for multiple frames)
        if len(self.prev_positions) >= 5 and len(set(self.prev_positions[-5:])) <= 2:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Priority-based rule system to prevent conflicts
        rules = []
        turns = []
        moves = []

        # PRIORITY 1: Escape mechanism when stuck
        if self.stuck_counter > 5:
            rules.append(1.0)  # Maximum priority
            # Choose escape direction based on clearest path
            escape_directions = [self.ir_values[i] for i in [1, 2, 3, 4, 5, 6, 7]]
            max_distance_idx = escape_directions.index(max(escape_directions))
            escape_angles = [45, 90, 135, 180, -135, -90, -45]
            turns.append(escape_angles[max_distance_idx])
            moves.append(3)
            self.stuck_counter = 0  # Reset after escape attempt
        
        # PRIORITY 2: Completely trapped - aggressive escape
        elif self.front_near() * self.left_near() * self.right_near() > 0.7:
            rules.append(self.front_near() * self.left_near() * self.right_near())
            # Find the clearest direction among rear sensors
            rear_distances = [self.ir_values[i] for i in [4, 5, 6, 7]]
            best_rear_idx = rear_distances.index(max(rear_distances))
            rear_angles = [180, -135, -90, -45]
            turns.append(rear_angles[best_rear_idx])
            moves.append(2)
        
        # PRIORITY 3: Front blocked - avoid collision
        elif self.front_near() > 0.5:
            rules.append(self.front_near())
            # Choose turn direction based on clearer side
            left_space = (self.ir_values[6] + self.ir_values[7]) / 2
            right_space = (self.ir_values[1] + self.ir_values[2]) / 2
            if left_space > right_space:
                turns.append(-45)
            else:
                turns.append(45)
            moves.append(1)
        
        # PRIORITY 4: Side obstacles - navigate around
        elif self.left_near() > 0.6:
            rules.append(self.left_near())
            turns.append(30)
            moves.append(4)
        elif self.right_near() > 0.6:
            rules.append(self.right_near())
            turns.append(-30)
            moves.append(4)
        
        # PRIORITY 5: Clear path - move toward food
        elif self.front_far() > 0.7:
            rules.append(self.front_far())
            # Smooth turning toward food
            target_angle = self.target
            if abs(target_angle) < 10:  # Food nearly straight ahead
                turns.append(target_angle * 0.2)
                moves.append(8)
            elif target_angle < -15:  # Food is left
                turns.append(-20)
                moves.append(6)
            elif target_angle > 15:  # Food is right
                turns.append(20)
                moves.append(6)
            else:
                turns.append(target_angle * 0.5)
                moves.append(7)

        # Default behavior if no rules fire strongly
        if not rules or max(rules) < 0.1:
            rules.append(0.1)
            # Random exploration to break deadlock
            import random
            turns.append(random.choice([-30, -15, 15, 30]))
            moves.append(2)

        # Simple defuzzification - take the highest priority rule
        if rules:
            max_idx = rules.index(max(rules))
            ans_turn = turns[max_idx]
            ans_move = moves[max_idx]
        else:
            ans_turn = 0.0
            ans_move = 1.0

        # Apply movement limits
        ans_turn = max(-50, min(50, ans_turn))
        ans_move = max(1, min(8, ans_move))

        self.turn(ans_turn)
        self.move(ans_move)
        
    def front_far(self):
        irfront = self.ir_values[0]
        if irfront <= 10:
            return 0.0
        elif irfront >= 40:
            return 1.0
        else:
            return (irfront-10.0) / 30.0

    def front_near(self):
        return 1 - self.front_far()

    def left_far(self):
        irleft = self.ir_values[6]
        if irleft <= 10:
            return 0.0
        elif irleft >= 30: 
            return 1.0
        else:
            return (irleft-10.0) / 20.0

    def left_near(self):
        return 1 - self.left_far()

    def right_far(self):
        irright = self.ir_values[2]
        if irright <= 10:
            return 0.0
        elif irright >= 30:
            return 1.0
        else:
            return (irright-10.0) / 20.0

    def right_near(self):
        return 1 - self.right_far()

    def smell_right(self):
        target = self.smell()
        if target >= 90:
            return 1.0
        elif target <= 0:
            return 0.0
        else:
            return target / 90.0

    def smell_center(self):
        target = abs(self.smell())
        if target >= 45:
            return 1.0
        elif target <= 0:
            return 0.0
        else:
            return target / 45.0

    def smell_left(self):
        target = self.smell()
        if target <= -90:
            return 1.0
        elif target >= 0:
            return 0.0
        else:
            return -target / 90.0
    
    # Additional membership functions for improved obstacle avoidance
    def rear_left_far(self):
        ir_rear_left = self.ir_values[7]  # IR-7 is rear-left
        if ir_rear_left <= 15:
            return 0.0
        elif ir_rear_left >= 35:
            return 1.0
        else:
            return (ir_rear_left - 15.0) / 20.0
    
    def rear_right_far(self):
        ir_rear_right = self.ir_values[1]  # IR-1 is rear-right  
        if ir_rear_right <= 15:
            return 0.0
        elif ir_rear_right >= 35:
            return 1.0
        else:
            return (ir_rear_right - 15.0) / 20.0
    
    def diagonal_left_near(self):
        ir_diag_left = self.ir_values[5]  # IR-5 is diagonal left-rear
        if ir_diag_left >= 25:
            return 0.0
        elif ir_diag_left <= 10:
            return 1.0
        else:
            return (25.0 - ir_diag_left) / 15.0
    
    def diagonal_right_near(self):
        ir_diag_right = self.ir_values[3]  # IR-3 is diagonal right-rear
        if ir_diag_right >= 25:
            return 0.0
        elif ir_diag_right <= 10:
            return 1.0
        else:
            return (25.0 - ir_diag_right) / 15.0

if __name__ == '__main__':
    app = PySimbotApp(FuzzyRobot, ROBOT_NUM, mapPath=MAP_FILE, interval=TIME_INTERVAL, maxtick=MAX_TICK)
    app.run()
