#!/usr/bin/python3

import os, sys, platform
import random
import math

# Force video settings for macOS display
if platform.system() == "Darwin":  # macOS
    os.environ["KIVY_GL_BACKEND"] = "gl"
    os.environ["SDL_VIDEODRIVER"] = "cocoa"
    os.environ["KIVY_WINDOW"] = "sdl2"
    print("🖥️  macOS Display Settings Applied")

from pysimbotlib.Window import PySimbotApp
from pysimbotlib.Robot import Robot
from kivy.core.window import Window
from kivy.logger import Logger

# Number of robot that will be run
ROBOT_NUM = 1

# Delay between update (default: 1/60 (or 60 frame per sec))
TIME_INTERVAL = 1.0/60

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
        
        # ===== INTELLIGENT ANTI-OSCILLATION SYSTEM =====
        self.position_history = []  # Track recent positions
        self.turn_history = []      # Track recent turns  
        self.stuck_counter = 0      # Count stuck iterations
        self.last_food_angle = 180  # Previous food angle
        self.oscillation_penalty = 0.0  # Penalty for oscillating behavior
        
        # ===== ADAPTIVE LEARNING PARAMETERS =====
        self.performance_score = 0   # Overall performance tracking
        self.food_progress_score = 0 # Progress towards food
        self.exploration_bias = 0.1  # Encourages exploration vs exploitation
        self.momentum_factor = 0.0   # Maintains consistent direction
        
        # ===== DYNAMIC FUZZY PARAMETERS =====
        self.sensitivity_multiplier = 1.0  # Adaptive sensitivity
        self.aggression_level = 1.0        # How aggressive in food seeking
        self.precision_threshold = 15.0     # Angle threshold for precision mode
        
        print("🧠 Intelligent Anti-Oscillation Fuzzy Robot initialized!")

    def update(self):
        ''' Update method - INTELLIGENT FUZZY LOGIC SYSTEM with ANTI-OSCILLATION '''
        self.ir_values = self.distance()
        self.target = self.smell()
        
        # Enhanced fuzzy rule system - comprehensive behavior
        rules = list()
        turns = list()
        moves = list()
        
        # ===== INTELLIGENT FUZZY RULES WITH ANTI-OSCILLATION =====
        
        # Rule 0: Strong food signal + clear path - aggressive pursuit
        rule0_strength = self.path_clear() * self.smell_strong()
        if rule0_strength > 0.3:
            rules.append(rule0_strength * 1.8)  # High priority
            # Apply momentum to reduce oscillation
            base_turn = self.target * 0.6 * self.sensitivity_multiplier
            if len(self.turn_history) > 0 and self.oscillation_penalty > 0.2:
                momentum = self.turn_history[-1] * self.momentum_factor * 0.3
                momentum_adjusted_turn = base_turn * (1 - self.momentum_factor * 0.3) + momentum
            else:
                momentum_adjusted_turn = base_turn
            turns.append(max(-50, min(50, momentum_adjusted_turn)))
            moves.append(int(7 + 2 * self.aggression_level))
        
        # Rule 1: Food left + path clear - intelligent left turn
        rule1_strength = self.smell_left() * self.path_clear() * (1 - self.oscillation_penalty)
        if rule1_strength > 0.2:
            rules.append(rule1_strength)
            # Adaptive turn rate based on distance and performance
            turn_intensity = -30 * self.smell_left() * self.sensitivity_multiplier
            if abs(self.target) < self.precision_threshold:  # Precision mode
                turn_intensity *= 0.7  # Gentler turns when close
            turns.append(max(-55, turn_intensity))
            moves.append(int(5 + self.aggression_level))
        
        # Rule 2: Food right + path clear - intelligent right turn
        rule2_strength = self.smell_right() * self.path_clear() * (1 - self.oscillation_penalty)
        if rule2_strength > 0.2:
            rules.append(rule2_strength)
            # Adaptive turn rate based on distance and performance
            turn_intensity = 30 * self.smell_right() * self.sensitivity_multiplier
            if abs(self.target) < self.precision_threshold:  # Precision mode
                turn_intensity *= 0.7  # Gentler turns when close
            turns.append(min(55, turn_intensity))
            moves.append(int(5 + self.aggression_level))
        
        # Rule 3: Food centered + clear path - precision targeting
        rule3_strength = self.smell_center() * self.path_clear() * self.aggression_level
        if rule3_strength > 0.3:
            rules.append(rule3_strength * 2.2)  # Highest priority for centered food
            # Micro-adjustments for precision
            precision_turn = self.target * 0.4 if abs(self.target) < self.precision_threshold else 0
            turns.append(precision_turn)
            # Speed boost when aligned with food
            moves.append(int(8 + 2 * self.aggression_level))
        
        # Rule 4: Front obstacle - intelligent avoidance
        rule4_strength = self.front_near() * (1 + self.oscillation_penalty * 0.5)
        if rule4_strength > 0.3:
            rules.append(rule4_strength)
            # Smart obstacle avoidance considering food direction
            if self.left_far() > self.right_far() + 0.1:
                base_turn = -45
            elif self.right_far() > self.left_far() + 0.1:
                base_turn = 45
            else:
                # Choose direction closer to food
                base_turn = 35 if self.target > 0 else -35
            
            turns.append(base_turn * self.sensitivity_multiplier)
            moves.append(max(1, int(3 - self.oscillation_penalty * 2)))
        
        # Rule 5: Ultra-precision mode - very close to food
        food_closeness = self.smell_strong()
        ultra_close = 1.0 if abs(self.target) < 10 else 0.0
        if food_closeness > 0.6 or ultra_close > 0:
            rule5_strength = (food_closeness + ultra_close) * self.path_clear()
            if rule5_strength > 0:
                rules.append(rule5_strength * 2.8)  # Maximum priority
                # Ultra-precise steering
                micro_turn = self.target * 0.3 * self.sensitivity_multiplier
                turns.append(max(-25, min(25, micro_turn)))
                moves.append(int(6 + self.aggression_level))
        
        # Rule 6: Intelligent escape from complex obstacles
        obstacle_count = self.front_near() + self.left_near() + self.right_near() + self.back_near()
        if obstacle_count > 1.5 or self.stuck_counter > 20:
            rule6_strength = min(1.0, (obstacle_count / 4.0) + (self.stuck_counter / 40.0))
            rules.append(rule6_strength * 1.1)
            
            # Advanced escape strategy considering food direction
            space_values = [self.front_far(), self.right_far(), self.back_far(), self.left_far()]
            space_angles = [0, 90, 180, -90]
            
            # Weight escape directions by food alignment
            weighted_scores = []
            for space, angle in zip(space_values, space_angles):
                food_alignment = 1.0 - abs(angle - self.target) / 180.0
                weighted_score = space * 0.7 + food_alignment * 0.3
                weighted_scores.append(weighted_score)
            
            best_idx = weighted_scores.index(max(weighted_scores))
            escape_turn = space_angles[best_idx] * 0.6
            
            # Add randomness to break deadlock
            if self.stuck_counter > 30:
                escape_turn += random.uniform(-25, 25)
            
            turns.append(max(-75, min(75, escape_turn)))
            moves.append(int(3 + self.exploration_bias * 4))
        
        # Rule 7: Advanced obstacle navigation with food bias
        moderate_food = (self.smell_center() * 0.4 + self.smell_left() * 0.3 + 
                        self.smell_right() * 0.3) * self.aggression_level
        obstacle_interference = self.path_blocked()
        
        if moderate_food > 0.25 and obstacle_interference > 0.3:
            rule7_strength = moderate_food * (1 - obstacle_interference * 0.6)
            if rule7_strength > 0:
                rules.append(rule7_strength)
                
                # Intelligent pathfinding around obstacles
                food_bias_left = self.smell_left() * self.left_far()
                food_bias_right = self.smell_right() * self.right_far()
                
                if food_bias_left > food_bias_right + 0.1:  # Prefer left
                    if self.left_far() > 0.4:
                        turns.append(-20 * self.sensitivity_multiplier)
                    else:  # Wall hug right when can't go left
                        turns.append(35 * self.sensitivity_multiplier)
                elif food_bias_right > food_bias_left + 0.1:  # Prefer right
                    if self.right_far() > 0.4:
                        turns.append(20 * self.sensitivity_multiplier)
                    else:  # Wall hug left when can't go right
                        turns.append(-35 * self.sensitivity_multiplier)
                else:  # Equal preference - choose based on space
                    turns.append((self.right_far() - self.left_far()) * 30)
                
                moves.append(int(4 + self.exploration_bias * 3))
        
        # Rule 8: Anti-stagnation - force exploration when stuck
        if self.stuck_counter > 15:
            exploration_strength = min(1.0, self.stuck_counter / 30.0)
            rules.append(exploration_strength * 0.8)
            
            # Biased random exploration towards food
            if abs(self.target) > 90:  # Food is behind
                exploration_turn = random.uniform(-50, 50)
            else:  # Food is ahead, explore perpendicular
                exploration_turn = random.choice([-40, 40]) + random.uniform(-10, 10)
            
            turns.append(exploration_turn)
            moves.append(int(3 + self.exploration_bias * 4))
        
        # ===== ENHANCED FUZZY DEFUZZIFICATION =====
        ans_turn = 0.0
        ans_move = 0.0
        total_strength = sum(rules)
        
        if total_strength > 0:
            # Advanced weighted defuzzification with momentum
            for r, t, m in zip(rules, turns, moves):
                ans_turn += t * r
                ans_move += m * r
            
            # Normalize by total rule strength
            ans_turn = ans_turn / total_strength
            ans_move = ans_move / total_strength
            
            # Apply anti-oscillation momentum
            if len(self.turn_history) > 0 and self.oscillation_penalty > 0.2:
                momentum = self.turn_history[-1] * self.momentum_factor * 0.25
                ans_turn = ans_turn * (1 - self.momentum_factor * 0.25) + momentum
            
            # Dynamic range adjustment based on performance
            if self.performance_score > 10:  # High performance - more aggressive
                ans_turn *= 1.15
                ans_move *= 1.2
            elif self.oscillation_penalty > 0.5:  # Oscillating - more conservative
                ans_turn *= 0.75
                ans_move *= 0.85
                
        else:
            # Intelligent fallback - biased exploration
            food_direction = 1 if self.target > 0 else -1
            ans_turn = random.uniform(-15, 15) + food_direction * 12
            ans_move = int(3 + self.exploration_bias * 2)
        
        # Enhanced constraints with adaptive limits
        max_turn = 70 if self.performance_score > 8 else 55
        min_move = 2 if abs(self.target) < self.precision_threshold else 1
        max_move = int(9 + self.aggression_level) if self.path_clear() > 0.7 else 7
        
        final_turn = max(-max_turn, min(max_turn, ans_turn))
        final_move = max(min_move, min(max_move, int(ans_move)))
        
        self.turn(final_turn)
        self.move(final_move)
        
        # ===== INTELLIGENT BEHAVIOR ANALYSIS & ANTI-OSCILLATION =====
        self._update_position_history()
        self._detect_and_prevent_oscillation(final_turn, final_move)
        self._update_performance_metrics()
        self._adapt_parameters()
        
        # Enhanced debug output with intelligence metrics
        oscillation_status = "🔄" if self.oscillation_penalty > 0.3 else "✅"
        performance_level = "🏆" if self.performance_score > 10 else "📈" if self.performance_score > 5 else "🎯"
        
        print(f"{performance_level} Rules: {len(rules)}, Turn: {final_turn:.1f}°, Move: {final_move}, Food: {self.target:.1f}°, Score: {self.performance_score:.1f} {oscillation_status}")
        
    # ===== INTELLIGENT BEHAVIOR MANAGEMENT =====
    
    def _update_position_history(self):
        """Track position history for oscillation detection"""
        self.position_history.append(self.pos)
        if len(self.position_history) > 10:  # Keep last 10 positions
            self.position_history.pop(0)
    
    def _detect_and_prevent_oscillation(self, turn_amount, move_amount):
        """Detect oscillating behavior and apply penalties"""
        self.turn_history.append(turn_amount)
        if len(self.turn_history) > 8:  # Keep last 8 turns
            self.turn_history.pop(0)
        
        # Check for oscillation patterns
        if len(self.turn_history) >= 6:
            recent_turns = self.turn_history[-6:]
            
            # Pattern 1: Alternating left-right turns
            alternating = sum(1 for i in range(1, len(recent_turns)) 
                             if recent_turns[i] * recent_turns[i-1] < 0) >= 4
            
            # Pattern 2: Stuck in same area
            if len(self.position_history) >= 5:
                recent_positions = self.position_history[-5:]
                max_distance = max(abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1]) 
                                 for pos in recent_positions)
                stuck_in_area = max_distance < 60  # Less than 60 pixels movement
            else:
                stuck_in_area = False
            
            # Apply oscillation penalty
            if alternating or stuck_in_area:
                self.oscillation_penalty = min(1.0, self.oscillation_penalty + 0.08)
                self.stuck_counter += 1
            else:
                self.oscillation_penalty = max(0.0, self.oscillation_penalty - 0.04)
                self.stuck_counter = max(0, self.stuck_counter - 1)
    
    def _update_performance_metrics(self):
        """Track and update performance metrics"""
        current_food_angle = abs(self.target)
        
        # Reward getting closer to food
        if current_food_angle < abs(self.last_food_angle) - 3:
            self.performance_score += 1.5
            self.food_progress_score += 1.0
        elif current_food_angle > abs(self.last_food_angle) + 8:
            self.performance_score = max(0, self.performance_score - 0.3)
            self.food_progress_score = max(0, self.food_progress_score - 0.3)
        
        # Penalty for oscillation
        if self.oscillation_penalty > 0.4:
            self.performance_score = max(0, self.performance_score - 0.8)
        
        # Update momentum based on food progress
        if current_food_angle < 25:  # Close to food
            self.momentum_factor = min(0.7, self.momentum_factor + 0.08)
        else:
            self.momentum_factor = max(0.0, self.momentum_factor - 0.03)
        
        self.last_food_angle = self.target
    
    def _adapt_parameters(self):
        """Dynamically adapt fuzzy parameters based on performance"""
        # Increase sensitivity when performing well
        if self.performance_score > 8:
            self.sensitivity_multiplier = min(1.4, self.sensitivity_multiplier + 0.015)
            self.aggression_level = min(1.6, self.aggression_level + 0.04)
        elif self.performance_score < 3:
            self.sensitivity_multiplier = max(0.6, self.sensitivity_multiplier - 0.008)
        
        # Adjust precision based on distance to food
        current_food_distance = abs(self.target)
        if current_food_distance < 15:
            self.precision_threshold = 6.0   # High precision when close
        elif current_food_distance < 35:
            self.precision_threshold = 10.0  # Medium precision
        else:
            self.precision_threshold = 18.0  # Lower precision when far
        
        # Reduce exploration when close to food
        if current_food_distance < 25:
            self.exploration_bias = max(0.03, self.exploration_bias - 0.008)
        else:
            self.exploration_bias = min(0.15, self.exploration_bias + 0.003)
    
    # ===== ENHANCED FUZZY MEMBERSHIP FUNCTIONS =====
    
    def front_far(self):
        """Enhanced front sensor fuzzy membership"""
        irfront = self.ir_values[0]
        threshold = 18 + self.sensitivity_multiplier * 8
        if irfront <= threshold * 0.7:
            return 0.0
        elif irfront >= threshold * 2.2:
            return 1.0
        else:
            return (irfront - threshold * 0.7) / (threshold * 1.5)
    
    def front_near(self):
        return 1.0 - self.front_far()
    
    def left_far(self):
        """Enhanced left sensor fuzzy membership"""
        irleft = self.ir_values[6]
        threshold = 16 + self.sensitivity_multiplier * 6
        if irleft <= threshold * 0.8:
            return 0.0
        elif irleft >= threshold * 2:
            return 1.0
        else:
            return (irleft - threshold * 0.8) / (threshold * 1.2)
    
    def left_near(self):
        return 1.0 - self.left_far()
    
    def right_far(self):
        """Enhanced right sensor fuzzy membership"""
        irright = self.ir_values[2]
        threshold = 16 + self.sensitivity_multiplier * 6
        if irright <= threshold * 0.8:
            return 0.0
        elif irright >= threshold * 2:
            return 1.0
        else:
            return (irright - threshold * 0.8) / (threshold * 1.2)
    
    def right_near(self):
        return 1.0 - self.right_far()
    
    def back_far(self):
        """Enhanced back sensor fuzzy membership"""
        irback = self.ir_values[4]
        if irback <= 10:
            return 0.0
        elif irback >= 30:
            return 1.0
        else:
            return (irback - 10.0) / 20.0
    
    def back_near(self):
        return 1.0 - self.back_far()
    
    def smell_right(self):
        """Enhanced right food detection"""
        target = self.target
        if target <= 0:
            return 0.0
        elif target >= 90:
            return 1.0 * self.aggression_level
        else:
            return (target / 90.0) * self.aggression_level
    
    def smell_left(self):
        """Enhanced left food detection"""
        target = self.target
        if target >= 0:
            return 0.0
        elif target <= -90:
            return 1.0 * self.aggression_level
        else:
            return (-target / 90.0) * self.aggression_level
    
    def smell_center(self):
        """Enhanced center food detection with adaptive threshold"""
        target = abs(self.target)
        threshold = self.precision_threshold
        if target >= threshold:
            return 0.0
        else:
            return (1.0 - (target / threshold)) * self.aggression_level
    
    def smell_strong(self):
        """Enhanced strong food signal detection"""
        target_abs = abs(self.target)
        threshold = self.precision_threshold * 2.5
        
        if target_abs >= threshold:
            return 0.0
        elif target_abs <= threshold / 8:
            return 1.0 * self.aggression_level
        else:
            return ((threshold - target_abs) / (threshold * 0.875)) * self.aggression_level
    
    def smell_weak(self):
        return max(0.0, 1.0 - self.smell_strong())
    
    def path_clear(self):
        """Enhanced path clearance with adaptive sensitivity"""
        base_clearance = self.front_far() * 0.65 + self.left_far() * 0.175 + self.right_far() * 0.175
        return min(1.0, base_clearance * self.sensitivity_multiplier)
    
    def path_blocked(self):
        return max(0.0, 1.0 - self.path_clear())

if __name__ == '__main__':
    print(f"🤖 Starting Intelligent Anti-Oscillation Fuzzy Robot Simulation...")
    print(f"📍 Robot Start Position: {START_POINT}")
    print(f"🎯 Food Position: (500, 50)")
    print(f"⏱️  Max Ticks: {MAX_TICK}")
    print(f"🎮 Opening Simulation Window...")
    
    # Force window to be visible and configurable
    try:
        Window.size = (800, 600)
        Window.top = 100
        Window.left = 100
        print("🖥️  Window settings applied")
    except Exception as e:
        print(f"⚠️  Could not set window properties: {e}")
    
    app = PySimbotApp(FuzzyRobot, ROBOT_NUM, mapPath=MAP_FILE, interval=TIME_INTERVAL, maxtick=MAX_TICK)
    print("🚀 Starting intelligent simulation loop...")
    app.run()
    print("✅ Simulation completed!")