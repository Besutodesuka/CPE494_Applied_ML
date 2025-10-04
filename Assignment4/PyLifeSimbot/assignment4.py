#!/usr/bin/python3

from pysimbotlib.core import PySimbotApp, Simbot, Robot, Util
from kivy.logger import Logger
from kivy.config import Config

from copy import deepcopy
import random
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

# # Force the program to show user's log only for "info" level or more. The info log will be disabled.
# Config.set('kivy', 'log_level', 'debug')
Config.set('graphics', 'maxfps', 1)

# Global variables for tracking deaths and learning curve
death_records = []
interval_size = 1000  # Record deaths every 1000 ticks
last_plot_tick = 0  # Track last time we plotted

class StupidRobot(Robot):

    RULE_LENGTH = 11
    NUM_RULES = 10


    def __init__(self, **kwarg):
        super(StupidRobot, self).__init__(**kwarg)
        self.RULES = [[0] * self.RULE_LENGTH for _ in range(self.NUM_RULES)]

        # initial list of rules
        self.rules = [0.] * self.NUM_RULES
        self.turns = [0.] * self.NUM_RULES
        self.moves = [0.] * self.NUM_RULES

        self.fitness = 0
        self.lazy_count = 0
        self.headache_count = 0
        self.energy = 300
        self.just_eat = False

    def update(self):
        ''' Update method which will be called each frame
        '''        
        self.ir_values = self.distance()
        self.S0, self.S1, self.S2, self.S3, self.S4, self.S5, self.S6, self.S7 = self.ir_values
        self.target = self.smell_nearest()
        if self.energy < 100 :
            self.set_color(255,255,0,1)
        elif self.energy < 300 :
            self.set_color(255,0,0,1)
        elif self.energy < 500 :
            self.set_color(0,255,0,1)
        elif self.energy < 700 :
            self.set_color(0,0,255,1)
        elif self.energy < 900 :
            self.set_color(255,0,255,1)
        elif self.energy < 1200 :
            self.set_color(0,0,0,1)

        for i, RULE in enumerate(self.RULES):
            self.rules[i] = 1.0
            for k, RULE_VALUE in enumerate(RULE):
                if k < 8:
                    if RULE_VALUE % 5 == 1:
                        if k == 0: self.rules[i] *= self.S0_near()
                        elif k == 1: self.rules[i] *= self.S1_near()
                        elif k == 2: self.rules[i] *= self.S2_near()
                        elif k == 3: self.rules[i] *= self.S3_near()
                        elif k == 4: self.rules[i] *= self.S4_near()
                        elif k == 5: self.rules[i] *= self.S5_near()
                        elif k == 6: self.rules[i] *= self.S6_near()
                        elif k == 7: self.rules[i] *= self.S7_near()
                    elif RULE_VALUE % 5 == 2:
                        if k == 0: self.rules[i] *= self.S0_far()
                        elif k == 1: self.rules[i] *= self.S1_far()
                        elif k == 2: self.rules[i] *= self.S2_far()
                        elif k == 3: self.rules[i] *= self.S3_far()
                        elif k == 4: self.rules[i] *= self.S4_far()
                        elif k == 5: self.rules[i] *= self.S5_far()
                        elif k == 6: self.rules[i] *= self.S6_far()
                        elif k == 7: self.rules[i] *= self.S7_far()
                elif k == 8:
                    temp_val = RULE_VALUE % 6
                    if temp_val == 1: self.rules[i] *= self.smell_left()
                    elif temp_val == 2: self.rules[i] *= self.smell_center()
                    elif temp_val == 3: self.rules[i] *= self.smell_right()
                elif k==9: self.turns[i] = (RULE_VALUE % 181) - 90
                elif k==10: self.moves[i] = (RULE_VALUE % 21) - 10
        
        answerTurn = 0.0
        answerMove = 0.0
        for turn, move, rule in zip(self.turns, self.moves, self.rules):
            answerTurn += turn * rule
            answerMove += move * rule

        if int(answerMove) == 0 and int(answerTurn) == 0:
            self.lazy_count += 1
        
        if int(answerMove) < 0 or abs(answerTurn) > 30:
            self.headache_count += 1

        self.turn(answerTurn)
        self.move(answerMove)

        # every step the robot lost its energy
        self.energy -= 1

        # penalty for lazy behavior
        if int(answerMove) == 0 and int(answerTurn) == 0:
            self.energy -= 40

        # penalty for headache (backward movement or sharp turns)
        if int(answerMove) < 0 or abs(answerTurn) > 45:
            self.energy -= 15

        # if the robot hit, it also lost energy this make it not risk going to walls
        if self.just_hit :
            self.energy -= 5

        # if the robot eat food, it gets some energy back
        if self.just_eat :
            self.energy += 500

        if self.energy < 0 :
            # die and find a new robot
            print (f"Robot died at tick: {self._sm.iteration}")

            # record that there is a dead one
            global death_records, last_plot_tick
            death_records.append(self._sm.iteration)

            # Plot learning curve every 1000 ticks
            current_tick = self._sm.iteration
            if current_tick - last_plot_tick >= interval_size:
                plot_learning_curve(current_tick)
                last_plot_tick = current_tick

            # generate new robot by using genetic operations
            temp = StupidRobot()
            temp = self.generate_new_robot()
            self.RULES = deepcopy(temp.RULES)

            # give this new born some energy
            self.energy = 300

            # reset counters
            self.lazy_count = 0
            self.headache_count = 0
            self.fitness = 0

        self.fitness = self.eat_count * 100 - self.collision_count - self.lazy_count
        
    def generate_new_robot(self):
        simbot = self._sm

        # Tournament selection - select from top 25% based on energy
        def select() -> StupidRobot:
            alive_robots = simbot.robots
            # Tournamet select with size 3. Select random robit let it compete
            size = 3
            tournament = []
            fit_temp = []
            while len(tournament) < size:
                robot = random.choice(alive_robots)
                if robot.fitness not in fit_temp:
                    tournament.append(robot)
                    fit_temp.append(robot.fitness)

            # Sort by energy (fitness)
            tournament.sort(key=lambda r: max(r.fitness, 0), reverse=True)
            return tournament[0]

        select1 = select()
        select2 = select()

        while select1 == select2:
            select2 = select()

        temp = StupidRobot()

        RULE_LENGTH = 11
        NUM_RULES = 10
        crossover_point = random.randint(1, RULE_LENGTH*NUM_RULES-1)

        # Initialize offspring's RULES with deep copies from select1
        for i in range(crossover_point // RULE_LENGTH):
            temp.RULES[i] = list(select1.RULES[i])
        
        # Handle the crossover point within a rule
        rule_idx = crossover_point // RULE_LENGTH
        point_in_rule = crossover_point % RULE_LENGTH

        # Copy first part of the rule from select1, second part from select2
        temp.RULES[rule_idx] = list(select1.RULES[rule_idx][:point_in_rule]) + list(select2.RULES[rule_idx][point_in_rule:])

        # Copy remaining rules from select2
        for i in range(crossover_point // RULE_LENGTH + 1, NUM_RULES):
            temp.RULES[i] = list(select2.RULES[i])

        # Doing mutation
        # With very low probability (0.01), change one byte to a new random value
        mutation_rate = 0.015
        for i in range(self.NUM_RULES):
            for k in range(self.RULE_LENGTH):
                if random.random() < mutation_rate:
                    temp.RULES[i][k] = random.randrange(256)

        return temp

    def S0_near(self):
        if self.S0 <= 0: return 1.0
        elif self.S0 >= 100: return 0.0
        else: return 1 - (self.S0 / 100.0)

    def S0_far(self):
        if self.S0 <= 0: return 0.0
        elif self.S0 >= 100: return 1.0
        else: return self.S0 / 100.0
    
    def S1_near(self):
        if self.S1 <= 0: return 1.0
        elif self.S1 >= 100: return 0.0
        else: return 1 - (self.S1 / 100.0)
    
    def S1_far(self):
        if self.S1 <= 0: return 0.0
        elif self.S1 >= 100: return 1.0
        else: return self.S1 / 100.0
    
    def S2_near(self):
        if self.S2 <= 0: return 1.0
        elif self.S2 >= 100: return 0.0
        else: return 1 - (self.S2 / 100.0)
    
    def S2_far(self):
        if self.S2 <= 0: return 0.0
        elif self.S2 >= 100: return 1.0
        else: return self.S2 / 100.0
    
    def S3_near(self):
        if self.S3 <= 0: return 1.0
        elif self.S3 >= 100: return 0.0
        else: return 1 - (self.S3 / 100.0)
    
    def S3_far(self):
        if self.S3 <= 0: return 0.0
        elif self.S3 >= 100: return 1.0
        else: return self.S3 / 100.0
    
    def S4_near(self):
        if self.S4 <= 0: return 1.0
        elif self.S4 >= 100: return 0.0
        else: return 1 - (self.S4 / 100.0)
    
    def S4_far(self):
        if self.S4 <= 0: return 0.0
        elif self.S4 >= 100: return 1.0
        else: return self.S4 / 100.0
    
    def S5_near(self):
        if self.S5 <= 0: return 1.0
        elif self.S5 >= 100: return 0.0
        else: return 1 - (self.S5 / 100.0)
    
    def S5_far(self):
        if self.S5 <= 0: return 0.0
        elif self.S5 >= 100: return 1.0
        else: return self.S5 / 100.0
    
    def S6_near(self):
        if self.S6 <= 0: return 1.0
        elif self.S6 >= 100: return 0.0
        else: return 1 - (self.S6 / 100.0)
    
    def S6_far(self):
        if self.S6 <= 0: return 0.0
        elif self.S6 >= 100: return 1.0
        else: return self.S6 / 100.0
    
    def S7_near(self):
        if self.S7 <= 0: return 1.0
        elif self.S7 >= 100: return 0.0
        else: return 1 - (self.S7 / 100.0)
    
    def S7_far(self):
        if self.S7 <= 0: return 0.0
        elif self.S7 >= 100: return 1.0
        else: return self.S7 / 100.0
    
    def smell_right(self):
        if self.target >= 45: return 1.0
        elif self.target <= 0: return 0.0
        else: return self.target / 45.0
    
    def smell_left(self):
        if self.target <= -45: return 1.0
        elif self.target >= 0: return 0.0
        else: return 1-(-1*self.target)/45.0
    
    def smell_center(self):
        if self.target <= 45 and self.target >= 0: return self.target / 45.0
        if self.target <= -45 and self.target <= 0: return 1-(-1*self.target)/45.0
        else: return 0.0

def write_rule(robot, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(robot.RULES)

def plot_learning_curve(current_tick=None):
    """Plot the learning curve showing death count over time intervals"""
    global death_records, interval_size

    if len(death_records) == 0:
        print("No deaths recorded during simulation")
        return

    # Determine the number of intervals
    max_tick = max(death_records) if death_records else 100000
    num_intervals = (max_tick // interval_size) + 1

    # Count deaths in each interval
    death_counts = [0] * num_intervals
    for death_tick in death_records:
        interval_idx = death_tick // interval_size
        if interval_idx < num_intervals:
            death_counts[interval_idx] += 1

    # Create x-axis (interval midpoints)
    intervals = [(i * interval_size + interval_size / 2) for i in range(num_intervals)]

    # Plot the learning curve as bar chart
    plt.figure(figsize=(14, 7))

    # Create bar chart with light blue color and edge
    plt.bar(intervals, death_counts, width=interval_size*0.9,
            color='#9DB4C0', edgecolor='#5D7A8C', linewidth=0.5,
            alpha=0.8, label='Deaths per interval')

    # Add smooth trend line (polynomial)
    if len(intervals) > 2:
        z = np.polyfit(intervals, death_counts, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(intervals), max(intervals), 300)
        plt.plot(x_smooth, p(x_smooth), color='#5D7A8C',
                linewidth=3, alpha=0.7, label='Trend line')

    # Style like the example image
    plt.xlabel('Simulation Ticks', fontsize=13, weight='bold')
    plt.ylabel('Count', fontsize=13, weight='bold')

    # Show current tick in title if provided
    title = f'Learning Curve - Deaths over Time'
    if current_tick is not None:
        title += f' (Up to Tick: {current_tick})'
    plt.title(title, fontsize=15, weight='bold', pad=20)

    # Grid with lighter style
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    plt.grid(False, axis='x')

    # Set background color
    ax = plt.gca()
    ax.set_facecolor('#E8ECEF')
    plt.gcf().patch.set_facecolor('white')

    # Legend
    plt.legend(loc='upper right', framealpha=0.9)

    # Add statistics box
    total_deaths = len(death_records)
    avg_deaths = np.mean(death_counts) if len(death_counts) > 0 else 0
    max_deaths = max(death_counts) if len(death_counts) > 0 else 0
    min_deaths = min(death_counts) if len(death_counts) > 0 else 0

    stats_text = f'Total Deaths: {total_deaths}\n'
    stats_text += f'Avg/Interval: {avg_deaths:.1f}\n'
    stats_text += f'Max: {max_deaths} | Min: {min_deaths}'

    plt.text(0.02, 0.98, stats_text,
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#5D7A8C', alpha=0.9, linewidth=1.5),
             fontsize=10, family='monospace')

    plt.tight_layout()

    # Save with tick number in filename for incremental plots
    if current_tick is not None:
        # filename = f'learning_curve_tick_{current_tick}.png'
        filename = 'learning_curve.png'
    else:
        filename = 'learning_curve_final.png'

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Learning curve saved to {filename} (Total deaths: {total_deaths})")
    plt.close()  # Close figure to free memory


def before_simulation(simbot: Simbot):
    for robot in simbot.robots:
        # random RULES value for the first generation
        if simbot.simulation_count == 0:
            Logger.info("GA: initial population")
            for i, RULE in enumerate(robot.RULES):
                for k in range(len(RULE)):
                    robot.RULES[i][k] = random.randrange(256)
        
def after_simulation(simbot: Simbot):

    for robot in simbot.robots:
        robot.fitness = robot.energy

    # descending sort and rank: the best will be at index 0
    simbot.robots.sort(key=lambda robot: robot.fitness, reverse=True)

    write_rule(simbot.robots[0], "best_robot.csv")

    # Plot final learning curve
    print("\n=== Final Learning Curve ===")
    plot_learning_curve(current_tick=None)

if __name__ == '__main__':

    app = PySimbotApp(robot_cls=StupidRobot, 
                        num_robots=20,
                        num_objectives=4,
                        theme='default',
                        simulation_forever=False,
                        max_tick=100000,
                        interval=1/1000.0,
                        food_move_after_eat=True,
                        robot_see_each_other=True,
                        # map="no_wall",
                        customfn_before_simulation=before_simulation, 
                        customfn_after_simulation=after_simulation)
    app.run()