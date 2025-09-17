#!/usr/bin/python3

from pysimbotlib.core import PySimbotApp, Simbot, Robot, Util
from kivy.logger import Logger
from kivy.config import Config

import random
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# # Force the program to show user's log only for "info" level or more. The info log will be disabled.
# Config.set('kivy', 'log_level', 'debug')
Config.set('graphics', 'maxfps', 10)
mutation_rate = 0.02  # 1% mutation rate
pop = 100
keep_best = 1
collision_penalty = 1

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

    def update(self):
        ''' Update method which will be called each frame
        '''        
        self.ir_values = self.distance()
        self.S0, self.S1, self.S2, self.S3, self.S4, self.S5, self.S6, self.S7 = self.ir_values
        self.target = self.smell()

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

        self.turn(answerTurn)
        self.move(answerMove)
    
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

# initializing next generation robot list
next_gen_robots = list()

max_fitness_history = []
avg_fitness_history = []
generation_numbers = []

def before_simulation(simbot: Simbot):
    for robot in simbot.robots:
        # random RULES value for the first generation
        if simbot.simulation_count == 0:
            Logger.info("GA: initial population")
            for i, RULE in enumerate(robot.RULES):
                for k in range(len(RULE)):
                    robot.RULES[i][k] = random.randrange(256)
        # used the calculated RULES value from the previous generation
        else:
            Logger.info("GA: copy the rules from previous generation")
            for simbot_robot, robot_from_last_gen in zip(simbot.robots, next_gen_robots):
                simbot_robot.RULES = robot_from_last_gen.RULES

def after_simulation(simbot: Simbot):
    Logger.info("GA: Start GA Process ...")

    # There are some simbot and robot calcalated statistics and property during simulation
    # - simbot.score
    # - simbot.simulation_count
    # - simbot.eat_count
    # - simbot.food_move_count
    # - simbot.score
    # - simbot.scoreStr

    # - simbot.robot[0].eat_count
    # - simbot.robot[0].collision_count
    # - simbot.robot[0].color

    fitness_values = []

    # evaluation – compute fitness values here
    for robot in simbot.robots:
        # the closer the food the more reward
        food_pos = simbot.objectives[0].pos
        robot_pos = robot.pos
        distance = Util.distance(food_pos, robot_pos)
        near = 500 - int(distance)
        # deduct point from collision
        collision_loss = robot.collision_count * collision_penalty
        # eat bonus
        obj_bonus =  robot.eat_count * 10
        # out of comfort zone
        # far_from_home = Util.distance(simbot.robot_start_pos,robot_pos) - 100
        robot.fitness = near - collision_loss + obj_bonus #+ far_from_home

    # descending sort and rank: the best 10 will be on the list at index 0 to 9
    simbot.robots.sort(key=lambda robot: robot.fitness, reverse=True)
    for robot in simbot.robots:
        fitness_values.append(robot.fitness)

    Logger.info(f"fitness sorted: {fitness_values}")

    # empty the list
    next_gen_robots.clear()



    # adding the best to the next generation.
    next_gen_robots.extend(simbot.robots[:keep_best])
    Logger.info(f"number of elite: {len(next_gen_robots)} with top fitness: {simbot.robots[0].fitness}")
    

    num_robots = len(simbot.robots)

    def softmax(x):
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum(axis=-1) 

    def roulette_select(prop):
        index = np.random.choice(range(num_robots), p=prop)
        return simbot.robots[int(index)]

    def random_select(prop):
        # natural select kill lowest fitness before breeding
        # index = random.randrange(num_robots - keep_best)
        index = random.randrange(num_robots)
        return simbot.robots[index]

    # dcreate offspring with n = population size - keep best
    prop = softmax(np.array(fitness_values))
    for _ in range(num_robots - keep_best):
        select1 = roulette_select(prop)   # elite also  can be parent for offspring
        select2 = roulette_select(prop)   

        # prevent self cross over and infinite loop so normal random is needed when some prop is over 70% from softmax
        while select1 == select2:
            select2 = random_select(prop)

        # Doing crossover
        #     using next_gen_robots for temporary keep the offsprings, later they will be copy
        #     to the robots
        
        # initial parameter
        offspring = StupidRobot()
        offspring.RULES = [[0] * offspring.RULE_LENGTH for _ in range(offspring.NUM_RULES)]
        # Hints on making Crossover 
        # Crossover
        RULE_LENGTH = 11
        NUM_RULES = 10
        crossover_point = random.randint(1,RULE_LENGTH*NUM_RULES - 1)
        # First part from parent1, second part from parent2
        #  list slicing will use number behide ":"" as upper bound and the number in front of ":" is starting point

        offspring.RULES[:crossover_point//RULE_LENGTH] = select1.RULES[:crossover_point//RULE_LENGTH]
        offspring.RULES[crossover_point//RULE_LENGTH][:crossover_point%RULE_LENGTH] = select1.RULES[crossover_point//RULE_LENGTH][:crossover_point%RULE_LENGTH]
        offspring.RULES[crossover_point//RULE_LENGTH][crossover_point%RULE_LENGTH:] = select2.RULES[crossover_point//RULE_LENGTH][crossover_point%RULE_LENGTH:]
        offspring.RULES[crossover_point//RULE_LENGTH+1:] = select2.RULES[crossover_point//RULE_LENGTH+1:]

        next_gen_robots.append(offspring)

    # Doing mutation
    #     generally scan for all next_gen_robots we have created, and with very low
    #     propability, change one byte to a new random value.
    
    # for all robot except elite one
    for robot in next_gen_robots[keep_best:]:  # Skip the best robot (elite) to save elite
        # go through each gene
        for i in range(robot.NUM_RULES):
            # select point of mutation
            for j in range(robot.RULE_LENGTH):
                # if random number lies in mutation rate region (lef tside) then randomly change it with same logic as random initialize
                if random.random() < mutation_rate:
                    robot.RULES[i][j] = random.randrange(256)

    # write the best rule to file
    write_rule(simbot.robots[0], "best_gen{0}.csv".format(simbot.simulation_count))

    max_fitness = max(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)

    max_fitness_history.append(max_fitness)
    avg_fitness_history.append(avg_fitness)
    generation_numbers.append(simbot.simulation_count)

    Logger.info(f"GA: Generation {simbot.simulation_count} - Max Fitness: {max_fitness:.2f}, Avg Fitness: {avg_fitness:.2f}")

    # Plot learning curves every 10 generations or at specific milestones
    if simbot.simulation_count % 5 == 0 or simbot.simulation_count == 1:
        plot_learning_curves()

def plot_learning_curves():
    """Plot the learning curves showing max and average fitness over generations"""
    if len(generation_numbers) < 2:
        return  # Need at least 2 points to plot

    plt.figure(figsize=(10, 6))

    # Plot maximum fitness
    plt.plot(generation_numbers, max_fitness_history, 'b-', label='Maximum Fitness', linewidth=2)

    # Plot average fitness
    plt.plot(generation_numbers, avg_fitness_history, 'g-', label='Average Fitness', linewidth=1.5)

    # Fill area between max and avg for better visualization
    plt.fill_between(generation_numbers, avg_fitness_history, max_fitness_history,
                     alpha=0.2, color='yellow')

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('GA Learning Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    latest_gen = generation_numbers[-1]
    latest_max = max_fitness_history[-1]
    latest_avg = avg_fitness_history[-1]

    stats_text = f"Generation {latest_gen}\nMax: {latest_max:.1f}\nAvg: {latest_avg:.1f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'learning_curve_gen_{latest_gen}.png', dpi=100)
    plt.close()  # Close to prevent memory issues with many plots

    Logger.info(f"GA: Learning curve saved as learning_curve_gen_{latest_gen}.png")

if __name__ == '__main__':

    app = PySimbotApp(robot_cls=StupidRobot, 
                        num_robots=pop,
                        theme='default',
                        simulation_forever=True,
                        max_tick=200,
                        interval=1/100.0,
                        food_move_after_eat=False,
                        customfn_before_simulation=before_simulation, 
                        customfn_after_simulation=after_simulation)
    app.run()