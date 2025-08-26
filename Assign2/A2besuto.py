#!/usr/bin/python3

import os, sys
import random

from pysimbotlib.Window import PySimbotApp
from pysimbotlib.Robot import Robot
from kivy.core.window import Window
from kivy.logger import Logger
import math

# Number of robot that will be run
ROBOT_NUM = 1

# Delay between update (default: 1/60 (or 60 frame per sec))
TIME_INTERVAL = 1.0/60 #10frame per second 

# Max tick
MAX_TICK = 3000

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

        # initial list of rules
        rules = list()
        turns = list()
        moves = list()

        # 1. if front (0,1,7) are far and smell on the left **then** move moderately(5) and turn left (-30)
        # 2. if front (0,1,7) are far and smell on the right **then** move moderately(5) and turn right(-30)
        # ***(1-2 imply to straight line toward the target)***
        rules.append(self.mf_far(0) * self.mf_far(1) * self.mf_far(7) * self.mf_target('l'))
        turns.append(-30)
        moves.append(5)

        rules.append(self.mf_far(0) * self.mf_far(1) * self.mf_far(7) * self.mf_target('r'))
        turns.append(30)
        moves.append(5)

        # 3. if  left rear (6) is near **then turn** (10) a bit to right side
        # 4. if  right rear (2) is near **then turn** (-10) a bit to left side 
        # ***(3-4 avoiding turning straight to the wall while move in parallel more like adjusting position so no need to move)***
        rules.append(self.mf_near(6))
        turns.append(10)
        moves.append(1)

        rules.append(self.mf_near(2))
        turns.append(-10)
        moves.append(1)

        # 5. if front is near (0) but front left (7) is far (left have space) and smell left then turn left (-30) and move a little (2) to turn
        # 6. if front is near (0) but front right (1) is far (right have space)  and smell right then turn right (30) and move a little (2) to turn
        # ***(5-6 avoid incoming wall which if  there is no available  space in 45 degree direction then it will cancel out and  it will utilize other policy***
        rules.append(self.mf_near(0) * self.mf_far(7))
        turns.append(-60) # favor left side if the dilemma appear
        moves.append(3)

        rules.append(self.mf_near(0) * self.mf_far(1))
        turns.append(30)
        moves.append(3)

        # 7. if right rear is far then turn 45 and move 2
        # 8. if left rear is far then turn -40 and move 2
        # (7-8 is making model favor turning  to blank  space)
        rules.append(self.mf_far(6) * self.mf_far(7))
        turns.append(-40)
        moves.append(2)

        rules.append(self.mf_far(2) * self.mf_far(1))
        turns.append(40)
        moves.append(2)

        #if front is so clear get speed boost
        rules.append(self.mf_far(0))
        turns.append(0)
        moves.append(2)

        ans_turn = 0.0
        ans_move = 0.0
        for r, t, m in zip(rules, turns, moves):
            ans_turn += t * r
            ans_move += m * r

        print(ans_turn, ans_move)

        self.turn(ans_turn)
        self.move(ans_move)
        
    def mf_far(self, ir):
        distance = self.ir_values[ir]
        lower_bound = 5
        higher_bound = 20
        if distance>=higher_bound:
            return 1
        elif distance<lower_bound:
            return 0
        else:
            return (higher_bound - distance)/abs(higher_bound - lower_bound)
        
    def mf_near(self, ir):
        distance = self.ir_values[ir]
        lower_bound = 10
        higher_bound = 25
        if distance>=higher_bound:
            return 0
        elif distance<lower_bound:
            return 1
        else:
            return (higher_bound - distance)/abs(higher_bound - lower_bound)
        
    def mf_target(self, mode):
        # mode: ('l','r')
        if self.target < 180 and mode == 'r':
            return self.target/180
        elif self.target >= 180 and  mode == 'l':
            return (self.target-180)/180
        else:
            return 0

if __name__ == '__main__':
    app = PySimbotApp(FuzzyRobot, ROBOT_NUM, mapPath=MAP_FILE, interval=TIME_INTERVAL, maxtick=MAX_TICK)
    app.run()