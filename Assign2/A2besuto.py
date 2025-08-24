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

# def get_adjacent(l: list, n: int, k: int = 2):
#     temp = [l[n]]
#     for i in range(1,k+1):
#         temp.append(l[(n+i)%8])
#         temp.append(l[(n-i)%8])
#     return temp

# def mf_free(self, target, threadshold = 0.5):
#     # filter_ir = get_adjacent(self.ir_values, n = target, k = 1)
#     filter_ir = self.ir_values
#     degree = sum(filter_ir)/len(filter_ir)
#     if degree < threadshold:
#         return 0
#     return  ((degree / 100) - threadshold)/(1-threadshold)

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

        # go forward base on front distance
        rules.append(self.mf_far(0))
        turns.append(0)
        moves.append(5)

        # go forward base on front distance toward food
        target_sensor = math.floor(self.target/45)
        rules.append(max(
             self.mf_approaching(0),
            0
            # self.mf_near(0)
              )
              )
        turns.append(self.target)
        moves.append(2)
        

        # avoid left
        rules.append(max(
            self.mf_near(6),
            self.mf_near(7),
            # self.mf_near(5)
                  ))
        turns.append(45)
        moves.append(3)

         # avoid right
        rules.append(max(
            self.mf_near(2),
            self.mf_near(1),
            # self.mf_near(3)
                  ))
        turns.append(-45)
        moves.append(3)
        # rules.append(
        #     self.mf_near(0)
        #       )
        # turns.append(90)
        # moves.append(-2)

        # rules.append(
        #     self.mf_far(math.floor(self.target/45))
        #           )
        # turns.append(self.target)
        # moves.append(1)


        # # # turn left if food is left
        # rules.append(self.mf_target('l')*self.mf_far(1)*self.mf_far(2)*self.mf_far(3))
        # turns.append(self.target)
        # moves.append(3)

        # # # turn left if food is right
        # rules.append(self.mf_target('r')*self.mf_far(7)*self.mf_far(6)*self.mf_far(5))
        # turns.append(self.target)
        # moves.append(3)

        # # # turn back if food is back
        # rules.append(self.mf_target('b')*self.mf_far(3)*self.mf_far(4)*self.mf_far(5))
        # turns.append(self.target)
        # moves.append(3)

        # escape from obstrucle
        # for i, v in enumerate(self.ir_values):
        #     if v == max(self.ir_values):
        #         rules.append(self.mf_near(i)*(self.mf_near(0) * self.mf_near(1) * self.mf_near(7)))
        #         turns.append(45 * (i-4))
        #         moves.append(10)
        
        ans_turn = 0.0
        ans_move = 0.0
        for r, t, m in zip(rules, turns, moves):
            ans_turn += t * r
            ans_move += m * r

        print(ans_turn, ans_move)

        self.turn(ans_turn)
        self.move(ans_move)

    def mf_free(self, threadshold = 0.5):
        degree = sum(self.ir_values)/len(self.ir_values)
        if degree < threadshold:
            return 0
        return  ((degree / 100) - threadshold)/(1-threadshold)
        
    def mf_far(self, ir):
        distance = self.ir_values[ir]
        lower_bound = 5
        higher_bound = 25
        if distance>=higher_bound:
            return 1
        elif distance<lower_bound:
            return 0
        else:
            return (higher_bound - distance)/abs(higher_bound - lower_bound)
        
    def mf_near(self, ir):
        distance = self.ir_values[ir]
        lower_bound = 5
        higher_bound = 20
        if distance>=higher_bound:
            return 0
        elif distance<lower_bound:
            return 1
        else:
            return (higher_bound - distance)/abs(higher_bound - lower_bound)
        
        
    def mf_approaching(self, ir):
        distance = self.ir_values[ir]
        lower_bound = 60
        higher_bound = 30
        if distance>lower_bound:
            return 0
        elif distance<higher_bound:
            return 1
        else:
            return (lower_bound - distance)/abs(higher_bound - lower_bound)
        
    def mf_target(self, mode):
        # mode: ('f','l','b','r')
        #  at front
        if self.target > 315 or self.target <= 45:
            if mode == 'f':
                # calculate
                if self.target > 315:
                    return (self.target-315)/90
                else:
                    return self.target/90
        else:
            lower_bound = 45
            higher_bound = 135
            if mode == 'b':
                lower_bound+=90
                higher_bound+=90
            if mode == 'l':
                lower_bound+=180
                higher_bound+=180
            if self.target > lower_bound and self.target <= higher_bound:
                return (self.target-lower_bound)/90
        return 0

if __name__ == '__main__':
    app = PySimbotApp(FuzzyRobot, ROBOT_NUM, mapPath=MAP_FILE, interval=TIME_INTERVAL, maxtick=MAX_TICK)
    app.run()