#!/usr/bin/python3

import os, platform
if platform.system() == "Linux" or platform.system() == "Darwin":
    os.environ["KIVY_VIDEO"] = "ffpyplayer"
    
from pysimbotlib.core import PySimbotApp, Robot
from kivy.config import Config
# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set('kivy', 'log_level', 'info')

import math
def identify_nearest_sensor(target_angle: float) -> float:
    return math.floor(target_angle/45)

def get_turn_degree(target_sensor: int) -> int:
    # right
    if target_sensor in (1,2,3):
        return target_sensor*45
    # left 5 -> 3, 6 -> 2 7 -> 1
    elif target_sensor in (5,6,7):
        return -1*((target_sensor*45)-180)
    elif target_sensor == 0:
        return 0
    else:
        return 180

def get_adjacent_index(index: int, k:int = 2)->list:
    result = [index]
    for i in range(1,k+1):
        result.append((index+i)%8)
        if index-i >= 0:
            result.append(index-i)
        else:
            result.append((8-i))
    return result

def search(L:list, value:float):
    for i, v in  enumerate(L):
        if v == value:
            return i
        
def select_with_index(L:list[float],index:list[int]):
    result = []
    for i,j in enumerate(L):
        if i in index:
            result.append(j)
    return result

def check_distance(sensor_reading: list[int] ,distance:int):
    for numbers in sensor_reading:
        if numbers < distance:
            return False
    return True

class GreedyWalkRobot(Robot):

    def update(self):
        safety_distance = 25
        close_distance = 10
        rear_sensor = [1,7]
        outer_rear = [2,6]
        front_sensor = [0] + rear_sensor + outer_rear
        degree_of_turning = 0.5
        turn_degree = 15
        step = 5
        surrounding_dist = self.distance()
        direction = self.smell()
        front_area = select_with_index(surrounding_dist, front_sensor)
        rear_area = select_with_index(surrounding_dist, rear_sensor)
        outer_rear_area = select_with_index(surrounding_dist, outer_rear)
        print(surrounding_dist)

        if check_distance(front_area, safety_distance*1):
            self.move(step)
            self.turn(direction*degree_of_turning)
            print("toward food")
        elif (not check_distance(rear_area, close_distance) and surrounding_dist[0] > safety_distance):
            print("out of object")
            # this is left
            if surrounding_dist[7] == min(rear_area):
                self.turn(-1*turn_degree)
            elif surrounding_dist[1] == min(rear_area):
                self.turn(turn_degree)
        elif (not check_distance(outer_rear_area, close_distance*1) and surrounding_dist[0] > safety_distance):
            print("out of object")
            # this is left
            mul = 1.75
            # mul = 1
            if surrounding_dist[6] == min(rear_area):
                self.turn(-1*turn_degree*mul)
            else:
                self.turn(turn_degree*mul)
        elif check_distance(rear_area, close_distance) and surrounding_dist[0] > safety_distance:
            self.move(step)
            print("go go")
            # self.move(step/2)
        elif not check_distance(front_area, safety_distance):
            print("bumped")
            if surrounding_dist[7]<surrounding_dist[1]:
                self.turn(turn_degree*5)
            else:
                self.turn(-turn_degree*5)
            self.move(step)
        # else:
        #     self.turn(30)

        # # determine direction of the target
        # front_sensor = [0,1,7,2,6]
        # # go in 4 direction
        # surrounding_dist = self.distance()
        # direction = self.smell()
        # # print(surrounding_dist, direction)
        # # determine quardrant of target
        # block_area = select_with_index(surrounding_dist, front_sensor)
        # # print(block_area)        
        # if any([i <= 10 for i in block_area]) :
        #     avg_dist_left = (surrounding_dist[7] + surrounding_dist[6])/2
        #     avg_dist_right = (surrounding_dist[1] + surrounding_dist[2])/2
        #     if avg_dist_left < 10:
        #         self.turn(60)
        #         print("l")
        #     elif avg_dist_right < 10:
        #         self.turn(-60)
        #         print("r")
        #     else:
        #         self.turn(30)
        #         print("not sure")
        #     surrounding_dist = self.distance()
        #     block_area = select_with_index(surrounding_dist, front_sensor)
        #     if sum(block_area)/len(block_area) < 10:
        #         self.turn(-30)
        #         # self.move(10)
        #     if surrounding_dist[0] > 20:
        #         self.move(10)
        # else:
        #     self.move(3)
        # surrounding_dist = self.distance()
        # block_area = select_with_index(surrounding_dist, front_sensor)
        # if sum(block_area)/len(block_area) > 70:
        #     print("lock on")
        #     self.turn(direction)
        # surrounding_dist = self.distance()

if __name__ == '__main__':
    app = PySimbotApp(robot_cls=GreedyWalkRobot, num_robots=1)
    app.run()