#!/usr/bin/python3

import os, platform
if platform.system() == "Linux" or platform.system() == "Darwin":
    os.environ["KIVY_VIDEO"] = "ffpyplayer"
    
from pysimbotlib.core import PySimbotApp, Robot
from kivy.config import Config
# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set('kivy', 'log_level', 'info')

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
        close_distance = 11
        rear_sensor = [1,7]
        outer_rear = [2,6]
        front_sensor = [0] + rear_sensor
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

if __name__ == '__main__':
    app = PySimbotApp(robot_cls=GreedyWalkRobot, num_robots=1, max_tick=4000)
    app.run()