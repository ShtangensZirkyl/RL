import math
import random

import matplotlib.pyplot as plt
import pandas as pd

from island import Island
from ship import Ship
from target import Target

X_LOWER_BOUND = 0
X_UPPER_BOUND = 10
Y_LOWER_BOUND = 0
Y_UPPER_BOUND = 10
STEPS_LIMIT = 25
USE_SYNTHETICS = True


class Environment:
    def __init__(self, path='../data/tests.csv'):
        self.action_space = ['left', 'right', 'idle', 'speed_low', 'speed_high']
        self.n_actions = len(self.action_space)
        self.n_features = 4

        self.prev_dist = 0
        self.prev_i_dist = 0

        self.flag = None
        self.ship = None
        self.target = None
        self.dt = 0.8
        self.steps_limit = 100

        self.index = -1
        if not USE_SYNTHETICS:
            self.data = pd.read_csv(path)
            self.data = self.data.drop(['Unnamed: 0'], axis=1)
            self.data = self.data.sample(n=100000)

        self.build_environment()

    def build_environment(self):
        self.index += 1
        x_flag = 0
        if USE_SYNTHETICS:
            rate = random.uniform(0.4, 0.6)
            dist = random.uniform(5, 15)
            angle = random.uniform(0, 2 * math.pi)
            self.ship = Ship(0, 0, 1)
            self.target = Target(rate * dist * (math.cos(angle)),
                                 rate * dist * (1 + math.sin(angle)),
                                 1, angle - math.pi,
                                 1)
            y_flag = 15
        else:
            if self.index == len(self.data):
                self.index = 0
            self.ship = Ship(0, 0, self.data['speed'].values[self.index])
            y_flag = self.ship.v
            self.target = Target(self.data['dist1'].values[self.index] * math.cos(
                math.pi / 2 - math.radians(self.data['peleng1'].values[self.index])),
                                 self.data['dist1'].values[self.index] * math.sin(
                                     math.pi / 2 - math.radians(self.data['peleng1'].values[self.index])),
                                 1, math.pi / 2 - math.radians(self.data['course1'].values[self.index]),
                                 self.data['speed1'].values[self.index])
        self.flag = Island(x_flag, y_flag, 1)
        self.steps_limit = 1 * y_flag / self.ship.v / self.dt
        a = self.ship.getCoords()
        b = self.flag.getCoords()
        dxf, dyf = b[0] - a[0], b[1] - a[1]
        f_angle = math.atan2(dyf, dxf)
        self.ship.direction = f_angle
        self.prev_dist = self.flag.get_dist(self.ship.x, self.ship.y)
        self.prev_i_dist = self.target.get_dist(self.ship.x, self.ship.y)

    def get_state(self):
        a = self.ship.getCoords()
        b = self.target.getCoords()
        c = self.flag.getCoords()
        dxi, dyi = b[0] - a[0], b[1] - a[1]
        dxf, dyf = c[0] - a[0], c[1] - a[1]
        f_angle = math.atan2(dyf, dxf)
        i_angle = math.atan2(dyi, dxi)
        return [f_angle - self.ship.direction, self.target.get_dist(a[0], a[1]), i_angle - self.ship.direction,
                self.flag.get_dist(a[0], a[1])]

    def reset(self):
        self.build_environment()
        return self.get_state()

    def step(self, action):
        angle_change_reward = 0

        if action == 0:
            self.ship.direction += math.pi / 12
            angle_change_reward = 10
        elif action == 1:
            self.ship.direction -= math.pi / 12
            angle_change_reward = 40
        elif action == 3 and self.ship.v > 0.4:
            self.ship.v -= 0.2
        elif action == 4 and self.ship.v < 1.4:
            self.ship.v += 0.2

        self.ship.move(self.dt)
        self.target.move(self.dt)

        self.ship.add_position()

        next_state = self.get_state()

        if self.flag.belongs_to_boarder(self.ship.x, self.ship.y):
            reward = 20000
            reward += (self.steps_limit - len(self.ship.get_positions())) * 100
            done = True

        elif self.target.belongs_to_boarder(self.ship.x, self.ship.y):
            reward = -10000
            done = True

        elif len(self.ship.get_positions()) > self.steps_limit:
            done = True
            reward = -1000 * (self.prev_dist - self.flag.get_dist(self.ship.x, self.ship.y)) ** 2
            reward -= 8000
        else:
            reward = (self.prev_dist - self.flag.get_dist(self.ship.x, self.ship.y)) ** 3
            # reward -= 0.1 * (self.prev_i_dist - self.island.get_dist(self.ship.x, self.ship.y)) ** 3
            reward -= angle_change_reward
            done = False

        self.prev_dist = self.flag.get_dist(self.ship.x, self.ship.y)
        self.prev_i_dist = self.target.get_dist(self.ship.x, self.ship.y)

        return next_state, reward, done

    def draw_map(self):
        field, ax = plt.subplots()
        ax.set(xlim=(X_UPPER_BOUND * -2, X_UPPER_BOUND * 2),
               ylim=(Y_UPPER_BOUND * -2, Y_UPPER_BOUND * 2))
        ax.set_aspect(1)
        ax.add_artist(self.target.draw('green'))
        ax.add_artist(self.flag.draw_island())
        pos = self.ship.get_positions()
        for i in range(len(pos)):
            ax.plot(pos[i][0], pos[i][1], '.r')
        plt.close()
        return field


if __name__ == '__main__':
    env = Environment()
    env.build_environment()
    env.draw_map()
