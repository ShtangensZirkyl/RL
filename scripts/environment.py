import math
import matplotlib.pyplot as plt
import random
from island import Island
from ship import Ship
from target import Target
import pandas as pd
import os

X_LOWER_BOUND = 0
X_UPPER_BOUND = 10
Y_LOWER_BOUND = 0
Y_UPPER_BOUND = 10
STEPS_LIMIT = 25


class Environment:
    def __init__(self, path='../data/tests.csv'):
        self.action_space = ['left', 'right', 'idle']
        self.n_actions = len(self.action_space)
        self.n_features = 7

        self.prev_dist = 0
        self.prev_i_dist = 0

        self.flag = None
        self.ship = None
        self.target = None
        self.dt = 0.1
        self.steps_limit = 100

        self.index = -1
        self.data = pd.read_csv(path)
        self.data = self.data.drop(['Unnamed: 0'], axis=1)
        self.data = self.data.sample(n=100000)

        self.build_environment()

    def build_environment(self):
        self.index += 1
        # rate = random.uniform(0.4, 0.6)
        # dist = random.uniform(5, 15)
        # angle = random.uniform(0, 2 * math.pi)
        if self.index == len(self.data):
            self.index = 0
        self.ship = Ship(0, 0, self.data['speed'].values[self.index])
        x_flag = 0
        y_flag = self.ship.v
        self.flag = Island(x_flag, y_flag, 1)
        self.steps_limit = 2.5 / self.dt
        self.target = Target(self.data['dist1'].values[self.index] * math.cos(math.pi / 2 - math.radians(self.data['peleng1'].values[self.index])),
                             self.data['dist1'].values[self.index] * math.sin(math.pi / 2 - math.radians(self.data['peleng1'].values[self.index])),
                             1, math.pi / 2 - math.radians(self.data['course1'].values[self.index]),
                             self.data['speed1'].values[self.index])
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
        return [a[0], a[1], self.ship.direction,
                c[0], c[1],
                # f_angle - self.ship.direction,
                self.target.get_dist(a[0], a[1]),
                # i_angle - self.ship.direction,
                self.flag.get_dist(a[0], a[1])]

    def reset(self):
        self.build_environment()
        return self.get_state()

    def angle_to_destination(self):
        x, y = self.ship.x - self.flag.xcentr, self.ship.y - self.flag.ycentr
        return abs(math.atan2(y, x))

    def step(self, action):
        angle_change_reward = 0

        if action == 0:             # turn left
            self.ship.direction += math.pi / 12
            angle_change_reward = 2
        elif action == 1:           # turn right
            self.ship.direction -= math.pi / 12
            angle_change_reward = 1

        self.ship.move(self.dt)
        self.target.move(self.dt)

        self.ship.add_position()

        next_state = self.get_state()

        if self.flag.belongs_to_boarder(self.ship.x, self.ship.y):
            reward = 500
            # reward += (self.steps_limit - len(self.ship.get_positions())) * 100
            done = True

        elif self.target.belongs_to_boarder(self.ship.x, self.ship.y):
            reward = -200
            done = True

        elif len(self.ship.get_positions()) > self.steps_limit:
            done = True
            # reward = -100 * (self.prev_dist - self.flag.get_dist(self.ship.x, self.ship.y)) ** 2
            reward = -150
        else:
            # coefficient = 1000 better
            # reward = 10 * (self.prev_dist - self.flag.get_dist(self.ship.x, self.ship.y)) ** 3
            # reward = -10 * (self.flag.get_dist(self.ship.x, self.ship.y)) ** 3
            # print("Reward for dist " + str(reward))
            # coefficient = 1 better
            # reward = self.angle_to_destination()*0.1
            # print("Reward for ang " + str(reward))
            reward = -angle_change_reward
            # print("Reward for ang change " + str(reward))
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
