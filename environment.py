import math
import matplotlib.pyplot as plt
import random
from island import Island
from ship import Ship


X_LOWER_BOUND = 0
X_UPPER_BOUND = 10
Y_LOWER_BOUND = 0
Y_UPPER_BOUND = 10


class Environment(object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['left', 'right', 'idle']
        self.n_actions = len(self.action_space)
        self.n_features = 4

        self.prev_dist = 0
        self.prev_i_dist = 0

        self.flag = Island()
        self.ship = Ship()
        self.island = Island()

        self.build_environment()

    def get_state(self):
        a = self.ship.getCoords()
        b = self.island.getCoords()
        c = self.flag.getCoords()
        dxi, dyi = b[0] - a[0], b[1] - a[1]
        dxf, dyf = c[0] - a[0], c[1] - a[1]
        f_angle = math.atan2(dyf, dxf)
        i_angle = math.atan2(dyi, dxi)
        return [f_angle - self.ship.direction, (dxf ** 2 + dyf ** 2) ** 0.5,
                i_angle - self.ship.direction, (dxi ** 2 + dyi ** 2) ** 0.5]

    def build_environment(self):
        rate = random.uniform(0.3, 0.5)
        dist = random.uniform(7, 15)
        angle = random.uniform(0, 2 * math.pi)
        self.flag = Island(0, 0, 1)
        self.ship = Ship(dist * math.cos(angle), dist * math.sin(angle))
        self.island = Island(rate * dist * math.cos(angle), rate * dist * math.sin(angle))
        a = self.ship.getCoords()
        c = self.flag.getCoords()
        dxf, dyf = c[0] - a[0], c[1] - a[1]
        f_angle = math.atan2(dyf, dxf)
        self.ship.direction = f_angle
        self.prev_dist = self.flag.get_dist(self.ship.x, self.ship.y)
        self.prev_i_dist = self.island.get_dist(self.ship.x, self.ship.y)

    def reset(self):
        self.build_environment()
        return self.get_state()

    def step(self, action):
        dt = 1

        if action == 0:
            self.ship.direction += math.pi / 6
        elif action == 1:
            self.ship.direction -= math.pi / 6

        self.ship.move(dt)

        self.ship.add_position()

        next_state = self.get_state()

        if self.flag.belongs_to_boarder(self.ship.x, self.ship.y):
            reward = 20000
            done = True

        elif self.island.belongs_to_boarder(self.ship.x, self.ship.y):
            reward = -10000
            done = True

        elif len(self.ship.get_positions()) > 20:
            done = True
            reward = -1000 * (self.prev_dist - self.flag.get_dist(self.ship.x, self.ship.y)) ** 2
        else:
            reward = (self.prev_dist - self.flag.get_dist(self.ship.x, self.ship.y)) ** 3
            reward -= 0.1 * (self.prev_i_dist - self.island.get_dist(self.ship.x, self.ship.y)) ** 3
            if reward < 0:
                reward *= 10
            done = False

        self.prev_dist = self.flag.get_dist(self.ship.x, self.ship.y)
        self.prev_i_dist = self.island.get_dist(self.ship.x, self.ship.y)

        return next_state, reward, done

    def draw_map(self):
        field, ax = plt.subplots()
        ax.set(xlim=(X_UPPER_BOUND * -2, X_UPPER_BOUND * 2),
               ylim=(Y_UPPER_BOUND * -2, Y_UPPER_BOUND * 2))
        ax.set_aspect(1)
        ax.add_artist(self.island.draw_island('green'))
        ax.add_artist(self.flag.draw_island())
        pos = self.ship.get_positions()
        for i in range(len(pos) - 1):
            ax.plot([pos[i][0], pos[i + 1][0]], [pos[i][1], pos[i + 1][1]], marker='o', color='r')
        plt.show()


if __name__ == '__main__':
    env = Environment()
