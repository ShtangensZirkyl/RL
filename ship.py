import math


class Ship:
    def __init__(self, x=1, y=5, v=1, direction=0):
        self.x = x
        self.y = y
        self.v = v
        self.direction = direction
        self.prev_d = self.direction
        self.positions = [[self.x, self.y]]
        self.cum_d = 0

    def move(self, dt=1):
        self.x += dt * self.v * math.cos(self.direction)
        self.y += dt * self.v * math.sin(self.direction)
        self.cum_d += abs(self.direction - self.prev_d)
        self.prev_d = self.direction

    def getCoords(self):
        return [self.x, self.y]

    def add_position(self):
        self.positions.append([self.x, self.y])

    def get_positions(self):
        return self.positions
