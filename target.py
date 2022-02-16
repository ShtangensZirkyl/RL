import matplotlib.pyplot as plt
import math


class Target:
    def __init__(self, xcentr=5, ycentr=5, radius=1, angle=math.pi / 2, vel=10):
        self.x = xcentr
        self.y = ycentr
        self.radius = radius
        self.direction = angle
        self.v = vel

    def belongs_to_boarder(self, x, y):
        dist = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5
        if dist <= self.radius:
            return True
        else:
            return False

    def get_dist(self, x, y):
        return ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5 - self.radius

    def draw(self, color='blue'):
        return plt.Circle((self.x, self.y), self.radius, color=color)

    def getCoords(self):
        return [self.x, self.y]

    def move(self, dt=1):
        self.x += dt * self.v * math.cos(self.direction)
        self.y += dt * self.v * math.sin(self.direction)