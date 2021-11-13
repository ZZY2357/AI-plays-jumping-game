import pyxel
import numpy as np
import math
import random
import time

pyxel.init(100, 60)

class Vector:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def add(self, v):
        return Vector(self.x + v.x, self.y + v.y)

    def mult(self, n):
        return Vector(self.x * n, self.y * n)

    def div(self, n):
        return Vector(self.x / n, self.y / n)

class NN:
    def __init__(self, inodes, hnodes, onodes):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        self.wih = np.random.randn(self.hnodes, self.inodes) - 0.5
        self.who = np.random.randn(self.onodes, self.hnodes) - 0.5

    def activation_function(self, i):
        return (abs(i) + i) / 2

    def predict(self, input):
        inputs = np.array(input, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

class Box:
    def __init__(self, x, y):
        self.pos = Vector(x, y)
        self.v = 3
    
    def update(self):
        if self.pos.x + 30 < 0:
            self.pos.x = pyxel.width + random.randint(30, 300)
            self.v += 0.5
        self.pos.x -= self.v

    def draw(self):
        pyxel.rect(self.pos.x, self.pos.y, 5, 5, 11)

class Agent:
    def __init__(self, x, y):
        self.pos = Vector(x, y)
        self.velo = Vector()
        self.acce = Vector()
        self.birth_time = time.time()
        self.is_dead = False
        self.fitness = 0

        self.nn = NN(3, 5, 3)

    def apply_force(self, v):
        self.acce = self.acce.add(v)

    def jump(self):
        if self.pos.y >= box.pos.y - 5:
            self.apply_force(Vector(0, -10))
            self.fitness -= 1

    def update(self):
        method = self.nn.predict([self.pos.x, self.pos.y, box.pos.x])
        left_right = (method[1][0] - method[0][0])
        if left_right > 3: left_right = 3
        if left_right < -3: left_right = -3
        self.apply_force(Vector(left_right, 0))
        if method[2][0] > 50:
            self.jump()

        self.apply_force(Vector(0, 3))
        self.apply_force(self.velo.mult(-0.1))

        self.velo = self.velo.add(self.acce)
        self.pos = self.pos.add(self.velo)
        self.acce = self.acce.mult(0)

        if self.pos.y >= box.pos.y - 5:
            self.apply_force(Vector(0, -self.velo.y))
            self.pos.y = box.pos.y - 5
        if self.pos.x < 0:
            self.pos.x = 0
        if self.pos.x + 5 > pyxel.width:
            self.pos.x = pyxel.width - 5

        if self.pos.x + 5 >= box.pos.x and self.pos.x <= box.pos.x + 5 and \
            self.pos.y + 10 >= box.pos.y and self.pos.y <= box.pos.y + 5:
            self.is_dead = True
            self.fitness += time.time() - self.birth_time

    def draw(self):
        pyxel.rect(self.pos.x, self.pos.y, 5, 10, 12)

start_time = time.time()
box = Box(pyxel.width + 30, 50)
agents = []
mating_pool = []

for i in range(100):
    agents.append(Agent(4, 45))

def end_epcho():
    global start_time, agents
    end_time = time.time()
    for i in agents:
        count = math.floor(i.fitness / (end_time - start_time) * 100)
        for j in range(count):
            mating_pool.append(i)
    for i in range(len(agents)):
        f = random.choice(mating_pool)
        m = random.choice(mating_pool)
        child = Agent(4, 45)
        child.nn.wih = (f.nn.wih + m.nn.wih) / 2
        child.nn.who = (f.nn.who + m.nn.who) / 2
        if random.uniform(0, 1) > 0.3:
            # mutate
            child.nn.wih += (np.random.randn(5, 3) - 0.5) * 100
            child.nn.who += (np.random.randn(3, 5) - 0.5) * 100
        agents[i] = child
    mating_pool.clear()

    box.pos = Vector(pyxel.width + 30, 50)
    box.v = 3
    start_time = time.time()

def update():
    if pyxel.btnp(pyxel.KEY_Q):
        pyxel.quit()
    box.update()

    death = 0
    for i in agents:
        if i.is_dead:
            death += 1
        else:
            i.update()
    if death >= len(agents):
        end_epcho()

def draw():
    pyxel.cls(0)
    box.draw()

    for i in agents:
        if not i.is_dead:
            i.draw()

pyxel.run(update, draw)
