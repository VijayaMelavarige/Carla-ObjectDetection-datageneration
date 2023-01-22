import time
import math
import numpy as np



class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def vector3d_to_array(vec3d):
    return np.array([vec3d.x, vec3d.y, vec3d.z])


def degrees_to_radians(degrees):
    return degrees * math.pi / 180

def dist_func(x, y): 
    return sum((x - y)**2)

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()