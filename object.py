import numpy as np
import taichi as ti

import constants as c

@ti.dataclass
class Object:
    pos: ti.math.vec2
    vel: ti.math.vec2
    acc: ti.math.vec2
    rad: float
    hits: int
    dist: float

    def radius_px(self):
        # self.rad is in range [-1, 1], so scale that to match the min and max
        # size for an object.
        min_val = c.MIN_OBJ_RADIUS_PX
        val_range = c.MAX_OBJ_RADIUS_PX - c.MIN_OBJ_RADIUS_PX
        return min_val + ((self.rad + 1) / 2) * val_range

    @ti.func
    def radius(self):
        # self.rad is in range [-1, 1], so scale that to match the min and max
        # size for an object.
        min_val = c.MIN_OBJ_RADIUS
        val_range = c.MAX_OBJ_RADIUS - c.MIN_OBJ_RADIUS
        return min_val + ((self.rad + 1) / 2) * val_range

    @ti.func
    def next_state(self):
        new_pos = self.pos + c.DT * self.vel
        new_dist = self.dist + ti.math.distance(self.pos, new_pos)
        drag = 100 * (self.vel ** 2) * (self.radius() ** 2)
        new_vel = (1 - c.VISCOSITY) * self.vel + c.DT * (self.acc - drag)
        return Object(new_pos, new_vel, 0.0, self.rad, self.hits, new_dist)
