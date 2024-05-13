"""Representation for Ball objects and their controllers.

Ball is a class representing the physical state of a single ball in the
simulation. Its controller takes View objects as input and produces Reaction
objects as output. Constants indicate what each value in these vectors
represent to the simulation.
"""

from itertools import count

import taichi as ti

import constants as c

# Index values for the View and Reaction vectors.
index = count(0)
# This ball's state
POS_X = next(index)
POS_Y = next(index)
VEL_X = next(index)
VEL_Y = next(index)
# The slope of the topography at this ball's position.
SLP_X = next(index)
SLP_Y = next(index)
# State of the nearest ball.
NRST_POS_X = next(index)
NRST_POS_Y = next(index)
NRST_VEL_X = next(index)
NRST_VEL_Y = next(index)
# Average positions of all balls.
AVG_POS_X = next(index)
AVG_POS_Y = next(index)
NUM_INPUTS = next(index)

index = count(0)
# Desired acceleration
ACC_X = next(index)
ACC_Y = next(index)
# Desired ball radius
RAD = next(index)
NUM_OUTPUTS = next(index)

View = ti.types.vector(NUM_INPUTS, float)
Reaction = ti.types.vector(NUM_OUTPUTS, float)


@ti.dataclass
class Ball:
    # Per-ball physical state
    pos: ti.math.vec2
    vel: ti.math.vec2
    acc: ti.math.vec2
    rad: float
    # Per-ball metrics (aggregated in simulator.py)
    hits: int
    lin_disp: float
    ang_disp: float

    def radius_px(self):
        """Convert RAD value to radius in pixels for rendering."""
        # self.rad is in range [-1, 1], so scale that to match the min and max
        # size for an object.
        min_val = c.MIN_BALL_RADIUS_PX
        val_range = c.MAX_BALL_RADIUS_PX - c.MIN_BALL_RADIUS_PX
        return min_val + ((self.rad + 1) / 2) * val_range

    @ti.func
    def radius(self):
        """Convert RAD value to radius at simulation scale for simulating."""
        # self.rad is in range [-1, 1], so scale that to match the min and max
        # size for an object.
        min_val = c.MIN_BALL_RADIUS
        val_range = c.MAX_BALL_RADIUS - c.MIN_BALL_RADIUS
        return min_val + ((self.rad + 1) / 2) * val_range

    @ti.func
    def next_state(self, prev_vel):
        """Compute next physical state using acceleration and velocity."""
        new_pos = self.pos + c.DT * self.vel
        new_lin_disp = self.lin_disp + ti.math.distance(self.pos, new_pos)
        theta = ti.math.acos(self.vel.normalized().dot(prev_vel.normalized()))
        new_ang_disp = self.ang_disp + ti.select(ti.math.isnan(theta), 0.0, theta)
        drag = 100 * (self.vel ** 2) * (self.radius() ** 2)
        new_vel = (1 - c.VISCOSITY) * self.vel + c.DT * (self.acc - drag)
        return Ball(new_pos, new_vel, 0.0, self.rad, self.hits,
                    new_lin_disp, new_ang_disp)

