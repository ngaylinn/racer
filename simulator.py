"""Simulator, a class to simulate ballial balls rolling on some topography.

This class is instantiated once for this program and reused for all
simulations. It runs c.NUM_WORLDS simulations in parallel, each with
c.NUM_BALLS balls in it, and records some metrics to summarize the simulation.
It uses a double-buffer to track the physical state of all the balls without
allocating memory for the full simulation state history. If the debug param of
the constructor is set to True, the simulator will also capture debug data that
visualize.py will render (this takes more memory and is slightly slower).

To use the simulator, follow these steps:
    - Set the controllers and topographies attributes
    - Call randomize_balls() to set up a random initial configuration.
    - Call step() repeatedly to move the simulation forward incrementally.
    - When finished, call get_metrics() to get the simulation results.
    - OR call run() to combine the last two steps in one call.
"""

import numpy as np
import pandas as pd
import taichi as ti

import ball
from ball import Ball, View, Reaction
import constants as c

@ti.data_oriented
class Simulator:
    def __init__(self, debug=False):
        self.debug = debug

        # Physical state of the simulation:
        self.balls = Ball.field(shape=(c.NUM_WORLDS, c.NUM_BALLS))
        self.next_balls = Ball.field(shape=(c.NUM_WORLDS, c.NUM_BALLS))

        # Controllers for the balls and the topographies for them to roll on.
        self.controllers = None
        self.topographies = None

        # Intermediate data used for calling ball controllers.
        self.avg_pos = ti.math.vec2.field(shape=(c.NUM_WORLDS))
        self.avg_vel = ti.math.vec2.field(shape=(c.NUM_WORLDS))
        if debug:
            self.views = View.field(shape=(c.NUM_WORLDS, c.NUM_BALLS))
            self.reactions = Reaction.field(
                shape=(c.NUM_WORLDS, c.NUM_BALLS))

    def randomize_balls(self):
        """Scatter balls randomly across the worlds."""
        shape = (c.NUM_WORLDS, c.NUM_BALLS)
        count = np.prod(shape)
        # TODO: Would it be better to do this in a kernel?
        self.balls.pos.from_numpy(
            np.random.random(count * 2)
            .astype(np.float32)
            .reshape(shape + (2,)))
        self.balls.vel.fill(0.0)
        self.balls.acc.fill(0.0)
        self.balls.rad.fill(0.0)
        self.balls.hits.fill(0)
        self.balls.ang_disp.fill(0.0)
        self.balls.lin_disp.fill(0.0)

        # TODO: Reove angular displacement trick, since it's not working?
        # To calculate displacement, we peek at the velocity values from the
        # balls in the previous step, so initialize to 0.0.
        self.next_balls.vel.fill(0.0)

    @ti.func
    def project_vec(from_vec, onto_vec):
        """Find the projection of one vector onto another."""
        return (from_vec.dot(onto_vec) / onto_vec.dot(onto_vec)) * onto_vec

    @ti.func
    def topo_index(self, pos: ti.math.vec2, offset: int):
        """Transform coordinates from [0.0, 1.0] to integer map indices."""
        max_x, max_y = c.WORLD_SHAPE - ti.Vector([offset + 1] * 2)
        return ti.math.ivec2(
            ti.math.clamp(int(pos.x * max_x), offset, max_x),
            ti.math.clamp(int(pos.y * max_y), offset, max_y))

    @ti.func
    def acc_from_topography(self, w, pos):
        """Find acceleration due to the topography's slope at pos."""
        # Look up the map coordinates for this pos, but nudge the coordinates
        # "inward" if they're on the edge, so all neighboring cells are valid.
        x, y = self.topo_index(pos, offset=1)
        force = ti.math.vec2(0.0, 0.0)
        # To estimate the normal at cell (x, y), look at the neighboring cells
        # and average their contributions to this vector.
        for x_from in range(x - 1, x + 2):
            for y_from in range(y - 1, y + 2):
                z_delta = (self.topographies.lookup(w, x_from, y_from) -
                           self.topographies.lookup(w, x, y))
                x_delta = (x_from - x) / c.WORLD_SIZE
                y_delta = (y_from - y) / c.WORLD_SIZE
                # At the sample position, this vector is 0, 0, 0, so it doesn't
                # contribute to the normal, and doesn't need special casing.
                force += 10000 * ti.math.vec2(x_delta, y_delta) * z_delta
        # This is an approximation that doesn't handle sharp discontinuities
        # well. So, put a clamp on it to prevent outrageous acceleration values.
        # TODO: Is this a reasonable range?
        return ti.math.clamp(force, -1.0, 1.0)

    @ti.kernel
    def update_averages_kernel(self, balls: ti.template()):
        self.avg_pos.fill(0.0)
        self.avg_vel.fill(0.0)
        for w, b in ti.ndrange(c.NUM_WORLDS, c.NUM_BALLS):
            obj = balls[w, b]
            self.avg_pos[w] += obj.pos / c.NUM_BALLS
            self.avg_vel[w] += obj.vel / c.NUM_BALLS

    @ti.kernel
    def view_and_react_kernel(self, balls: ti.template(), debug: bool):
        for w, b1 in ti.ndrange(c.NUM_WORLDS, c.NUM_BALLS):
            ball1 = balls[w, b1]
            view = ball.View(0.0)
            # Record proprioception data (per ball view inputs).
            view[ball.POS_X] = ball1.pos.x
            view[ball.POS_Y] = ball1.pos.y
            view[ball.VEL_X] = ball1.vel.x
            view[ball.VEL_Y] = ball1.vel.y

            # Record the force acting on the ball based on the incline of the
            # topography at its current position.
            acc = self.acc_from_topography(w, ball1.pos)
            view[ball.SLP_X] = acc.x
            view[ball.SLP_Y] = acc.y

            # Zero out all the nearest view values, so we don't get historical
            # values hanging around because an ball noved out of view.
            view[ball.NRST_POS_X] = 0.0
            view[ball.NRST_POS_Y] = 0.0
            view[ball.NRST_VEL_X] = 0.0
            view[ball.NRST_VEL_X] = 0.0

            # Record the relative position and velocity of the nearest ball.
            min_distance = ti.math.inf
            # TODO: Optimize?
            for b2 in range(c.NUM_BALLS):
                if b1 != b2:
                    ball2 = balls[w, b2]
                    distance = ti.math.distance(ball1.pos, ball2.pos)
                    if distance < c.VIEW_RADIUS and distance < min_distance:
                        rel_pos = ball2.pos - ball1.pos
                        rel_vel = ball2.vel - ball1.vel
                        min_distance = distance
                        view[ball.NRST_POS_X] = rel_pos.x
                        view[ball.NRST_POS_Y] = rel_pos.y
                        view[ball.NRST_VEL_X] = rel_vel.x
                        view[ball.NRST_VEL_X] = rel_vel.y

            # Record relative positions to the centroid of all balls.
            rel_pos = self.avg_pos[w] - ball1.pos
            rel_vel = self.avg_vel[w] - ball1.vel
            view[ball.AVG_POS_X] = rel_pos.x
            view[ball.AVG_POS_Y] = rel_pos.y

            # Let this ball react to the view we just computed.
            reaction = self.controllers.activate(view, w, b1)
            ball1.acc.x = acc.x + 1.2 * reaction[ball.ACC_X]
            ball1.acc.y = acc.y + 1.2 * reaction[ball.ACC_Y]
            ball1.rad = reaction[ball.RAD]

            balls[w, b1] = ball1
            if ti.static(self.debug):
                self.views[w, b1] = view
                self.reactions[w, b1] = reaction

    def view_and_react(self, debug=False):
        # Update avarages first, so they're ready for view_and_react_kernel.
        self.update_averages_kernel(self.balls)
        self.view_and_react_kernel(self.balls, debug)
        self.controllers.finalize_activation()

    @ti.func
    def wall_collide(self, balls: ti.template(), w, b):
        lower_bound = balls[w, b].radius()
        upper_bound = 1.0 - balls[w, b].radius()
        if balls[w, b].pos.x < lower_bound:
            balls[w, b].pos.x = lower_bound
            balls[w, b].vel.x *= -1
            balls[w, b].hits += 1
        elif balls[w, b].pos.x > upper_bound:
            balls[w, b].pos.x = upper_bound
            balls[w, b].vel.x *= -1
            balls[w, b].hits += 1
        if balls[w, b].pos.y < lower_bound:
            balls[w, b].pos.y = lower_bound
            balls[w, b].vel.y *= -1
            balls[w, b].hits += 1
        elif balls[w, b].pos.y > upper_bound:
            balls[w, b].pos.y = upper_bound
            balls[w, b].vel.y *= -1
            balls[w, b].hits += 1

    @ti.func
    def ball_collide(self, balls: ti.template(), w, b1):
        # Look at all other balls to see if they collide with this one.
        # NOTE: This is a very simple collision detection and resolution
        # algorithm. It's not very realistic, merely good enough for this
        # experiment.
        # TODO: Optimize?
        for b2 in range(c.NUM_BALLS):
            # Only consider each pair of balls once.
            if b1 <= b2:
                continue
            pos1, pos2 = balls[w, b1].pos, balls[w, b2].pos
            vec = pos1 - pos2
            distance = vec.norm()
            rad1, rad2 = balls[w, b1].radius(), balls[w, b2].radius()
            overlap = rad1 + rad2 - distance
            # If these balls are close enough that they overlap each other...
            if overlap > 0.0:
                # Displace the two balls so they no longer overlap. We push both
                # balls away from each other an equal amount along an
                # imaginary line drawn between them.
                balls[w, b1].pos += overlap * 0.5 * vec.normalized()
                balls[w, b2].pos -= overlap * 0.5 * vec.normalized()
                # Swap velocities of the two balls (assume fully elastic
                # collisions with equal mass)
                balls[w, b1].vel, balls[w, b2].vel = (
                    balls[w, b2].vel, balls[w, b1].vel)
                balls[w, b1].hits += 1
                balls[w, b2].hits += 1
                # Assume only one collision per ball.
                break

    @ti.kernel
    def update_balls_kernel(self, balls: ti.template(),
                            next_balls: ti.template()):
        for w, b in ti.ndrange(c.NUM_WORLDS, c.NUM_BALLS):
            # TODO: Maybe remove angular displacement since it didn't work?
            # Since we haven't updated this ball in next_balls, it's still
            # the ball from the step before. Exploit this to get the previous
            # velocity, which is used to calculate displacement.
            prev_vel = next_balls[w, b].vel
            # Get the desired next state for all balls
            next_balls[w, b] = balls[w, b].next_state(prev_vel)
            # Then resolve colisions to make sure all states are valid.
            self.wall_collide(next_balls, w, b)
            self.ball_collide(next_balls, w, b)

    def update_balls(self):
        self.update_balls_kernel(self.balls, self.next_balls)
        self.balls, self.next_balls = self.next_balls, self.balls

    @ti.kernel
    def get_metrics_kernel(self, ang_disp: ti.types.ndarray(),
                           lin_disp: ti.types.ndarray(),
                           hits: ti.types.ndarray()):
        # Aggregate metrics on the GPU to minimize data transfer between
        # simulations.
        for w, b in ti.ndrange(c.NUM_WORLDS, c.NUM_BALLS):
            ang_disp[w] += self.balls[w, b].ang_disp
            lin_disp[w] += self.balls[w, b].lin_disp
            hits[w] += self.balls[w, b].hits

    def get_metrics(self):
        ang_disp = np.zeros(c.NUM_WORLDS, dtype=np.float32)
        lin_disp = np.zeros(c.NUM_WORLDS, dtype=np.float32)
        hits = np.zeros(c.NUM_WORLDS, dtype=np.float32)
        self.get_metrics_kernel(ang_disp, lin_disp, hits)
        return pd.DataFrame({
            'world': w,
            'ang_disp': ang_disp[w],
            'lin_disp': lin_disp[w],
            'hits': hits[w]
        } for w in range(c.NUM_WORLDS))

    def step(self):
        self.view_and_react()
        self.update_balls()

    def run(self):
        for _ in range(c.NUM_STEPS):
            self.step()
        return self.get_metrics()
