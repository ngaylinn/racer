import numpy as np
import taichi as ti

import constants as c
import agent
from object import Object

@ti.data_oriented
class Simulator:
    def __init__(self, num_worlds=1):
        self.num_worlds = num_worlds
        self.topographies = ti.field(float, shape=(num_worlds,) + c.WORLD_SHAPE)
        self.objects = Object.field(shape=(num_worlds, c.NUM_OBJECTS))
        self.__objects = Object.field(shape=(num_worlds, c.NUM_OBJECTS))
        self.agents = agent.Agent.field(shape=(num_worlds, c.NUM_OBJECTS))
        # These are only fields so we can visualize them for debugging,
        # otherwise they could just be computed on the fly and thrown away.
        self.avg_pos = ti.math.vec2.field(shape=(num_worlds))
        self.avg_vel = ti.math.vec2.field(shape=(num_worlds))
        self.views = agent.View.field(shape=(num_worlds, c.NUM_OBJECTS))

    def randomize_objects(self):
        shape = (self.num_worlds, c.NUM_OBJECTS)
        count = np.prod(shape)
        self.objects.pos.from_numpy(
            np.random.random(count * 2)
            .astype(np.float32)
            .reshape(shape + (2,)))
        self.objects.vel.fill(0.0)
        self.objects.acc.fill(0.0)
        self.objects.rad.fill(0.0)
        self.objects.hits.fill(0)
        self.objects.dist.fill(0.0)

    @ti.func
    def project_vec(from_vec, onto_vec):
        return (from_vec.dot(onto_vec) / onto_vec.dot(onto_vec)) * onto_vec

    @ti.func
    def topo_index(self, pos: ti.math.vec2, offset: int):
        max_x, max_y = c.WORLD_SHAPE - ti.Vector([1, 1])
        return ti.math.ivec2(
            ti.math.clamp(int(pos.x * max_x), offset, max_x - offset),
            ti.math.clamp(int(pos.y * max_y), offset, max_y - offset))

    @ti.func
    def acc_from_topography(self, w, pos):
        x, y = self.topo_index(pos, 1)
        force = ti.math.vec2(0.0, 0.0)
        # To estimate the normal at cell (x, y), look at the neighboring cells and
        # average their contributions to this vector.
        for x_from in range(x - 1, x + 2):
            for y_from in range(y - 1, y + 2):
                z_delta = (self.topographies[w, x_from, y_from] -
                           self.topographies[w, x, y])
                x_delta = (x_from - x) / c.WORLD_SIZE
                y_delta = (y_from - y) / c.WORLD_SIZE
                # Note at the sample position, this vector is 0, 0, 0, so it does
                # not contribute to the normal and does not require special casing.
                force += 10000 * ti.math.vec2(x_delta, y_delta) * z_delta
        return force

    @ti.kernel
    def __update_averages(self, objects: ti.template()):
        self.avg_pos.fill(0.0)
        self.avg_vel.fill(0.0)
        for w, o in ti.ndrange(self.num_worlds, c.NUM_OBJECTS):
            obj = objects[w, o]
            self.avg_pos[w] += obj.pos / c.NUM_OBJECTS
            self.avg_vel[w] += obj.vel / c.NUM_OBJECTS

    @ti.kernel
    def __update_views(self, objects: ti.template()):
        for w, o1 in ti.ndrange(self.num_worlds, c.NUM_OBJECTS):
            # Record proprioception data (per object view inputs).
            self.views[w, o1][agent.POS_X] = objects[w, o1].pos.x
            self.views[w, o1][agent.POS_Y] = objects[w, o1].pos.y
            self.views[w, o1][agent.VEL_X] = objects[w, o1].vel.x
            self.views[w, o1][agent.VEL_Y] = objects[w, o1].vel.y

            # Record the force acting on the object based on the incline of the
            # topography at its current position.
            acc = self.acc_from_topography(w, objects[w, o1].pos)
            self.views[w, o1][agent.SLP_X] = acc.x
            self.views[w, o1][agent.SLP_Y] = acc.y

            # Zero out all the nearest view values, so we don't get historical
            # values hanging around because an object noved out of view.
            self.views[w, o1][agent.NRST_POS_X] = 0.0
            self.views[w, o1][agent.NRST_POS_Y] = 0.0
            self.views[w, o1][agent.NRST_VEL_X] = 0.0
            self.views[w, o1][agent.NRST_VEL_X] = 0.0

            # Record the relative position and velocity of the nearest object.
            min_distance = ti.math.inf
            # TODO: Optimize?
            for o2 in range(c.NUM_OBJECTS):
                if o1 != o2:
                    distance = ti.math.distance(
                        objects[w, o1].pos, objects[w, o2].pos)
                    if distance < c.VIEW_RADIUS and distance < min_distance:
                        rel_pos = objects[w, o2].pos - objects[w, o1].pos
                        rel_vel = objects[w, o2].vel - objects[w, o1].vel
                        min_distance = distance
                        self.views[w, o1][agent.NRST_POS_X] = rel_pos.x
                        self.views[w, o1][agent.NRST_POS_Y] = rel_pos.y
                        self.views[w, o1][agent.NRST_VEL_X] = rel_vel.x
                        self.views[w, o1][agent.NRST_VEL_X] = rel_vel.y

            # Record relative positions to the centroid of all objects.
            rel_pos = self.avg_pos[w] - objects[w, o1].pos
            rel_vel = self.avg_vel[w] - objects[w, o1].vel
            self.views[w, o1][agent.AVG_POS_X] = rel_pos.x
            self.views[w, o1][agent.AVG_POS_Y] = rel_pos.y

            # Let this agent react to the view we just computed.
            reaction = self.agents[w, o1].react(self.views[w, o1])
            objects[w, o1].acc.x = acc.x + 1.2 * reaction[agent.ACC_X]
            objects[w, o1].acc.y = acc.y + 1.2 * reaction[agent.ACC_Y]
            objects[w, o1].rad = reaction[agent.RAD]

    def view_and_react(self):
        self.__update_averages(self.objects)
        self.__update_views(self.objects)

    @ti.func
    def wall_collide(self, objects: ti.template(), w, o):
        lower_bound = objects[w, o].radius()
        upper_bound = 1.0 - objects[w, o].radius()
        if objects[w, o].pos.x < lower_bound:
            objects[w, o].pos.x = lower_bound
            objects[w, o].vel.x *= -1
            objects[w, o].hits += 1
        elif objects[w, o].pos.x > upper_bound:
            objects[w, o].pos.x = upper_bound
            objects[w, o].vel.x *= -1
            objects[w, o].hits += 1
        if objects[w, o].pos.y < lower_bound:
            objects[w, o].pos.y = lower_bound
            objects[w, o].vel.y *= -1
            objects[w, o].hits += 1
        elif objects[w, o].pos.y > upper_bound:
            objects[w, o].pos.y = upper_bound
            objects[w, o].vel.y *= -1
            objects[w, o].hits += 1

    @ti.func
    def object_collide(self, objects: ti.template(), w, o1):
        # Look at all other objects to see if they collide with this one.
        # TODO: Optimize?
        for o2 in range(c.NUM_OBJECTS):
            if o1 == o2:
                break
            pos1, pos2 = objects[w, o1].pos, objects[w, o2].pos
            vec = pos1 - pos2
            distance = vec.norm()
            rad1, rad2 = objects[w, o1].radius(), objects[w, o2].radius()
            overlap = rad1 + rad2 - distance
            if overlap > 0.0:
                # Displace the two objects so they no longer overlap. We push both
                # objects away from each other an equal amount along an
                # imaginary line drawn between the two.
                objects[w, o1].pos += overlap * 0.5 * vec.normalized()
                objects[w, o2].pos -= overlap * 0.5 * vec.normalized()
                # Swap velocities of the two objects (fully elastic collision with
                # equal mass)
                objects[w, o1].vel, objects[w, o2].vel = (
                    objects[w, o2].vel, objects[w, o1].vel)
                objects[w, o1].hits += 1
                objects[w, o2].hits += 1
                # Assume only one collision per object.
                break

    @ti.kernel
    def __update_objects(self, prev_objects: ti.template(),
                         next_objects: ti.template()):
        for w, o in ti.ndrange(self.num_worlds, c.NUM_OBJECTS):
            next_objects[w, o] = prev_objects[w, o].next_state()
            self.wall_collide(next_objects, w, o)
            self.object_collide(next_objects, w, o)

    def update_objects(self):
        self.__update_objects(self.objects, self.__objects)
        self.objects, self.__objects = self.__objects, self.objects

    def step(self):
        self.view_and_react()
        self.update_objects()
