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
        #self.agents = agent.Agent.field(shape=(num_worlds, c.NUM_OBJECTS))
        self.controllers = None
        # These are only fields so we can visualize them for debugging,
        # otherwise they could just be computed on the fly and thrown away.
        self.avg_pos = ti.math.vec2.field(shape=(num_worlds))
        self.avg_vel = ti.math.vec2.field(shape=(num_worlds))
        self.views = agent.View.field(shape=(num_worlds, c.NUM_OBJECTS))
        self.reactions = agent.Reaction.field(
            shape=(num_worlds, c.NUM_OBJECTS))

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
        self.objects.ang_disp.fill(0.0)
        self.objects.lin_disp.fill(0.0)
        # To calculate displacement, we peek at the velocity values from the
        # objects in the previous step, so initialize to 0.0.
        self.__objects.vel.fill(0.0)

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
        # This is an approximate way of calculating acceleration due to an
        # incline which doesn't handle sharp discontinuities well. So, put a
        # clamp on it to prevent outrageous acceleration values.
        # TODO: Is this a reasonable range?
        return ti.math.clamp(force, -1.0, 1.0)

    @ti.kernel
    def __update_averages(self, objects: ti.template()):
        self.avg_pos.fill(0.0)
        self.avg_vel.fill(0.0)
        for w, o in ti.ndrange(self.num_worlds, c.NUM_OBJECTS):
            obj = objects[w, o]
            self.avg_pos[w] += obj.pos / c.NUM_OBJECTS
            self.avg_vel[w] += obj.vel / c.NUM_OBJECTS

    @ti.kernel
    def __update_views(self, objects: ti.template(), debug: bool):
        for w, o1 in ti.ndrange(self.num_worlds, c.NUM_OBJECTS):
            object1 = objects[w, o1]
            view = agent.View(0.0)
            # Record proprioception data (per object view inputs).
            view[agent.POS_X] = object1.pos.x
            view[agent.POS_Y] = object1.pos.y
            view[agent.VEL_X] = object1.vel.x
            view[agent.VEL_Y] = object1.vel.y

            # Record the force acting on the object based on the incline of the
            # topography at its current position.
            acc = self.acc_from_topography(w, object1.pos)
            view[agent.SLP_X] = acc.x
            view[agent.SLP_Y] = acc.y

            # Zero out all the nearest view values, so we don't get historical
            # values hanging around because an object noved out of view.
            view[agent.NRST_POS_X] = 0.0
            view[agent.NRST_POS_Y] = 0.0
            view[agent.NRST_VEL_X] = 0.0
            view[agent.NRST_VEL_X] = 0.0

            # Record the relative position and velocity of the nearest object.
            min_distance = ti.math.inf
            # TODO: Optimize?
            for o2 in range(c.NUM_OBJECTS):
                if o1 != o2:
                    object2 = objects[w, o2]
                    distance = ti.math.distance(object1.pos, object2.pos)
                    if distance < c.VIEW_RADIUS and distance < min_distance:
                        rel_pos = object2.pos - object1.pos
                        rel_vel = object2.vel - object1.vel
                        min_distance = distance
                        view[agent.NRST_POS_X] = rel_pos.x
                        view[agent.NRST_POS_Y] = rel_pos.y
                        view[agent.NRST_VEL_X] = rel_vel.x
                        view[agent.NRST_VEL_X] = rel_vel.y

            # Record relative positions to the centroid of all objects.
            rel_pos = self.avg_pos[w] - object1.pos
            rel_vel = self.avg_vel[w] - object1.vel
            view[agent.AVG_POS_X] = rel_pos.x
            view[agent.AVG_POS_Y] = rel_pos.y

            # Let this agent react to the view we just computed.
            reaction = self.controllers.activate(view, w, o1)
            object1.acc.x = acc.x + 1.2 * reaction[agent.ACC_X]
            object1.acc.y = acc.y + 1.2 * reaction[agent.ACC_Y]
            object1.rad = reaction[agent.RAD]

            objects[w, o1] = object1
            self.views[w, o1] = view
            self.reactions[w, o1] = reaction

    def view_and_react(self, debug=False):
        self.__update_averages(self.objects)
        self.__update_views(self.objects, debug)

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
        # NOTE: This is a very simple collision detection and resolution
        # algorithm. It's not very realistic, merely good enough for this
        # experiment.
        # TODO: Optimize?
        for o2 in range(c.NUM_OBJECTS):
            # Only consider each pair of objects once.
            if o1 <= o2:
                continue
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
            # Since we haven't updated this object in next_objects, it's still
            # the object from the step before. Exploit this to get the previous
            # velocity, which is used to calculate displacement.
            prev_vel = next_objects[w, o].vel
            next_objects[w, o] = prev_objects[w, o].next_state(prev_vel)
            self.wall_collide(next_objects, w, o)
            self.object_collide(next_objects, w, o)

    def update_objects(self):
        self.__update_objects(self.objects, self.__objects)
        self.objects, self.__objects = self.__objects, self.objects

    @ti.kernel
    def get_metrics_kernel(self, ang_disp: ti.types.ndarray(),
                           lin_disp: ti.types.ndarray(),
                           hits: ti.types.ndarray()):
        for w, o in ti.ndrange(self.num_worlds, c.NUM_OBJECTS):
            ang_disp[w] += self.objects[w, o].ang_disp
            lin_disp[w] += self.objects[w, o].lin_disp
            hits[w] += self.objects[w, o].hits

    def get_metrics(self):
        ang_disp = np.zeros(self.num_worlds, dtype=np.float32)
        lin_disp = np.zeros(self.num_worlds, dtype=np.float32)
        hits = np.zeros(self.num_worlds, dtype=np.float32)
        self.get_metrics_kernel(ang_disp, lin_disp, hits)
        return {
            'ang_disp': np.nan_to_num(ang_disp),
            'lin_disp': np.nan_to_num(lin_disp),
            'hits': np.nan_to_num(hits)
        }

        # ang_disp = ti.field(float, self.num_worlds)
        # lin_disp = ti.field(float, self.num_worlds)
        # hits = ti.field(float, self.num_worlds)
        # self.get_metrics_kernel(ang_disp, lin_disp, hits)
        # # return {
        # #     'ang_disp': np.array([0.0] * c.NUM_WORLDS),
        # #     'lin_disp': np.array([0.0] * c.NUM_WORLDS),
        # #     'inv_hits': np.array([1.0] * c.NUM_WORLDS)
        # # }
        # return {
        #     'ang_disp': np.nan_to_num(ang_disp.to_numpy()),
        #     'lin_disp': np.nan_to_num(lin_disp.to_numpy()),
        #     'inv_hits': 1 / (1 + np.nan_to_num(hits.to_numpy()))
        # }

    def step(self):
        self.view_and_react()
        self.update_objects()

    def run(self):
        for _ in range(c.NUM_STEPS):
            self.step()
        return self.get_metrics()
