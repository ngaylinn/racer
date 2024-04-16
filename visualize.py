import shutil
import time
import warnings

import taichi as ti
from tqdm import trange

import agent
import constants as c

FRAME_RATE = 24

def __render_topography(gui, simulator):
    # TODO: Optimize?
    gui.set_image(simulator.topographies.to_numpy()[0])


def __draw_x(gui, pos, color):
    offset = 7 / c.WORLD_SIZE
    gui.line(pos + ti.math.vec2( offset, offset),
             pos + ti.math.vec2(-offset, -offset), color=color)
    gui.line(pos + ti.math.vec2( offset, -offset),
             pos + ti.math.vec2(-offset, offset), color=color)


def __render_debug(gui, simulator):
    # Draw a field of view for every object.
    for o in range(c.NUM_OBJECTS):
        gui.circle(simulator.objects[0, o].pos, color=0xffffaa,
                   radius=c.VIEW_RADIUS_PX)
    # Layer on top arrows indicating status of each object.
    for o in range(c.NUM_OBJECTS):
        direction = ti.math.vec2(simulator.views[0, o][agent.NRST_POS_X],
                                 simulator.views[0, o][agent.NRST_POS_Y])
        # Draw arrows from each object to the nearest object of each
        # kind that it can see.
        gui.arrow(simulator.objects[0, o].pos, direction, color=0x0000ff)

        # Also draw an arrow indicating this object's intended direction.
        gui.arrow(simulator.objects[0, o].pos,
                  simulator.agents[0, o].heading(),
                  color=0xff00ff)

    # Draw a centroid for each kind of object.
    __draw_x(gui, simulator.avg_pos[0], 0x0000ff)


def __render_objects(gui, simulator):
    # Draw a circle for every object.
    for o in range(c.NUM_OBJECTS):
        obj = simulator.objects[0, o]
        gui.circle(obj.pos, radius=obj.radius_px(), color=0x0000ff)


def __simulate_and_render_step(gui, simulator, debug=False):
    __render_topography(gui, simulator)
    simulator.view_and_react()
    if debug:
        __render_debug(gui, simulator)
    simulator.update_objects()
    __render_objects(gui, simulator)


def show(simulator, debug=False):
    gui = ti.GUI('MicRoboBiome', c.WORLD_SHAPE,
                 background_color=0xffffff, show_gui=True)
    step = 0
    while gui.running:
        if step == 0:
            simulator.randomize_objects()
        __simulate_and_render_step(gui, simulator, debug)
        step = (step + 1) % c.NUM_STEPS
        # TODO: Maybe restore?
        # if step == c.NUM_STEPS - 1:
        #     evaluator.score_one(simulator)
        #     evaluator.visualize(gui)
        gui.show()
        if step == c.NUM_STEPS - 1:
            time.sleep(2.0)


def save(simulator, filename, debug=False, progress=None):
    gui = ti.GUI('MicRoboBiome', c.WORLD_SHAPE,
                 background_color=0xffffff, show_gui=False)
    video_manager = ti.tools.VideoManager(
        output_dir='output', framerate=FRAME_RATE, automatic_build=False)
    simulator.randomize_objects()
    if progress is None:
        progress = trange(c.NUM_STEPS + 2*FRAME_RATE)
        progress.set_description('Making Video')
    for _ in range(c.NUM_STEPS):
        gui.clear()
        __simulate_and_render_step(gui, simulator, debug)
        # Ignore spurious warnings from Taichi's image writing code.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            video_manager.write_frame(gui.get_image())
        progress.update()
    for _ in range(FRAME_RATE * 2):
        # Ignore spurious warnings from Taichi's image writing code.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            video_manager.write_frame(gui.get_image())
        # TODO: Maybe restore?
        # evaluator.score_one(simulator)
        # evaluator.visualize(gui)
        progress.update()
    video_manager.make_video(gif=False, mp4=True)
    shutil.copyfile(video_manager.get_output_filename('.mp4'), filename)


def save_batch(simulator, filename, batch_size, debug=False):
    progress = trange(batch_size * (c.NUM_STEPS + 2*FRAME_RATE))
    progress.set_description('Making Videos')
    for i in range(batch_size):
        save(simulator, filename.format(i=i), debug, progress)
