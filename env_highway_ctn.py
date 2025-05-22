
"""Environment used to train vehicles to improve traffic on a highway."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import os
import atexit
import time
import traceback
import numpy as np
import random
import shutil
import subprocess
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer
from flow.utils.flow_warnings import deprecated_attribute

import gym
from gym.spaces import Box
from gym.spaces import Tuple
from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException
import traci

import sumolib

from flow.core.util import ensure_dir
from flow.core.kernel import Kernel
from flow.utils.exceptions import FatalFlowError
from flow.core.rewards import desired_velocity
import string
import copy

import math

# from gym.spaces.box import Box

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25
}


# class MultiAgentHighwayPOEnv(MultiEnv):
class MultiAgentHighwayPOEnv(gym.Env, metaclass=ABCMeta):

    def __init__(self, env_params, sim_params, network, args, simulator='traci', scenario=None):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.args = args
        self.vehicle_in_main = []
        self.env_params = env_params
        if scenario is not None:
            deprecated_attribute(self, "scenario", "network")
        self.network = scenario if scenario is not None else network
        self.net_params = self.network.net_params
        self.initial_config = self.network.initial_config
        self.sim_params = deepcopy(sim_params)
        # check whether we should be rendering
        self.should_render = self.sim_params.render
        self.sim_params.render = False
        time_stamp = ''.join(str(time.time()).split('.'))
        if os.environ.get("TEST_FLAG", 0):
            # 1.0 works with stress_test_start 10k times
            time.sleep(1.0 * int(time_stamp[-6:]) / 1e6)
        # FIXME: this is sumo-specific
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()
        # time_counter: number of steps taken since the start of a rollout
        self.time_counter = 0
        # step_counter: number of total steps taken
        self.step_counter = 0
        # initial_state:
        self.initial_state = {}
        self.state = None
        self.obs_var_labels = []

        # simulation step size
        self.sim_step = sim_params.sim_step

        # the simulator used by this environment
        self.simulator = simulator

        # create the Flow kernel
        self.k = Kernel(simulator=self.simulator,
                        sim_params=self.sim_params)

        # use the network class's network parameters to generate the necessary
        # network components within the network kernel
        self.k.network.generate_network(self.network)

        # initial the vehicles kernel using the VehicleParams object
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))

        # initialize the simulation using the simulation kernel. This will use
        # the network kernel as an input in order to determine what network
        # needs to be simulated.
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)

        # pass the kernel api to the kernel and it's subclasses
        self.k.pass_api(kernel_api)

        # the available_routes variable contains a dictionary of routes
        # vehicles can traverse; to be used when routes need to be chosen
        # dynamically
        self.available_routes = self.k.network.rts

        # store the initial vehicle ids
        self.initial_ids = deepcopy(self.network.vehicles.ids)

        # store the initial state of the vehicles' kernel (needed for restarting
        # the simulation)
        self.k.vehicle.kernel_api = None
        self.k.vehicle.master_kernel = None
        self.initial_vehicles = deepcopy(self.k.vehicle)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        self.setup_initial_state()

        # use pyglet to render the simulation
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            save_render = self.sim_params.save_render
            sight_radius = self.sim_params.sight_radius
            pxpm = self.sim_params.pxpm
            show_radius = self.sim_params.show_radius

            # get network polygons
            network = []
            # FIXME: add to network kernel instead of hack
            for lane_id in self.k.kernel_api.lane.getIDList():
                _lane_poly = self.k.kernel_api.lane.getShape(lane_id)
                lane_poly = [i for pt in _lane_poly for i in pt]
                network.append(lane_poly)

            # instantiate a pyglet renderer
            self.renderer = Renderer(
                network,
                self.sim_params.render,
                save_render,
                sight_radius=sight_radius,
                pxpm=pxpm,
                show_radius=show_radius)

            # render a frame
            self.render(reset=True)
        elif self.sim_params.render in [True, False]:
            # default to sumo-gui (if True) or sumo (if False)
            if (self.sim_params.render is True) and self.sim_params.save_render:
                self.path = os.path.expanduser('~') + '/flow_rendering/' + self.network.name
                os.makedirs(self.path, exist_ok=True)
        else:
            raise FatalFlowError(
                'Mode %s is not supported!' % self.sim_params.render)
        atexit.register(self.terminate)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(-float('inf'), float('inf'), shape=(5,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def step(self, rl_actions):

        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=True)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            # if crash:
            #     break

            # render a frame
            self.render()

        states = []

        # collect information of the state of the network based on the
        # environment class used
        self.state = states.copy()

        # collect observation new state associated with action
        next_observation = states.copy()

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.sims_per_step * (self.env_params.warmup_steps + self.env_params.horizon)
                )

        # compute the info for each agent
        infos = {}

        # compute the reward
        reward = self.compute_reward(rl_actions, fail=crash)

        return next_observation, reward, done, infos

    def apply_rl_actions(self, rl_actions=None):
        """See class definition."""

        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                if isinstance(actions, np.ndarray):   # for maddpg continuous action
                    accel = actions[0] * self.args.acc_step
                    lane_change = round(actions[1])
                else:
                    accel = (actions % 3 - 1) * self.args.acc_step
                    lane_change = actions // 3 - 1

                self.k.vehicle.apply_acceleration(rl_id, accel)
                self.k.vehicle.apply_lane_change(rl_id, lane_change)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(lead_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ])

            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        num_rl = len(rl_actions) + 1

        ids = self.vehicle_in_main.copy()
        speeds = {veh_id: self.k.vehicle.get_speed(veh_id) for veh_id in ids}
        target_dict = {'0': [3], '1': [1, 2], '2': [0]}

        norm_speed_list = [val/30 for _, val in speeds.items()]
        r_flow = sum(norm_speed_list)  # basic speed reward

        r_act = 0
        r_safe = 0
        r_pos = 0
        r_int = 0
        for rl_id, action in rl_actions.items():
            v_x = speeds[rl_id]
            if not isinstance(action, np.ndarray):
                if action in [2, 5, 8] or \
                        (v_x + self.args.acc_step > self.args.max_av_speed and action in [1, 4, 7]):
                    r_act += self.args.w_speed  # speeding or keep high speed

                if self.k.vehicle.get_headway(rl_id) < 10:
                    r_safe -= self.args.w_safe_gap  # safe headway

                dist_to_end = self.args.highway_length - self.k.vehicle.get_2d_position(rl_id)[0]
                if self.k.vehicle._TraCIVehicle__sumo_obs[rl_id][80] == 'highway_0':
                    _, v_y = self.action_2_ac_lc(action)
                    cur_lane = self.k.vehicle.get_lane(rl_id) - v_y
                    v_y = -v_y
                    target_lane = target_dict[self.k.vehicle._TraCIVehicle__sumo_obs[rl_id][84][1][4]]
                    dist_abs, dist_sign = self.y_dist_and_sign(cur_lane, target_lane, v_y)
                    field_val = np.exp(-dist_to_end ** 2 / (2 * self.args.w_sigma ** 2)) / (self.args.w_zeta * dist_abs + 1)
                    r_pos += (v_x * dist_to_end / self.args.w_sigma ** 2 +
                              self.args.w_zeta * v_y * dist_sign / (self.args.w_zeta * dist_abs + 1)) * field_val
                    if dist_to_end < 30 and cur_lane in target_lane and v_x > 5:
                        r_int += self.args.w_intention  # intention reward
            else:  # for maddpg
                if action[0] > 0 or \
                        (v_x + self.args.acc_step > self.args.max_av_speed and action[0] > -0.5):
                    r_act += self.args.w_speed
                if self.k.vehicle.get_headway(rl_id) < 10:
                    r_safe -= self.args.w_safe_gap  # safe headway
                dist_to_end = self.args.highway_length - self.k.vehicle.get_2d_position(rl_id)[0]
                if self.k.vehicle._TraCIVehicle__sumo_obs[rl_id][80] == 'highway_0':
                    v_y = round(action[1])
                    cur_lane = self.k.vehicle.get_lane(rl_id) - v_y
                    v_y = -v_y
                    target_lane = target_dict[self.k.vehicle._TraCIVehicle__sumo_obs[rl_id][84][1][4]]
                    dist_abs, dist_sign = self.y_dist_and_sign(cur_lane, target_lane, v_y)
                    field_val = np.exp(-dist_to_end ** 2 / (2 * self.args.w_sigma ** 2)) / (
                                self.args.w_zeta * dist_abs + 1)
                    r_pos += (v_x * dist_to_end / self.args.w_sigma ** 2 + self.args.w_zeta * v_y * dist_sign / (
                                self.args.w_zeta * dist_abs + 1)) * field_val
                    if dist_to_end < 30 and cur_lane in target_lane and v_x > 5:
                        r_int += self.args.w_intention  # intention reward

        r_act = r_act / num_rl
        r_pos = r_pos * 1e3 / num_rl

        reward = r_flow + r_act + r_safe + r_pos + r_int

        return reward

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)

    def restart_simulation(self, sim_params, render=None):

        self.k.close()

        # killed the sumo process if using sumo/TraCI
        if self.simulator == 'traci':
            self.k.simulation.sumo_proc.kill()

        if render is not None:
            self.sim_params.render = render

        if sim_params.emission_path is not None:
            ensure_dir(sim_params.emission_path)
            self.sim_params.emission_path = sim_params.emission_path

        self.k.network.generate_network(self.network)
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)
        self.k.pass_api(kernel_api)

        self.setup_initial_state()

    def setup_initial_state(self):

        # determine whether to shuffle the vehicles
        if self.initial_config.shuffle:
            random.shuffle(self.initial_ids)

        # generate starting position for vehicles in the network
        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))

        # save the initial state. This is used in the _reset function
        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.k.vehicle.get_type(veh_id)
            pos = start_pos[i][1]
            lane = start_lanes[i]
            speed = self.k.vehicle.get_initial_speed(veh_id)
            edge = start_pos[i][0]

            self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)

    def reset(self):

        # reset the time counter
        self.time_counter = 0

        # Now that we've passed the possibly fake init steps some rl libraries
        # do, we can feel free to actually render things
        if self.should_render:
            self.sim_params.render = True
            # got to restart the simulation to make it actually display anything
            self.restart_simulation(self.sim_params)

        if self.sim_params.restart_instance or \
                (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            # issue a random seed to induce randomness into the next rollout
            self.sim_params.seed = random.randint(0, 1e5)

            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            # restart the sumo instance
            self.restart_simulation(self.sim_params)

        # perform shuffling (if requested)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        # clear all vehicles from the network and the vehicles class
        if self.simulator == 'traci':
            for veh_id in self.k.kernel_api.vehicle.getIDList():  # FIXME: hack
                try:
                    self.k.vehicle.remove(veh_id)
                except (FatalTraCIError, TraCIException):
                    print(traceback.format_exc())

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.k.vehicle.get_ids()):
            # do not try to remove the vehicles from the network in the first
            # step after initializing the network, as there will be no vehicles
            if self.step_counter == 0:
                continue
            try:
                self.k.vehicle.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        # do any additional resetting of the vehicle class needed
        self.k.vehicle.reset()

        # reintroduce the initial vehicles to the network
        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed = \
                self.initial_state[veh_id]

            try:
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    edge=edge,
                    lane=lane_index,
                    pos=pos,
                    speed=speed)
            except (FatalTraCIError, TraCIException):
                # if a vehicle was not removed in the first attempt, remove it
                # now and then reintroduce it
                self.k.vehicle.remove(veh_id)
                if self.simulator == 'traci':
                    self.k.kernel_api.vehicle.remove(veh_id)  # FIXME: hack
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    edge=edge,
                    lane=lane_index,
                    pos=pos,
                    speed=speed)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        # update the information in each kernel to match the current state
        self.k.update(reset=True)

        # update the colors of vehicles
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()

        if self.simulator == 'traci':
            initial_ids = self.k.kernel_api.vehicle.getIDList()
        else:
            initial_ids = self.initial_ids

        # check to make sure all vehicles have been spawned
        if len(self.initial_ids) > len(initial_ids):
            missing_vehicles = list(set(self.initial_ids) - set(initial_ids))
            msg = '\nNot enough vehicles have spawned! Bad start?\n' \
                  'Missing vehicles / initial state:\n'
            for veh_id in missing_vehicles:
                msg += '- {}: {}\n'.format(veh_id, self.initial_state[veh_id])
            raise FatalFlowError(msg=msg)

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # observation associated with the reset (no warm-up steps)
        observation = np.copy(states)

        # perform (optional) warm-up steps before training
        for _ in range(self.env_params.warmup_steps):
            observation, _, _, _ = self.step(rl_actions=None)

        # render a frame
        self.render(reset=True)

        return observation

    def terminate(self):
        """Close the TraCI I/O connection.

        Should be done at end of every experiment. Must be in Env because the
        environment opens the TraCI connection.
        """
        try:
            # close everything within the kernel
            self.k.close()
            # close pyglet renderer
            if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
                self.renderer.close()
            # generate video
            elif (self.sim_params.render is True) and self.sim_params.save_render:
                images_dir = self.path.split('/')[-1]
                speedup = 10  # multiplier: renders video so that `speedup` seconds is rendered in 1 real second
                fps = speedup//self.sim_step
                p = subprocess.Popen(["ffmpeg", "-y", "-r", str(fps), "-i", self.path+"/frame_%06d.png",
                                      "-pix_fmt", "yuv420p", "%s/../%s.mp4" % (self.path, images_dir)])
                p.wait()
                shutil.rmtree(self.path)
        except FileNotFoundError:
            # Skip automatic termination. Connection is probably already closed
            print(traceback.format_exc())

    def render(self, reset=False, buffer_length=5):
        """Render a frame.

        Parameters
        ----------
        reset : bool
            set to True to reset the buffer
        buffer_length : int
            length of the buffer
        """
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    self.frame_buffer.append(self.frame.copy())
                    self.sights_buffer.append(self.sights.copy())
                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)
        elif (self.sim_params.render is True) and self.sim_params.save_render:
            # sumo-gui render
            self.k.kernel_api.gui.screenshot("View #0", self.path+"/frame_%06d.png" % self.time_counter)

    def pyglet_render(self):
        """Render a frame using pyglet."""
        # get human and RL simulation status
        human_idlist = self.k.vehicle.get_human_ids()
        machine_idlist = self.k.vehicle.get_rl_ids()
        human_logs = []
        human_orientations = []
        human_dynamics = []
        machine_logs = []
        machine_orientations = []
        machine_dynamics = []
        max_speed = self.k.network.max_speed()
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if 'track' in id:
                machine_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                machine_orientations.append(
                    self.k.vehicle.get_orientation(id))
                machine_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
            else:
                human_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                human_orientations.append(
                    self.k.vehicle.get_orientation(id))
                human_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
        for id in machine_idlist:
            machine_logs.append(
                [self.k.vehicle.get_timestep(id),
                 self.k.vehicle.get_timedelta(id),
                 id])
            machine_orientations.append(
                self.k.vehicle.get_orientation(id))
            machine_dynamics.append(
                self.k.vehicle.get_speed(id)/max_speed)

        # step the renderer
        self.frame = self.renderer.render(human_orientations,
                                          machine_orientations,
                                          human_dynamics,
                                          machine_dynamics,
                                          human_logs,
                                          machine_logs)

        # get local observation of RL vehicles
        self.sights = []
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if "track" in id:
                orientation = self.k.vehicle.get_orientation(id)
                sight = self.renderer.get_sight(
                    orientation, id)
                self.sights.append(sight)
        for id in machine_idlist:
            orientation = self.k.vehicle.get_orientation(id)
            sight = self.renderer.get_sight(
                orientation, id)
            self.sights.append(sight)

    def bypass(self):

        while True:
            self.k.simulation.simulation_step()
            self.k.update(reset=True)
            if len(self.k.vehicle.get_rl_ids()) > 1:
                break

        return None

    @staticmethod
    def action_2_ac_lc(act):
        ac = act % 3 - 1
        lc = act // 3 - 1
        return ac, lc

    @staticmethod
    def y_dist_and_sign(y, y_t, v_y):
        dist_sign = - v_y if y in y_t else np.sign(y - y_t[0])
        if len(y_t) == 1:
            dist_abs = abs(y - y_t[0])
        else:
            dist_abs = 0 if y in y_t else 1
        return dist_abs, dist_sign

    def plot_reward_pos(self, rl_actions):
        for rl_id, action in rl_actions.items():
            v_x = 30

            dist_to_end = self.args.highway_length - self.k.vehicle.get_2d_position(rl_id)[0]

            _, v_y = self.action_2_ac_lc(action)
            cur_lane = self.k.vehicle.get_lane(rl_id) - v_y
            v_y = -v_y
            target_lane = [3]
            dist_abs, dist_sign = self.y_dist_and_sign(cur_lane, target_lane, v_y)
            field_val = np.exp(-dist_to_end ** 2 / (2 * self.args.w_sigma ** 2)) / (self.args.w_zeta * dist_abs + 1)
            r_pos = (v_x * dist_to_end / self.args.w_sigma ** 2 + self.args.w_zeta * v_y * dist_sign / (self.args.w_zeta * dist_abs + 1)) * field_val

