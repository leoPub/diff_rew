"""Contains an experiment class for running simulations."""
import copy
import os
import random
import shutil

from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import numpy as np
import traci
import math
import pickle

from collections import defaultdict

# with TE


class Experiment:

    def __init__(self, flow_params, args):

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params, args)

        # Create the environment.
        self.env = create_env()
        self.vehicle_log = {}  # information of all vehicles, veh_id: [target, step_pos, step_lane, step_collision, is_arrived]
        self.in_out_num = {'in': [], 'out': [], 'delete': []}
        self.past_vehs = []
        self.args = args
        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, batch_buffer, batch_fu, mac, learner, logger, rl_actions=None, test=False):

        num_steps = self.env.env_params.horizon

        buffer = batch_buffer
        log_record = []
        action_record = [0] * 9

        for i in range(num_runs):

            self.env.reset()
            self.env.bypass()
            last_ids = []
            batch = batch_fu()
            episode = 0
            episode_return = 0
            episode_del = 0
            episode_velo = 0
            num_rl = 0

            for j in range(num_steps):

                t = j % (self.args.episode_limit + 1)

                if t == 0:
                    mac.init_hidden(batch_size=1)
                    batch = batch_fu()

                # vehicles in main road
                all_ids = self.env.k.vehicle.get_ids()
                veh_ids = self.in_main_road(all_ids, num_rl, del_veh=True)
                self.env.vehicle_in_main = veh_ids.copy()

                # log new inserted vehicle
                self.update_veh_log(veh_ids, last_ids)

                # get vehicles state
                veh_ids, veh_types, veh_lanes, x_pos, velo, veh_targ, state_dict, veh_map \
                    = self.get_vehicles_info(veh_ids)
                veh_map, left_hv, hv, right_hv = self.map_vehs(x_pos, veh_lanes, veh_ids)

                episode_velo += np.mean(velo)

                # get rl observation
                rl_ids = [id_veh for id_veh in veh_ids if not id_veh[5] == '0']
                num_rl = len(rl_ids)
                rl_obs = {veh: self.get_obs(veh, state_dict, left_hv, hv, right_hv) for veh in rl_ids}

                pre_transition_data = {
                    "state": [state_dict[veh] for veh in rl_ids],
                    "agents_num": [tuple([len(rl_ids)])],
                    "avail_actions": [self.get_available_action(rl_ids, state_dict, left_hv, hv, right_hv)],
                    "obs": rl_obs,
                }

                batch.update(self.fill_transition(pre_transition_data), ts=t)

                actions = mac.select_actions(batch, current_step=t, env_timestep=j, is_testing=False)
                actions_cpu = actions[0].cpu().numpy()
                action_to_exc = {rl_ids[i]: int(actions_cpu[i]) for i in range(len(rl_ids))}
                for i in range(len(rl_ids)):
                    action_record[actions_cpu[i]] = action_record[actions_cpu[i]] + 1

                _, reward, done, _ = self.env_step(action_to_exc)

                post_ids = self.in_main_road(self.env.k.vehicle.get_ids(), num_rl)
                new_ids = list(set(post_ids) - set(veh_ids))
                in_flow = {0: [], 1: [], 2: [], 3: []}
                for in_id in new_ids:
                    in_flow[self.env.k.vehicle.get_lane(in_id)] = [self.id_2_type(in_id) + 1, self.id_2_target(in_id) + 1]

                post_transition_data = {
                    "actions": actions,
                    "reward": [tuple([reward])],
                    "inflow": in_flow,
                    "terminated": [tuple([done])],
                }

                filled_post_transition = self.fill_transition(post_transition_data)

                batch.update(filled_post_transition, ts=t)
                episode_return += reward

                if t == self.args.episode_limit:
                    buffer.insert_episode_batch(batch)
                    episode += 1
                    if episode % 2000 == 0:
                        self.env.reset()
                        self.env.bypass()
                        log_record.append(copy.deepcopy(self.vehicle_log))
                        self.vehicle_log = {}

                        mac_params = mac.agent.state_dict()
                        for key, tensor in mac_params.items():
                            if tensor.device.type == 'cuda' and tensor.device.index == 1:
                                mac_params[key] = tensor.to('cuda:0')
                        os.makedirs(f'{self.args.log_dir}/saved_models/episode_{episode}/', exist_ok=True)
                        with open(f'{self.args.log_dir}/saved_models/episode_{episode}/tpe_mac_params.pkl', 'wb') as f:
                            pickle.dump(mac_params, f)
                        ep_log_dir = f'{self.args.log_dir}/saved_events/episode_{episode}/'
                        os.makedirs(ep_log_dir, exist_ok=True)
                        for file_name in os.listdir(self.args.log_dir):
                            if file_name.startswith("event"):
                                source_file = os.path.join(self.args.log_dir, file_name)
                                target_file = os.path.join(ep_log_dir, file_name)
                                shutil.copy(source_file, target_file)
                        print(f'episode {episode}  {self.args.log_dir}')
                        self.send_message(f'episode {episode}  {self.args.log_dir} {time.ctime()}')
                        # test code syncs

                    self.learn(buffer, learner, j, episode)
                    episode_return, action_record, episode_velo = \
                        self.episode_log(logger, episode, episode_return/t, action_record, episode_velo/t)

                last_ids = veh_ids.copy()

                if done or episode == self.args.run_episode:
                    break

        # com
        self.env.terminate()

        new_name = self.args.log_dir[:-7] + 'finished'
        os.rename(self.args.log_dir, new_name)

        return None

    def in_main_road(self, all_ids, num_rl, del_veh=False):

        # vehicles in main road
        veh_ids = []
        veh_type = []
        for id_veh in all_ids:
            if self.env.k.vehicle._TraCIVehicle__sumo_obs[id_veh][80] == 'highway_0':
                veh_ids.append(id_veh)
                veh_type.append(id_veh[5])

        # delete stopped vehicles
        new_ids = []
        num_delete = 0
        rl_id_num = 0
        for idx, ids in enumerate(veh_ids):
            if self.env.k.vehicle.get_speed(ids) < self.args.del_thres and del_veh and num_rl > 15:
                self.env.k.vehicle.remove(ids)
                num_delete += 1
                # new_ids.append(ids)
            else:
                if veh_type[idx] != '0':
                    if rl_id_num < 25:
                        rl_id_num += 1
                        new_ids.append(ids)
                    else:
                        self.env.k.vehicle.remove(ids)
                        num_delete += 1
                else:
                    new_ids.append(ids)

        if del_veh:
            self.in_out_num['delete'].append(num_delete)
        return new_ids

    def update_veh_log(self, veh_ids, last_ids):

        set_now = set(veh_ids)
        set_pas = set(last_ids)
        out_list = list(set_pas - set_now)
        in_list = list(set_now - set_pas)
        self.in_out_num['out'].append(len(out_list))
        self.in_out_num['in'].append(len(in_list))
        self.past_vehs += out_list
        for ids in in_list:
            self.vehicle_log[ids] = {'target': self.id_2_target(ids)}
        return in_list

    def get_vehicles_info(self, veh_ids):

        veh_ids_sorted = veh_ids.copy()
        state_dic = {}
        x_pos = [self.env.k.vehicle.get_2d_position(i)[0] for i in veh_ids]
        x_pos = np.array(x_pos)
        if not veh_ids == []:
            sorted_pairs = sorted(zip(x_pos, veh_ids_sorted), key=lambda x: x[0], reverse=True)
            x_pos, veh_ids = zip(*sorted_pairs)
        x_pos = np.array(x_pos)
        veh_ids = list(veh_ids)
        veh_types = [s[5] for s in veh_ids]
        veh_types = ['1' if x == '2' else x for x in veh_types]
        veh_lanes = self.env.k.vehicle.get_lane(veh_ids)
        velo = np.array([self.env.k.vehicle.get_speed(veh_ids)])
        veh_targ = [self.vehicle_log[veh_id]['target'] for veh_id in veh_ids]
        veh_map, left_hv, hv, right_hv = \
            self.map_vehs(x_pos, veh_lanes, veh_ids)

        for i, veh_id in enumerate(veh_ids):
            state_dic[veh_id] = [round(x_pos[i] / self.env.net_params.additional_params['length'], 4),
                                 veh_lanes[i],
                                 round(velo[0, i] / self.env.net_params.additional_params['speed_limit'], 4),
                                 int(veh_types[i]),
                                 int(veh_targ[i]),
                                 round(50.0 / left_hv[veh_id][0], 4),
                                 round(50.0 / hv[veh_id][0], 4),
                                 round(50.0 / right_hv[veh_id][0], 4)
                                 ]
        return veh_ids, veh_types, veh_lanes, x_pos, velo, veh_targ, state_dic, veh_map

    def map_vehs(self, x_pos, veh_lanes, ids):
        veh_map = np.zeros((self.env.net_params.additional_params['lanes'],
                            self.env.net_params.additional_params['length']))

        left_hv, hv, right_hv = {key: [1000, '', 1000, ''] for key in ids}, \
            {key: [1000, '', 1000, ''] for key in ids}, \
            {key: [1000, '', 1000, ''] for key in ids}
        for i in range(len(veh_lanes)):
            veh_map[veh_lanes[i], math.floor(x_pos[i])] = 1 + i
        for i in range(len(veh_lanes)):
            _x = math.floor(x_pos[i])
            for j in range(_x + 1, self.env.net_params.additional_params['length']):
                if not veh_map[veh_lanes[i], j] == 0:
                    hv[ids[i]][0] = j - _x
                    hv[ids[i]][1] = ids[int(veh_map[veh_lanes[i], j] - 1)]
                    break
            for j in range(_x, 0, -1):
                if veh_map[veh_lanes[i], j] != 0 and _x != j:
                    hv[ids[i]][2] = _x - j
                    hv[ids[i]][3] = ids[int(veh_map[veh_lanes[i], j] - 1)]
                    break
            if veh_lanes[i] + 1 in [0, 1, 2, 3]:
                for j in range(_x + 1, self.env.net_params.additional_params['length']):
                    if not veh_map[veh_lanes[i] + 1, j] == 0:
                        left_hv[ids[i]][0] = j - _x
                        left_hv[ids[i]][1] = ids[int(veh_map[veh_lanes[i] + 1, j] - 1)]
                        break
                for j in range(_x, 0, -1):
                    if not veh_map[veh_lanes[i] + 1, j] == 0:
                        left_hv[ids[i]][2] = _x - j
                        left_hv[ids[i]][3] = ids[int(veh_map[veh_lanes[i] + 1, j] - 1)]
                        break
            if veh_lanes[i] - 1 in [0, 1, 2, 3]:
                for j in range(_x + 1, self.env.net_params.additional_params['length']):
                    if not veh_map[veh_lanes[i] - 1, j] == 0:
                        right_hv[ids[i]][0] = j - _x
                        right_hv[ids[i]][1] = ids[int(veh_map[veh_lanes[i] - 1, j] - 1)]
                        break
                for j in range(_x, 0, -1):
                    if not veh_map[veh_lanes[i] - 1, j] == 0:
                        right_hv[ids[i]][2] = _x - j
                        right_hv[ids[i]][3] = ids[int(veh_map[veh_lanes[i] - 1, j] - 1)]
                        break
        return veh_map, left_hv, hv, right_hv

    def get_obs(self, rl_id, state_dict, left_hv, hv, right_hv):

        view_thres = 100
        # obs =  # whether to add self-info
        # [[],  # 左前方 车辆  0, 100 米内车辆数量，hw_1, hw_2
        #  [],  # 左后方 车辆  1
        #  [],  # 前方        2
        #  [],  # 后方        3
        #  [],  # 右前方      4
        #  [],  # 右后方      5
        # ]

        obs = [[] for i in range(6)]
        if not left_hv[rl_id][0] > view_thres:
            obs[0] = self.dict_state_diff(rl_id, left_hv[rl_id][1], state_dict)
        if not left_hv[rl_id][2] > view_thres:
            obs[1] = self.dict_state_diff(rl_id, left_hv[rl_id][3], state_dict)
        if not hv[rl_id][0] > view_thres:
            obs[2] = self.dict_state_diff(rl_id, hv[rl_id][1], state_dict)
        if not hv[rl_id][2] > view_thres:
            obs[3] = self.dict_state_diff(rl_id, hv[rl_id][3], state_dict)
        if not right_hv[rl_id][0] > view_thres:
            obs[4] = self.dict_state_diff(rl_id, right_hv[rl_id][1], state_dict)
        if not right_hv[rl_id][2] > view_thres:
            obs[5] = self.dict_state_diff(rl_id, right_hv[rl_id][3], state_dict)

        # size of obs_con: 1*8 + 6*5 = 38
        obs_con = np.zeros(38)
        obs_con[0:8] = np.array(state_dict[rl_id])
        for i in range(6):
            if not obs[i] == []:
                obs_con[8 + i*5: 13 + i*5] = np.array(obs[i])

        return obs_con

    @staticmethod
    def dict_state_diff(veh_1, veh_2, state_dict):
        state_1 = state_dict[veh_1].copy()
        state_2 = state_dict[veh_2].copy()
        diff_state = [state_1[i] - state_2[i] for i in range(3)]
        diff_state.append(state_2[3])
        diff_state.append(state_2[4])
        return diff_state

    def rl_actions(self, rl_obs):
        return None

    def get_available_action(self, rl_ids, state_dict, left_hv, hv, right_hv):
        speed_lim = self.env.net_params.additional_params['speed_limit']
        available_action = []
        for ids in rl_ids:
            available_action.append([1, 1, 1, 1, 1, 1, 1, 1, 1])

            if state_dict[ids][1] in [0]:
                available_action[-1][0:3] = [0, 0, 0]
            if state_dict[ids][1] in [3]:
                available_action[-1][6:9] = [0, 0, 0]
            if right_hv[ids][0] < 6 or right_hv[ids][2] < 6:
                available_action[-1][0:3] = [0, 0, 0]
            if left_hv[ids][0] < 6 or left_hv[ids][2] < 6:
                available_action[-1][6:9] = [0, 0, 0]

        return available_action

    def fill_transition(self, transition_dict):
        filled_dict = copy.deepcopy(transition_dict)
        if 'state' in transition_dict:
            filled_dict['state'] = [np.array([0] * 48)]
            while len(filled_dict['avail_actions'][0]) < self.args.n_agents:
                filled_dict['avail_actions'][0].append([1, 1, 1, 1, 1, 1, 1, 1, 1])
            filled_dict['obs'] = [[val for _, val in transition_dict['obs'].items()]]
            while len(filled_dict['obs'][0]) < self.args.n_agents:
                filled_dict['obs'][0].append(np.zeros(self.args.obs_dim))
        if 'actions' in transition_dict:

            in_flow = ()
            for i in range(self.args.lanes):
                if not filled_dict['inflow'][i] == []:
                    in_flow += tuple(filled_dict['inflow'][i])
                else:
                    in_flow += (0, 0,)
            filled_dict['inflow'] = in_flow

        return filled_dict

    def learn(self, buffer, learner, t_env, episode):
        for _ in range(self.args.num_circle):
            if buffer.can_sample(self.args.batch_size):

                episode_sample = buffer.sample(self.args.batch_size)

                # Truncate batch to only filled time steps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                learner.train(episode_sample, t_env, episode)
                # learner.train_predict(episode_sample, t_env)

    def could_change_lane(self, veh_id, direction):
        if direction == -1:
            right_hw = self.env.k.kernel_api.vehicle.getRightLeaders(veh_id)[0][1]
        return self.env.k.kernel_api.vehicle.getNeighbors(veh_id, direction)

    def env_step(self, rl_actions):

        next_observation, reward, done, infos = self.env.step(rl_actions)

        return next_observation, reward, done, infos

    def new_info_dict(self, veh_id):
        self.vehicle_log[veh_id] = {

        }
        return None

    @staticmethod
    def id_2_type(veh_id):
        veh_type = veh_id[5]
        veh_type = '1' if veh_type == '2' else veh_type
        veh_type = int(veh_type)
        return veh_type

    def id_2_target(self, veh_id):
        target = self.env.k.vehicle._TraCIVehicle__sumo_obs[veh_id][84][1][4]
        target = 2 - int(target)
        return target

    @staticmethod
    def episode_log(logger, episode, episode_return, action_rec, episode_velo):
        action_num = sum(action_rec) - action_rec[4]
        lc_num = sum(action_rec[0:3] + action_rec[6:9])
        acc = action_rec[2] + action_rec[5] + action_rec[8]
        dec = action_rec[0] + action_rec[3] + action_rec[6]
        logger.log_stat("episode_return", episode_return, episode)
        logger.log_stat("rl_lane_change", lc_num, episode)
        logger.log_stat("rl_acc", acc, episode)
        logger.log_stat("rl_dec", dec, episode)
        logger.log_stat("action_num", action_num, episode)
        logger.log_stat("episode_velo", episode_velo, episode)
        return 0, [0] * 9, 0

    def send_message(self, message):
        import requests
        server_ip = '172.168.137.3'
        server_port = 5000
        url = f'http://{server_ip}:{server_port}/report_progress'
        test_url = f'http://{server_ip}:{server_port}/'
        if self.is_port_available(test_url):
            response = requests.post(url, json={"progress": message})

    @staticmethod
    def is_port_available(url):
        import requests
        try:
            response = requests.get(url, timeout=5)  # 尝试连接，设置超时时间为5秒
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException:
            return False
