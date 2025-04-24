from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams, VehicleParams, \
    TrafficLightParams
from net_highway import HighwayNetwork
from flow.controllers import SimLaneChangeController, ContinuousRouter, IDMController, RLController, \
    SimCarFollowingController
from env_highway_ctn import MultiAgentHighwayPOEnv
import torch
from types import SimpleNamespace as simNp
from rl_comp.episode_buffer import ReplayBuffer, EpisodeBatch
from rl_comp.transforms import OneHot
from functools import partial
from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from utils.logging import get_logger, Logger
from config import config
from datetime import datetime
import os


def run_exp(flow_rate, penetration, run_episode=4000, render=False, test=False, log_dir='tb_logs', compared_method=None, print_warning=True):

    os.makedirs(log_dir, exist_ok=True)
    args = simNp(**config)
    args.device = args.GPU if args.use_cuda else "cpu"
    args.run_episode = run_episode
    args.log_dir = log_dir

    # Set up SUMO to render the results, take a time_step of 0.5 seconds per simulation step
    disable_tb = True,
    disable_ramp_meter = True,
    n_crit = 1000,
    feedback_coef = 20

    sim_params = SumoParams(
        sim_step=0.1,
        render=render,
        overtake_right=False,
        restart_instance=True,
        print_warnings=print_warning,
    )

    vehicles = VehicleParams()

    # Add a few vehicles to initialize the simulation. The vehicles have all lane changing enabled,
    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        acceleration_controller=(SimCarFollowingController, {}),
        routing_controller=(ContinuousRouter, {}),
        lane_change_params=SumoLaneChangeParams('only_strategic_aggressive'),
    )

    vehicles.add(
        veh_id="autonomous_1",
        lane_change_controller=(SimLaneChangeController, {}),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        lane_change_params=SumoLaneChangeParams('only_strategic_aggressive'),
        color='red',
    )

    vehicles.add(
        veh_id="auto_345671",
        lane_change_controller=(SimLaneChangeController, {}),
        # acceleration_controller=(SimCarFollowingController, {}),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        lane_change_params=SumoLaneChangeParams('only_strategic_aggressive'),
        color='green',
    )

    # These are additional params that configure the bottleneck experiment. They are explained in more
    # detail below.
    additional_env_params = {
        "target_velocity": 25,
        "max_accel": 1,
        "max_decel": 1,
        "lane_change_duration": 5,
        "add_rl_if_exit": False,
        "disable_tb": disable_tb,
        "disable_ramp_metering": disable_ramp_meter,
        "n_crit": n_crit,
        "feedback_coeff": feedback_coef,
    }
    # Set up the experiment to run for 1000 time steps i.e. 500 seconds (1000 * 0.5)
    env_params = EnvParams(
        horizon=1000, additional_params=additional_env_params)

    flow_rate = flow_rate * args.lanes
    # Add vehicle inflows at the front of the bottleneck. They enter with a flow_rate number of vehicles
    inflow = InFlows()
    [t1_rate, t2_rate, t3_rate] = [flow_rate * (1 - penetration), flow_rate * penetration / 2, flow_rate * penetration / 2]
    if penetration != 1.0:
        inflow.add(
            veh_type="human",
            edge="highway_0",
            vehs_per_hour=t1_rate,
            insertionChecks="all",
            # probability=p1,
            depart_lane="random",
            depart_speed=5)

    inflow.add(
        veh_type="autonomous_1",
        edge="highway_0",
        vehs_per_hour=t2_rate,
        insertionChecks="all",
        # probability=p2,
        # begin=3.5,
        depart_lane="random",
        depart_speed=5)

    inflow.add(
        veh_type="auto_345671",
        edge="highway_0",
        vehs_per_hour=t3_rate,
        insertionChecks="all",
        # probability=p3,
        # begin=6,
        depart_lane="random",
        depart_speed=5)

    # Initialize the traffic lights. The meanings of disable_tb and disable_ramp_meter are discussed later.
    traffic_lights = TrafficLightParams()

    additional_net_params = {"speed_limit": 25,
                             "length": 250,
                             "lanes": 4,
                             "num_edges": 1,
                             "use_ghost_edge": False,
                             # speed limit for the ghost edge
                             "ghost_speed_limit": 25,
                             # length of the downstream ghost edge with the reduced speed limit
                             "boundary_cell_length": 500
                             }

    net_params = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution="all")

    flow_params = dict(
        exp_tag='highway_ctn_exp',
        env_name=MultiAgentHighwayPOEnv,
        network=HighwayNetwork,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
        tls=traffic_lights,
    )

    # number of time steps
    flow_params['env'].horizon = int(3e10)

    buffer_scheme = {'state': {'vshape': args.state_shape},
                     'obs': {'vshape': args.obs_dim, 'group': 'agents'},
                     'actions': {'vshape': (1,),
                                 'group': 'agents',
                                 'dtype': torch.int64},
                     'avail_actions': {'vshape': (args.n_actions,),
                                       'group': 'agents',
                                       'dtype': torch.int32},
                     'reward': {'vshape': (1,)},
                     'terminated': {'vshape': (1,),
                                    'dtype': torch.uint8},
                     'inflow': {'vshape': (2*args.lanes,)},
                     'agents_num': {'vshape': (1,)}
                     }
    groups = {'agents': args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(output_dimension=args.n_actions)])}
    logger = Logger(get_logger())
    logger.setup_tb(log_dir)
    batch = partial(EpisodeBatch, buffer_scheme, groups, 1, args.episode_limit + 1,
                    preprocess=preprocess, device=args.device)
    buffer = ReplayBuffer(buffer_scheme, groups, args.buffer_size, args.episode_limit + 1,
                          args.burn_in_period,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    learner.cuda()
    # run the sumo simulation

    Experiment = exp_import(compared_method, test)
    exp = Experiment(flow_params, args)
    _ = exp.run(num_runs=1, batch_buffer=buffer, batch_fu=batch, mac=mac, learner=learner, logger=logger, test=test)
    return None


def exp_import(compared_method, test=False):
    if not test:
        from experiment_ctn import Experiment
    else:
        from experiment_test import Experiment
    return Experiment


if __name__ == "__main__":
    # main from here

    # test = True
    test = False
    now = datetime.now()
    time_str = now.strftime("%m%d%H%M%S")
    method = 'QMIX'  # maddpg mappo madqn qmix
    flow_rate = 250  # veh/(lane*hour)
    penetration = 1.0  # n_cav / n_veh  0.25 0.5 0.75 1.0

    # not marked dirs are with configurations of f250 and p0.5

    log_dir = f'tb_logs/{time_str + method}_f{flow_rate}_p{penetration}'

    run_exp(
        flow_rate=flow_rate,
        penetration=penetration,
        run_episode=52005,
        render=True,
        test=test,
        log_dir=log_dir,
        compared_method=method,
        print_warning=False,
            )
