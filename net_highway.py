"""Contains the highway network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # length of the highway
    "length": 1000,
    # number of lanes
    "lanes": 4,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 1,
    # whether to include a ghost edge. This edge is provided a different speed
    # limit.
    "use_ghost_edge": False,
    # speed limit for the ghost edge
    "ghost_speed_limit": 25,
    # length of the downstream ghost edge with the reduced speed limit
    "boundary_cell_length": 500
}


class HighwayNetwork(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a highway network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):

        nodes = [{'id': 'edge_0', 'x': 0.0, 'y': 0.0},
                 {'id': 'edge_1', 'x': 250.0, 'y': 0.0},
                 {'id': 'tail_0', 'x': 250.0, 'y': 20.0},
                 {'id': 'tail_1', 'x': 270.0, 'y': 0.0},
                 {'id': 'tail_2', 'x': 250.0, 'y': -20.0}, ]

        return nodes

    def specify_edges(self, net_params):

        edges = [
            {'id': 'highway_0', 'type': 'highwayType', 'from': 'edge_0', 'to': 'edge_1', 'length': 250.0},
            {'id': 'end_0', 'type': 'endType', 'from': 'edge_1', 'to': 'tail_0', 'length': 20.0},
            {'id': 'end_1', 'type': 'endType_1', 'from': 'edge_1', 'to': 'tail_1', 'length': 20.0},
            {'id': 'end_2', 'type': 'endType', 'from': 'edge_1', 'to': 'tail_2', 'length': 20.0},
            {'id': 'highway_r', 'type': 'highwayType', 'from': 'edge_1', 'to': 'edge_0', 'length': 250.0},
            {'id': 'end_0_r', 'type': 'endType', 'from': 'tail_0', 'to': 'edge_1', 'length': 20.0},
            {'id': 'end_1_r', 'type': 'endType_1', 'from': 'tail_1', 'to': 'edge_1', 'length': 20.0},
            {'id': 'end_2_r', 'type': 'endType', 'from': 'tail_2', 'to': 'edge_1', 'length': 20.0},
        ]

        return edges

    def specify_types(self, net_params):

        types = [{'id': 'highwayType', 'numLanes': 4, 'speed': 23},
                 {'id': 'endType', 'numLanes': 4, 'speed': 23},
                 {'id': 'endType_1', 'numLanes': 4, 'speed': 23}]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        # num_edges = net_params.additional_params.get("num_edges", 1)
        # rts = {}
        # for i in range(num_edges):
        #     rts["highway_{}".format(i)] = ["highway_{}".format(j) for
        #                                    j in range(i, num_edges)]
        #     if self.net_params.additional_params["use_ghost_edge"]:
        #         rts["highway_{}".format(i)].append("highway_end")

        rts = {'highway_0': [(['highway_0', 'end_0'], 0.33),
                             (['highway_0', 'end_1'], 0.7),
                             (['highway_0', 'end_2'], 0.33)],
               'end_0': ['end_0'],
               'end_1': ['end_1'],
               'end_2': ['end_2']
               }

        return rts

    def specify_connections(self, net_params):

        # conn = {'center0': [
        #     {'from': 'highway_0', 'to': 'end_0', 'fromLane': '3', 'toLane': '0'},
        #     {'from': 'highway_0', 'to': 'end_1', 'fromLane': '0', 'toLane': '0'},
        #     {'from': 'highway_0', 'to': 'end_1', 'fromLane': '1', 'toLane': '1'},
        #     {'from': 'highway_0', 'to': 'end_1', 'fromLane': '2', 'toLane': '2'},
        #     {'from': 'highway_0', 'to': 'end_1', 'fromLane': '3', 'toLane': '3'},
        #     {'from': 'highway_0', 'to': 'end_2', 'fromLane': '0', 'toLane': '0'}]}
        conn = {'center0': [
            {'from': 'highway_0', 'to': 'end_0', 'fromLane': '0', 'toLane': '0'},
            {'from': 'highway_0', 'to': 'end_0', 'fromLane': '1', 'toLane': '1'},
            {'from': 'highway_0', 'to': 'end_0', 'fromLane': '2', 'toLane': '2'},
            {'from': 'highway_0', 'to': 'end_0', 'fromLane': '3', 'toLane': '3'},
            {'from': 'highway_0', 'to': 'end_1', 'fromLane': '0', 'toLane': '0'},
            {'from': 'highway_0', 'to': 'end_1', 'fromLane': '1', 'toLane': '1'},
            {'from': 'highway_0', 'to': 'end_1', 'fromLane': '2', 'toLane': '2'},
            {'from': 'highway_0', 'to': 'end_1', 'fromLane': '3', 'toLane': '3'},
            {'from': 'highway_0', 'to': 'end_2', 'fromLane': '0', 'toLane': '0'},
            {'from': 'highway_0', 'to': 'end_2', 'fromLane': '1', 'toLane': '1'},
            {'from': 'highway_0', 'to': 'end_2', 'fromLane': '2', 'toLane': '2'},
            {'from': 'highway_0', 'to': 'end_2', 'fromLane': '3', 'toLane': '3'}]}

        return conn

    def specify_edge_starts(self):
        """See parent class."""

        edge_starts = [('highway_0', 0.0)]

        return edge_starts

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """Generate a user defined set of starting positions.

        This method is just used for testing.
        """
        return initial_config.additional_params["start_positions"], \
            initial_config.additional_params["start_lanes"]
