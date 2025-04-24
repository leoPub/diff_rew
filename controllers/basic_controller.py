from modules.agents import REGISTRY as agt_REG
from rl_comp.action_selectors import REGISTRY as act_REG
import torch as th


class BasicMAC:
    def __init__(self, configuration, group_config, params):
        self.agent_count = params.n_agents
        self.params = params
        self.input_config = self._get_input_shape(configuration)
        self._build_agents(self.input_config)
        self.output_type_config = params.agent_output_type

        self.action_selector = act_REG[params.action_selector](params)
        self.latent_states = None

    def select_actions(self, episode_data, current_step, env_timestep, batch_slice=slice(None), is_testing=False):
        available_actions = episode_data["avail_actions"][:, current_step]
        policy_outputs = self.forward(episode_data, current_step, test_mode=is_testing)
        selected_actions = self.action_selector.select_action(
            policy_outputs[batch_slice], available_actions[batch_slice], env_timestep, is_test=is_testing)
        return selected_actions

    def forward(self, episode_data, timestep, test_mode=False):
        agent_observations = self._build_inputs(episode_data, timestep)
        available_actions = episode_data["avail_actions"][:, timestep]
        policy_logits, self.latent_states, _, _ = self.agent(
            agent_observations, self.latent_states)

        if self.output_type_config == "pi_logits":
            if getattr(self.params, "mask_before_softmax", True):
                reshaped_available_actions = available_actions.reshape(
                    episode_data.batch_size * self.agent_count, -1)
                policy_logits[reshaped_available_actions == 0] = -1e10

            policy_logits = th.nn.functional.softmax(policy_logits, dim=-1)
            if not test_mode:
                action_space_size = policy_logits.size(-1)
                if getattr(self.params, "mask_before_softmax", True):
                    action_space_size = reshaped_available_actions.sum(
                        dim=1, keepdim=True).float()

                policy_logits = ((1 - self.action_selector.epsilon) * policy_logits
                                 + th.ones_like(policy_logits) * self.action_selector.epsilon / action_space_size)

                if getattr(self.params, "mask_before_softmax", True):
                    policy_logits[reshaped_available_actions == 0] = 0.0

        return policy_logits.view(episode_data.batch_size, self.agent_count, -1)

    def init_hidden(self, batch_size):
        self.latent_states = self.agent.init_hidden().unsqueeze(
            0).expand(batch_size, self.agent_count, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_controller):
        self.agent.load_state_dict(other_controller.agent.state_dict())

    def cuda(self):
        self.agent.to(self.params.GPU)

    def save_models(self, save_path):
        th.save(self.agent.state_dict(), f"{save_path}/agent.th")

    def load_models(self, load_path):
        self.agent.load_state_dict(
            th.load(f"{load_path}/agent.th", map_location=lambda storage, loc: storage))

    def _build_agents(self, input_config):
        self.agent = agt_REG[self.params.agent](input_config, self.params)

    def _build_inputs(self, batch_data, timestep):
        batch_size = batch_data.batch_size
        input_components = []
        input_components.append(batch_data["obs"][:, timestep])

        traffic_inflow = batch_data["inflow"].unsqueeze(2).repeat(1, 1, self.agent_count, 1)
        if self.params.obs_last_action:
            if timestep == 0:
                input_components.append(th.zeros_like(batch_data["actions_onehot"][:, timestep]))
                input_components.append(th.zeros_like(traffic_inflow[:, timestep]))
            else:
                input_components.append(batch_data["actions_onehot"][:, timestep - 1])
                input_components.append(traffic_inflow[:, timestep - 1])
        if self.params.obs_agent_id:
            input_components.append(th.eye(self.agent_count, device=batch_data.device).unsqueeze(
                0).expand(batch_size, -1, -1))

        input_components = th.cat([x.reshape(batch_size * self.agent_count, -1)
                                   for x in input_components], dim=1)
        return input_components

    def _get_input_shape(self, configuration):
        input_dimension = configuration["obs"]["vshape"]
        if self.params.obs_last_action:
            input_dimension += configuration["actions_onehot"]["vshape"][0]
            input_dimension += configuration["inflow"]["vshape"][0]
        if self.params.obs_agent_id:
            input_dimension += self.agent_count

        return input_dimension
