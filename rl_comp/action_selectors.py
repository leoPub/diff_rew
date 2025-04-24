import torch as th
from torch.distributions import Categorical

REGISTRY = {}


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass


class MultinomialActionSelector:

    def __init__(self, config):
        self.config = config

        self.epsilon_scheduler = DecayThenFlatSchedule(config.epsilon_start, config.epsilon_finish,
                                                       config.epsilon_anneal_time,
                                                       decay="linear")
        self.current_epsilon = self.epsilon_scheduler.eval(0)
        self.enable_test_greedy = getattr(config, "test_greedy", True)

    def select_action(self, agent_observations, available_actions, timestep, is_test=False):
        modified_policy = agent_observations.clone()
        modified_policy[available_actions == 0.0] = 0.0

        self.current_epsilon = self.epsilon_scheduler.eval(timestep)

        if is_test and self.enable_test_greedy:
            chosen_actions = modified_policy.max(dim=2)[1]
        else:
            chosen_actions = Categorical(modified_policy).sample().long()

        return chosen_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector:

    def __init__(self, config):
        self.config = config

        self.epsilon_scheduler = DecayThenFlatSchedule(config.epsilon_start, config.epsilon_finish,
                                                       config.epsilon_anneal_time,
                                                       decay="linear")
        self.current_epsilon = self.epsilon_scheduler.eval(0)

    def select_action(self, agent_observations, available_actions, timestep, is_test=False):
        self.current_epsilon = self.epsilon_scheduler.eval(timestep)

        if is_test:
            self.current_epsilon = 0.0

        masked_q_estimates = agent_observations.clone()
        masked_q_estimates[available_actions == 0.0] = -float("inf")

        random_values = th.rand_like(agent_observations[:, :, 0])
        random_selection_mask = (random_values < self.current_epsilon).long()
        random_choices = Categorical(available_actions.float()).sample().long()
        final_actions = random_selection_mask * random_choices + (1 - random_selection_mask) * \
                        masked_q_estimates.max(dim=2)[1]
        return final_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
