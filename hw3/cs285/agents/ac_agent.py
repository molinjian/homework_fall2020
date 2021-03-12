from collections import OrderedDict
import numpy as np

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.infrastructure import pytorch_util as ptu

from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.gae = self.agent_params['gae']
        self.gae_lambda = self.agent_params['gae_lambda']
        self.ppo = self.agent_params['ppo']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.agent_params['clip_eps'],
        )

        if self.ppo:
            self.old_actor = MLPPolicyAC(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['discrete'],
                self.agent_params['learning_rate'],
                self.agent_params['clip_eps'],
            )
            self.old_actor.load_state_dict(self.actor.state_dict())

        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        rewards = np.concatenate([r for r in re_n]) if self.gae else re_n
        assert rewards.shape == terminal_n.shape
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            loss_critic = self.critic.update(ob_no, ac_na, next_ob_no, rewards, terminal_n)

        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        old_log_prob = self.get_old_prob(self.old_actor, ob_no, ac_na) if self.ppo else None

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss_actor = self.actor.update(ob_no, ac_na, advantage, old_log_prob)

        if self.ppo:
            self.old_actor.load_state_dict(self.actor.state_dict())


        loss = OrderedDict()
        loss['Critic_Loss'] = loss_critic
        loss['Actor_Loss'] = loss_actor

        return loss

    def get_old_prob(self, old_policy, ob_no, ac_na):
        observations = ptu.from_numpy(ob_no)
        actions = ptu.from_numpy(ac_na)
        log_prob = old_policy.forward(observations).log_prob(actions)

        return ptu.to_numpy(log_prob)

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        v_s = self.critic.forward_np(ob_no)
        if not self.gae:
            v_s_next = self.critic.forward_np(next_ob_no) * (1 - terminal_n)
            adv_n = re_n + self.gamma * v_s_next - v_s
        else:
            index = 0
            adv_n = np.zeros(len(ob_no))
            for rewards in re_n:
                gae_deltas = []
                for i in range(len(rewards) - 1):
                    delta = rewards[i] + self.gamma * v_s[index + i + 1] - v_s[index + i]
                    gae_deltas.append(delta)
                i = len(rewards) - 1
                gae_deltas.append(rewards[i] - v_s[index + i])

                assert len(gae_deltas) == len(rewards)

                sum_deltas = 0
                for t in range(len(gae_deltas) - 1, -1, -1):
                    sum_deltas = gae_deltas[t] + sum_deltas * self.gamma * self.gae_lambda
                    adv_n[t + index] = sum_deltas

                index += len(rewards)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        concat_rew = False if self.gae else True
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew)
