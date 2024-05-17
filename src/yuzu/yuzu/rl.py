"""
yuzu.ml is a core reinforcement learning library based on yuzu core
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, NamedTuple
import yuzu as yz

class OneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_size = env.observation_space.n
        env.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,),)

    def observation(self, observation):
        return np.eye(self.obs_size)[observation].astype(np.float32)

class YuzuObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation):
        return yz.from_numpy(np.array(observation))

class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation):
        pass

@dataclass
class RLTrainStats:
    loss_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)
    length_history: list = field(default_factory=list)

class EpisodeBuffer(ABC):
    def __init__(self, maxlen, obs_shape, act_shape):
        self.observations = yz.zeros((maxlen,) + obs_shape)
        self.next_observations = yz.zeros((maxlen, ) + obs_shape)
        self.rewards = yz.zeros(maxlen)
        self.actions = yz.zeros((maxlen, ) + act_shape, dtype=yz.long)
        self.dones = yz.zeros((maxlen, 1))
        self.maxlen = maxlen
        self.cur_steps = 0
        self.cur_cum_reward = 0.0
        self.cur_observation = None

    @abstractmethod
    def add_record(self, cur_obs, observation, action, reward, done):
        pass

    @abstractmethod
    def stop_episode(self):
        pass

    @abstractmethod
    def start_episode(self):
        pass

    @abstractmethod
    def get_size(self):
        pass

    def step(self, env: gym.Env, agent: Agent, max_steps: int):
        if self.cur_observation is None:
            self.cur_observation, info = env.reset()
            self.start_episode()

        last_obs = self.cur_observation
        action = agent.get_action(last_obs)
        self.cur_observation, reward, terminated, truncated, info = env.step(action)
        self.add_record(last_obs, self.cur_observation, action, reward, terminated)
        self.cur_cum_reward += reward
        self.cur_steps += 1

        if self.cur_steps > max_steps or terminated:
            self.stop_episode()
            self.cur_steps = 0
            ret_reward = self.cur_cum_reward
            self.cur_cum_reward = 0.0
            self.cur_observation, info = env.reset()
            self.start_episode()
            return ret_reward
        return None

    def collect_one(self, env: gym.Env, agent: Agent, max_steps: int):
        while True:
            ret = self.step(env, agent, max_steps)
            if ret:
                return ret
        return None

class RolloutEpisodeBuffer(EpisodeBuffer):
    def __init__(self, maxlen, obs_shape, act_shape):
        super().__init__(maxlen, obs_shape, act_shape)
        self.episodes = []
        self.size = 0

    def add_record(self, cur_obs, observation, action, reward, done):
        self.observations[self.size] = cur_obs
        self.next_observations[self.size] = observation
        self.actions[self.size] = action
        self.rewards[self.size] = reward
        self.dones[self.size] = done
        self.size += 1

    def get_size(self):
        return self.size

    def start_episode(self):
        self.start = self.size

    def stop_episode(self):
        self.episodes.append((self.start, self.size))

    def cut(self):
        self.observations = self.observations[:self.size]
        self.next_observations = self.next_observations[:self.size]
        self.rewards = self.rewards[:self.size]
        self.actions = self.actions[:self.size]
        self.dones = self.dones[:self.size]

    def get_episodes(self):
        return self.episodes

class ReplayEpisodeBuffer(EpisodeBuffer):
    def __init__(self, maxlen, obs_shape, act_shape):
        super().__init__(maxlen, obs_shape, act_shape)
        self.cursor = 0
        self.size = 0

    def add_record(self, cur_obs, observation, action, reward, done):
        self.observations[self.cursor] = cur_obs
        self.next_observations[self.cursor] = observation
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.dones[self.cursor] = done 
        self.cursor += 1
        self.cursor %= self.maxlen
        self.size += 1
        self.size = min(self.size, self.maxlen)

    def get_size(self):
        return self.size

    def start_episode(self):
        pass

    def stop_episode(self):
        pass

    def sample(self, n):
        if self.size <= n:
            return (self.observations[:n], self.next_observations[:n], self.actions[:n], self.rewards[:n], self.dones[:n])
        else:
            perm = np.random.choice(range(self.size), n, replace=False)
            return (self.observations[perm], self.next_observations[perm], self.actions[perm], self.rewards[perm], self.dones[perm])
#
# Value iteration agent for tabular case
#
# we only try to approximate A(s,a) value in policy graident and set policy as
# pi(a|s) = 1 if a = argmax_a A(s,a)
#
# argmax_a A(s,a) = argmax_a Q(s,a) = r(s,a) + gamma * E[V(s')]
#
# value iteration algorithm:
# Q(s,a) = r(s,a) + gamma * E[V(s')]
# V(s) = max_a Q(s,a)
#
# pro: 
# - it coverges in tabular case gurnateed by fixed point iteration theorem (it's contraction)
#
# con:
# - requires full table which can be huge on memory and need to know state transition beforehand
#
@dataclass
class ValueIterationOptions:
    gamma: float = 0.99
    max_episode_len: int = 1024
    epochs: int = 1000
    report_train: Any = None
    report_reward: Any = None

class ValueIterationAgent(Agent):
    def __init__(self, state_size, action_size):
        inf = 1e9
        self.Q = np.random.rand(state_size, action_size)
        self.V = np.random.rand(state_size)

    def get_action(self, observation):
        return np.argmax(self.Q, axis=1)[observation]

# Assumes determinstic state transition
def value_iteration_train(create_env, agent: ValueIterationAgent, options: ValueIterationOptions):
    env = create_env()
    stats = RLTrainStats()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    transition = [[(-1,-1) for i in range(action_size)] for j in range(state_size)]
    for i in range(options.epochs):
        eb = RolloutEpisodeBuffer(options.max_episode_len, (1, ), (1,))
        eb.collect_one(env, agent, options.max_episode_len)

        tot_reward = yz.sum(eb.rewards).item()
        if options.report_reward:
            options.report_reward(tot_reward)
        stats.reward_history.append(tot_reward)

        m = eb.size
        for i in range(m):
            obs = int(eb.observations[i].item())
            act = int(eb.actions[i].item())
            nxt_obs = int(eb.next_observations[i].item())
            reward = eb.rewards[i].item()
            if transition[obs][act][0] == -1:
                transition[obs][act] = (nxt_obs, reward)

        for i in range(state_size):
            for j in range(action_size):
                if transition[i][j][0] != -1:
                    obs, reward = transition[i][j]
                    agent.Q[i][j] = reward + options.gamma*agent.V[obs]
        
        agent.V = np.max(agent.Q, axis=1)

    return stats

#
# DQN agent
#
# we only try to approximate A(s,a) value in policy graident and set policy as
# pi(a|s) = 1 if a = argmax_a A(s,a)
#
class DQNAgent(Agent):
    def __init__(self, model, dummy_model, device, action_space, eps_start, eps_end, eps_decay):
        if device:
            model = model.cuda()
            dummy_model = dummy_model.cuda()

        self.model = model
        self.target = dummy_model
        self.target.load_state_dict(self.model.state_dict())
        self.device = device
        self.action_space = action_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def apply_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def get_q_value(self, observation) -> yz.Node:
        if self.device:
            observation = observation.to(self.device)
        return self.model(observation)

    def get_q_value_target(self, observation) -> yz.Node:
        if self.device:
            observation = observation.to(self.device)
        return self.target(observation).detach()

    def get_eps_threshold(self):
        return self.eps_end + (self.eps_start- self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)

    def get_action(self, observation):
        sample = np.random.random()
        eps_threshold = self.get_eps_threshold()
        self.steps_done += 1
        if sample > eps_threshold:
            return self.get_q_value(observation).argmax(dim=1).detach().item()
        return self.action_space.sample()

@dataclass
class DQNOptions:
    optimizer: torch.optim.Optimizer
    gamma: float = 0.99
    num_steps : int = 1000
    max_episode_len: int = 1024
    replay_buf_size: int = 1024
    double_dqn: bool = False
    batch_size: int = 64
    train_interval: int = 128
    update_interval: int = 10000
    train_count: int = 64
    report_train: Any = None
    report_reward: Any = None

def dqn_train(create_env, agent: DQNAgent, device, options: DQNOptions):
    env = create_env()
    stats = RLTrainStats()
    eb = ReplayEpisodeBuffer(options.replay_buf_size, env.observation_space.shape, (1,))
    steps = 0
    for _ in range(options.num_steps):
        tot_reward = eb.step(env, agent, options.max_episode_len)
        if tot_reward:
            if options.report_reward:
                options.report_reward(tot_reward)
            stats.reward_history.append(tot_reward)
        steps += 1
        if steps % options.train_interval == 0:
            if eb.get_size() >= options.batch_size:
                for k in range(options.train_count):
                    b_obs, b_next_obs, b_acts, b_rewards, b_dones = eb.sample(options.batch_size)
                    b_obs, b_next_obs, b_acts, b_rewards, b_dones = b_obs.to(device), b_next_obs.to(device), b_acts.to(device), b_rewards.to(device), b_dones.to(device)
                    qs = agent.get_q_value(b_obs)
                    qvals = yz.gather(qs, dim=1, index=b_acts)
                    qvals_target = agent.get_q_value_target(b_next_obs)
                    if options.double_dqn:
                        qs2 = agent.get_q_value(b_next_obs)
                        q_acts = qs2.argmax(dim=1).long().reshape(-1,1).detach()
                        qvals_target = yz.gather(qvals_target, dim=1, index=q_acts)
                        y = b_rewards.reshape(-1,1) + (1.0-b_dones)*options.gamma*qvals_target
                    else:
                        y = b_rewards.reshape(-1,1) + (1.0-b_dones)*options.gamma*qvals_target.reduce_max(dim=1, keepdim=True)
                    loss = yz.wrap(F.smooth_l1_loss(qvals.inner(), y.inner()))

                    options.optimizer.zero_grad()
                    loss.backward()
                    options.optimizer.step()

                    if options.report_train:
                        options.report_train({
                            'loss': loss.item(),
                            'qval_mean': qvals.mean().item(),
                            'qval_target_mean': y.mean().item(),
                            'eps_threshold': agent.get_eps_threshold(),
                        })
                    stats.loss_history.append(loss.item())
        if steps % options.update_interval == 0: 
            agent.apply_target()
    return stats

#
# PPO agent 
#
# We are doing actor critic but try to be conservative about how much we change the
# policy by clipping importance sampling weight
#
# i.e. objective is KL(\theta | \theta_old) \leq delta
# 
# Policy graident is approximated as follows:
#
# ratio = \pi_{\theta}(a|s) / \pi_{\theta_old}(a|s) 
# L = min(ratio * A_{\theta_old}(s,a), clamp(ratio, 1-epsilon,1+epsilon) * A_{\theta_old}(s,a))
# grad = dL/d\theta
#
# pro:
# - it's one of state of art algorithms
# - very flexible on how you feed data (i.e. advantage can be normalized and still works fine)
#
# con:
# - many implementation details
# - very sensitivie on intial conditions as it's still policy graident that data is collected by running its policy
#
#
# implementation details done:
# - samples collected from vectorized environment and batches splitted in this "merged" buffer 
# -- running multiple training with same data randomized is required as PPO assumes we do this
#    and try to be conservative about the change by clipping ratio
# -- ratio must be all 1's in first minibatch'
# -- "inifite tape" that can still run the environment after agent dies so that we always get fixed size of observations
#   not implemented yet
# - layer initialization by orthogonal weight initialization
# -- seems to improve convergence
# -- (TODO) try implemnting this by myself
# - (TODO) implement GAE.
#

class A2CAgent(Agent):
    def __init__(self):
        pass

    @abstractmethod
    def get_policy(self, observation) -> yz.Distribution:
        pass

    @abstractmethod
    def get_value(self, observation) -> yz.Node:
        pass

    def get_action(self, observation):
        return self.get_policy(observation).sample().item()

class A2CMLAgent(A2CAgent):
    def __init__(self, model, device = None):
        if device:
            model = model.cuda()
        self.model = model
        self.device = device

    def get_policy(self, observation) -> yz.Distribution:
        if self.device:
            observation = observation.to(self.device)
        return self.model.get_policy(observation)

    def get_value(self, observation) -> yz.Node:
        if self.device:
            observation = observation.to(self.device)
        return self.model.get_value(observation)

@dataclass
class PPOOptions:
    optimizer: torch.optim.Optimizer
    gamma: float = 0.99
    epsilon: float = 0.1
    batch_nums: int = 4
    max_episode_len: int = 1024
    epochs: int = 1000
    entropy: float = 0.01
    num_envs: int = 4
    train_count: int = 1
    report_train: Any = None
    report_reward: Any = None
    create_env: Any = None
    batch_size: Any = None

def ppo_train(create_env, agent: A2CAgent, device, options: PPOOptions):
    env = create_env()
    stats = RLTrainStats()

    for i in range(options.epochs):
        episode_buffer = RolloutEpisodeBuffer(options.num_envs * options.max_episode_len, env.observation_space.shape, (1,),)
        for k in range(options.num_envs):
            episode_buffer.collect_one(env, agent, options.max_episode_len)

        episode_buffer.cut()
        m = episode_buffer.size
        batch_rewards_to_go = yz.zeros(m)
        for (s,e) in episode_buffer.episodes:
            for i in reversed(range(s, e-1)):
                batch_rewards_to_go[i] += episode_buffer.rewards[i] + options.gamma*batch_rewards_to_go[i+1]
            tot_reward = yz.sum(episode_buffer.rewards[s:e]).item()
            if options.report_reward:
                options.report_reward(tot_reward)
            stats.reward_history.append(tot_reward)

        batch_obs = episode_buffer.observations.to(device)
        batch_next_obs = episode_buffer.next_observations.to(device)
        batch_rewards = episode_buffer.rewards.to(device)
        batch_rewards_to_go = batch_rewards_to_go.to(device)
        batch_acts = episode_buffer.actions.to(device)

        A = (batch_rewards_to_go - agent.get_value(batch_obs).reshape(-1)).detach()
        logp_old = yz.wrap(agent.get_policy(batch_obs).log_prob(batch_acts.reshape(-1).inner())).detach()

        if options.batch_size:
            batch_size = options.batch_size
        else:
            batch_size = (m + options.batch_nums - 1) // options.batch_nums

        for _ in range(options.train_count):
            perm = np.random.permutation(m)
            batch_obs = batch_obs[perm]
            batch_next_obs = batch_next_obs[perm]
            batch_rewards = batch_rewards[perm]
            batch_rewards_to_go = batch_rewards_to_go[perm]
            batch_acts = batch_acts[perm]

            A = A[perm]
            logp_old = logp_old[perm]
            for j in range((m+batch_size-1)//batch_size):
                start = j*batch_size
                end = min((j+1)*batch_size,m)

                options.optimizer.zero_grad()
                y = options.gamma*agent.get_value(batch_next_obs[start:end]).reshape(-1) + batch_rewards[start:end]

                logp = yz.wrap(agent.get_policy(batch_obs[start:end]).log_prob(batch_acts[start:end].reshape(-1).inner()))

                log_ratio = (logp - logp_old[start:end])
                ratios = log_ratio.exp()

                # http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratios - 1) - log_ratio).mean()

                surr1 = ratios * A[start:end]
                surr2 = yz.clamp(ratios, 1-options.epsilon, 1+options.epsilon) * A[start:end]

                loss1 = yz.mse_loss(y, batch_rewards_to_go[start:end])
                loss2 = -yz.min(surr1,surr2).mean()

                entropy = agent.get_policy(batch_obs[start:end]).entropy()
                ent = entropy.mean()

                total_loss = (0.5*loss1 + loss2 - options.entropy * ent) / options.batch_nums
                total_loss.backward()
                options.optimizer.step()

                if options.report_train:
                    options.report_train({
                        'total_loss': total_loss.item(),
                        'value_loss': loss1.item(),
                        'policy_loss': loss2.item(),
                        'entropy': ent.item(),
                        'ratios': ratios.mean().item(),
                        'approx_kl': approx_kl.item()
                    })
                stats.loss_history.append(total_loss.item())
    return stats

def run(create_env, agent, device, max_episode_len=int(1e5)):
    env = create_env()
    while True:
        observation, info = env.reset()
        env.render()

        for j in range(max_episode_len):
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated:
                break

def run_jupyter(create_env, agent, device, max_episode_len=int(1e5)):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    env = create_env('rgb_array')

    while True:
        observation, info = env.reset()

        for j in range(max_episode_len):
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
            if terminated:
                break

