################################################################
# ----This is tested class for customized Rware Env.------------
################################################################

import time
import numpy as np
import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
#import rware
from a1_rware_wrappers_seql import RecordEpisodeStatistics

class CustomizeRware:
    def __init__(self, env_scenario, scenario, ma_algo_name):

        # Create environment (gym interface) with class instantiation
        self.env = gym.make(env_scenario)

        # This wrapper is to add to the functionality of "info" dict
        # which adds reward per ep, per ag/ep, ep len, and ep time
        self.env = RecordEpisodeStatistics(self.env, deque_size=1000)

        self.env_scenario_name = scenario
        print("Task name: ", self.env_scenario_name)

        self.ma_algo_name = ma_algo_name
        print("Multi-agent algo name: ", self.ma_algo_name)

        self.n_agents = self.env.n_agents
        print("No. of agents: ", self.n_agents)

        # View observation and action space
        self.observation_space = self.env.observation_space
        #print("Observation space: ", [self.observation_space[i] for i in range(self.n_agents)])
        self.action_space = self.env.action_space
        #print("Action space: ", [self.action_space[i] for i in range(self.n_agents)])

        # Generate Neighbor and Distance Mask
        self.distance_mask = np.zeros((self.n_agents, self.n_agents)).astype(int)
        #print("Distance mask: ", self.distance_mask)
        self.neighbor_mask = np.zeros((self.n_agents, self.n_agents)).astype(int)
        #print("Neighbor mask: ", self.neighbor_mask)
        cur_distance = list(range(self.n_agents))
        #print("Current distance: ", cur_distance)
        for i in range(self.n_agents):
            self.distance_mask[i] = cur_distance
            #print("Distance mask: ", distance_mask)
            cur_distance = [i + 1] + cur_distance[:-1]
            #print("Current distance: ", cur_distance)
            if i >= 1:
                self.neighbor_mask[i, i - 1] = 1
                #print("Neighbor mask: ", neighbor_mask)
            # if i >= 2:
            # neighbor_mask[i,i-2] = 1
            if i <= self.n_agents - 2:
                self.neighbor_mask[i, i + 1] = 1
                #print("Neighbor mask: ", neighbor_mask)
        #print("Length Distance mask: ", len(self.distance_mask))
        #print("Length Neighbor mask: ", len(self.neighbor_mask))

        #print("Distance mask: ", self.distance_mask)
        #print("Neighbor mask: ", self.neighbor_mask)

    def extract_sizes(self, spaces):
        """
        Extract space dimensions
        :param spaces: list of Gym spaces
        :return: list of ints with sizes for each agent
        """
        sizes = []
        for space in spaces:
            if isinstance(space, Box):
                size = sum(space.shape)
            elif isinstance(space, Dict):
                size = sum(self.extract_sizes(space.values()))
            elif isinstance(space, Discrete) or isinstance(space, MultiBinary):
                size = space.n
            elif isinstance(space, MultiDiscrete):
                size = sum(space.nvec)
            else:
                raise ValueError("Unknown class of space: ", type(space))
            sizes.append(size)
        return sizes

    def initialize_env_dims_ia2c(self):
        """
        - Initialize env dimensions for "ia2c" multi-agent algo
        - return: observation_sizes, action_sizes
        """

        # Initialize observation dims/ sizes and extract action dims/ sizes
        # We will have to determine obs dims for "ia2c" manually as this algo
        # takes obs dims inputs differently.
        observation_sizes_ia2c = []  # Sizes == dimensions
        #print("Observation sizes: ", observation_sizes_ia2c)
        action_sizes_ia2c = self.extract_sizes(self.env.action_space)  # Sizes == dimensions
        #print("Action sizes for ia2c: ", action_sizes_ia2c)
        # Extract scalar value for obs dims
        #print(self.env.observation_space[0].shape[0])

        for i in range(self.n_agents):
            num_n = 1 + np.sum(self.neighbor_mask[i])
            # "env.observation_space[0].shape[0]" here because in original CACC env this value is "5"
            # for the obs dims as output in original by the CACC env
            observation_sizes_ia2c.append(num_n * self.env.observation_space[0].shape[0])
        #print("Observation sizes for ia2c: ", observation_sizes_ia2c)
        return (observation_sizes_ia2c, action_sizes_ia2c)

    def initialize_env_dims_ma2c(self):
        """
        - Initialize env dimensions for "ia2c" multi-agent algo
        - return: observation_sizes, action_sizes
        """

        # Extract observation and action dims/ sizes
        observation_sizes_ma2c = self.extract_sizes(self.env.observation_space)  # Sizes == dimensions
        #print("Observation dimensions or sizes: ", observation_sizes_ma2c)
        action_sizes_ma2c = self.extract_sizes(self.env.action_space)  # Sizes == dimensions
        #print("Action sizes for ma2c: ", action_sizes_ma2c)
        #print("Observation sizes for ma2c: ", observation_sizes_ma2c)

        return (observation_sizes_ma2c, action_sizes_ma2c)

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agents):
            naction.append(action[self.neighbor_mask[i] == 1])
        #print("Naction: ", naction)

        return naction

    def _make_obs_ia2c(self):
        """
        - Modify state/ observation if algo/ agent == "ia2c"
        """
        observation = self.env.reset()
        #observation = list(observation)
        #observation.clear()
        next_obs = []
        for i in range(self.n_agents):
            current_obs = [observation[i]]
            #print("Current observation: ", current_obs)
            if self.ma_algo_name.startswith('ia2c'):
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    current_obs.append(observation[j])
                    #print("Current observation: ", current_obs)
            next_obs.append(np.concatenate(current_obs))
            #print("Next observation: ", next_obs)
        #print("Final next observation: ", next_obs)

        return next_obs

    def environment_reset(self):
        """
        - Override original "env reset()"nt for "ia2c" algo
        - Override original "env reset()" to add fingerprint attribute same as CACC env
        - return: new observation as list obj
        """

        # "env.action_space[0].n" to extract single agent action dims here because in original CACC env
        # this value is "4" for the single agent action dims as output by the env
        n_a = self.env.action_space[0].n
        self.fp = np.ones((self.n_agents, n_a)) / n_a
        # print("Fingerprint: ", self.fp)

        """
        # Disable this condition for now
        if self.ma_algo_name.startswith('ia2c'):
            #obs = self.env.reset()
            #obs = list(obs)
            #obs.clear()
            obs = self._make_obs_ia2c()
        else:
            obs = self.env.reset()
            obs = list(obs)
        """

        obs = self.env.reset()
        obs = list(obs)

        return obs

    def environment_step(self, actions):
        """
        - Take step in the environment
        :param actions: actions to apply for each agent
        - Override original "env step()" for "ia2c" algo
        - return: New observation as list obj, Global reward as summed scalar value,
                  done as scalar value, Ep-wise summed reward as a scalar value
                  and info dict for any further info retrieval
        """

        # Environment step

        """
        # Disable this condition for now
        if self.ma_algo_name.startswith('ia2c'):
            #next_obs = self.env.reset()
            #next_obs = list(next_obs)
            #next_obs.clear()
            next_obs = self._make_obs_ia2c()
            _, reward, done, info = self.env.step(actions)

        else:
            next_obs, reward, done, info = self.env.step(actions)
            next_obs = list(next_obs)
        """

        next_obs, reward, done, info = self.env.step(actions)
        next_obs = list(next_obs)

        # By me
        #print("Next observation: ", next_obs)

        # Obtain scalar value for reward with list.sum() of reward attribute
        reward = sum(reward)
        #print("Step reward: ", step_reward)

        # Obtain scalar value for done
        #done = done[0]
        # Obtain array with single done
        #done = np.array(done[0])
        # print("Done: ", done)

        #done = all(done)

        # Retrieve reward np.array or summed reward scalar value for each episode from Info dict
        # due to "RecordEpisodeStatistics" wrapper
        #reward = np.array(info['episode_reward'])
        # With above "reward" gave following error;
        # runtimeerror: the size of tensor a (60) must match the size of tensor b (4) at non-singleton dimension 1

        #reward = sum(info['episode_reward'])
        # print("Episode training rewards: ", reward)

        #reward = np.sum(np.array(reward))  # Same as below
        #reward = np.sum(reward)
        #print("Reward:", reward)  # Output == 0.0

        # Keep "global_reward" off here.
        # Can be retrieved during runtime from "info" dict if required.
        #global_reward = np.sum(info["episode_reward"])
        #print("Global_reward: ", global_reward)

        return next_obs, reward, done, info
        #return next_obs, step_reward, done, reward, info

    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def environment_render(self):
        """
        Render visualisation of environment
        """
        self.env.render()
        time.sleep(0.1)

    def environment_close(self):
        """
        Close environment
        """
        self.env.close()