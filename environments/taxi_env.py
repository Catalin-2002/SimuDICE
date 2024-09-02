import gymnasium as gym
from environments.env import Environment

class TaxiEnv(Environment):
    def __init__(self):
        self._env = gym.make('Taxi-v3')

    def reset_environment(self):
        return self._env.reset()

    def step(self, action):
        next_state, reward, done, _, _ = self._env.step(action)
        return next_state, reward, done
    
    def get_state_number(self):
        return self._env.observation_space.n

    def get_action_number(self):
        return self._env.action_space.n
