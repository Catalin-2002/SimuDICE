import gymnasium as gym
from environments.env import Environment

class FrozenLakeEnv(Environment):
    def __init__(self):
        self.env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)

    def reset_environment(self, seed_value):
        state, _ = self.env.reset(seed=seed_value)
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return next_state, reward, (terminated or truncated)

    def get_state_number(self):
        return self.env.observation_space.n

    def get_action_number(self):
        return self.env.action_space.n
