import gymnasium as gym
from environments.env import Environment

class FrozenLakeEnv(Environment):
    def __init__(self):
        self.env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
        self.env.reset()
        
    def reset_environment(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, truncated, _ = self.env.step(action)
        return next_state, reward, done
    
    def get_state_number(self):
        return self.env.observation_space.n

    def get_action_number(self):
        return self.env.action_space.n