from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def __init__(self, env, policy, configuration):
        pass 
    
    @abstractmethod
    def online_learn(self, episodes_number, debug=False):
        pass
    
    @abstractmethod
    def offline_learn(self, offline_data, debug=False):
        pass
    
    @abstractmethod
    def play(self, episodes_number, save_trajectories=False, debug=False):
        pass