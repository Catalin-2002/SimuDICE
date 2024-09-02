from abc import ABC, abstractmethod

class Environment(ABC):
    @abstractmethod
    def reset_environment(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_state_number(self):
        pass

    @abstractmethod
    def get_action_number(self):
        pass