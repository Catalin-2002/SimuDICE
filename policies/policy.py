from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, state_values):
        pass
    
    @abstractmethod
    def get_probabilities(self, state_values):
        pass