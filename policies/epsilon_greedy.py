import numpy as np
from policies.policy import Policy

class EpsilonGreedyPolicy(Policy): 
    def __init__(self, epsilon):
        self._epsilon = epsilon
    
    def get_epsilon(self):
        return self._epsilon    
    
    def select_action(self, q_values):
        if np.random.uniform() < self._epsilon or np.argmax(q_values) == 0:
            return np.random.choice(len(q_values))
        else:
            return np.argmax(q_values)
                
    def get_probabilities(self, q_values):
        probabilities = np.ones(len(q_values)) * self._epsilon / len(q_values)
        best_action = np.argmax(q_values)
        probabilities[best_action] += 1 - self._epsilon
        return probabilities
    
    