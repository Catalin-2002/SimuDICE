import numpy as np
from policies.policy import Policy

class EpsilonGreedyPolicy(Policy): 
    def __init__(self, epsilon):
        self._epsilon = epsilon
        np.random.seed(0)
    
    def get_epsilon(self):
        return self._epsilon    
    
    def select_action(self, q_values):
        if np.random.uniform() < self._epsilon:
            return np.random.choice(len(q_values))
        else:
            max_value = np.max(q_values)
            max_indices = np.where(np.isclose(q_values, max_value))[0]
            return np.random.choice(max_indices)
                
    def get_probabilities(self, q_values):
        probabilities = np.ones(len(q_values)) * self._epsilon / len(q_values)
        best_action = np.argmax(q_values)
        probabilities[best_action] += 1 - self._epsilon
        return probabilities
    
    