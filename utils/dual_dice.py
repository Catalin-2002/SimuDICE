import numpy as np

class DualDICE:    
    def __init__(self, num_states, num_actions, gamma): 
            self._num_states = num_states 
            self._num_actions = num_actions
            self._gamma = gamma
            self._estimator_dimensions = num_states * num_actions
            
            self._nu = np.zeros([self._estimator_dimensions])
            self._zeta = np.zeros([self._estimator_dimensions])
            
    def _calculate_index(self, state, action):
        return state * self._num_actions + action
    
    def get_weight_estimates(self, behavioral_data, Q_values, target_policy, regularizer= 1e-8):
        td_residuals = np.zeros([self._estimator_dimensions, self._estimator_dimensions]) # Time difference residuals
        total_weights = np.zeros([self._estimator_dimensions])
        initial_weights = np.zeros([self._estimator_dimensions])
        
        for episode in behavioral_data:
            step_number = 0
            initial_state = episode[0][0]
            for (state, action, _, next_state) in episode:
                nu_index = self._calculate_index(state, action)
                weight = self._gamma ** step_number
                
                td_residuals[nu_index, nu_index] += weight
                total_weights[nu_index] += weight
                
                
                next_probabilities = target_policy.get_probabilities(Q_values[next_state])
                
                for (next_action, next_probability) in enumerate(next_probabilities):
                    next_nu_index = self._calculate_index(next_state, next_action)
                    td_residuals[next_nu_index, nu_index] += -next_probability * self._gamma * weight
                
                initial_probabilities = target_policy.get_probabilities(Q_values[initial_state])
                for (initial_action, initial_probability) in enumerate(initial_probabilities):
                    initial_nu_index = self._calculate_index(state, initial_action)
                    initial_weights[initial_nu_index] += initial_probability * weight
                
                step_number += 1
        
        td_residuals /= np.sqrt(regularizer + total_weights)[None, :]
        td_errors = np.dot(td_residuals, td_residuals.T)
        
        self._nu = np.linalg.solve(td_errors + regularizer * np.eye(self._estimator_dimensions), initial_weights * (1 - self._gamma))
        self._zeta = np.dot(self._nu, td_residuals) / np.sqrt(regularizer + total_weights)
        
        return self._zeta