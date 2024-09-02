import random
import numpy as np

class TabularWorldModel:
    def __init__(self):
        self.model = {}

    def update_sampling_probabilities(self, new_sampling_probabilities):
        self._sampling_probabilities = new_sampling_probabilities
        
    def update(self, state, action, reward, next_state):
        if (state, action) not in self.model:
            self.model[(state, action)] = {'rewards': [], 'next_state': next_state}
        
        self.model[(state, action)]['rewards'].append(reward)

    def sample_state_action(self):
        if self._sampling_probabilities is None:
            raise ValueError("Probabilities matrix is not set. Please update probabilities first.")
        
        state_action_probs = []
        state_action_pairs = list(self.model.keys())
        
        for state, action in state_action_pairs:
            state_action_probs.append(self._sampling_probabilities[state, action])
        
        state_action_probs = np.array(state_action_probs)
        
        if state_action_probs.sum() == 0:
            state_action_probs = np.ones(len(state_action_probs))
    
        state_action_probs /= state_action_probs.sum()
        
        sampled_index = np.random.choice(len(state_action_pairs), p=state_action_probs)
        state, action = state_action_pairs[sampled_index]
        
        return state, action

    def predict(self, state, action):
        if (state, action) in self.model:
            rewards = self.model[(state, action)]['rewards']
            avg_reward = np.mean(rewards) if rewards else 0
            next_state = self.model[(state, action)]['next_state']
            return avg_reward, next_state
        else:
            return 0, None