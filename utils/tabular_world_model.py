import numpy as np

class TabularWorldModel:
    def __init__(self, state_number, action_number, seed=42):
        self._state_number = state_number
        self._action_number = action_number
        
        self._next_state_counts = np.zeros((state_number, action_number, state_number), dtype=int)
        self._reward_sum = np.zeros((state_number, action_number), dtype=float)
        self._reward_count = np.zeros((state_number, action_number), dtype=int)
        
        self._sampling_probabilities = None
        np.random.seed(seed)
        
    def train(self, trajectories):
        for state, action, reward, next_state in trajectories:
            self._next_state_counts[state, action, next_state] += 1
            
            self._reward_sum[state, action] += reward
            self._reward_count[state, action] += 1
        
        self._next_state_predictions = self._next_state_counts.argmax(axis=2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self._reward_predictions = np.nan_to_num(self._reward_sum / self._reward_count)

    def update_sampling_probabilities(self, new_sampling_probabilities):
        if new_sampling_probabilities.shape != (self._state_number, self._action_number):
            raise ValueError(f"Sampling probabilities must have shape ({self._state_number}, {self._action_number})")
        self._sampling_probabilities = new_sampling_probabilities.copy()
    
    def batch_sample_state_actions(self, sample_number):
        if self._sampling_probabilities is None:
            raise ValueError("Probabilities matrix is not set. Please update probabilities first.")
        
        state_indices, action_indices = np.indices((self._state_number, self._action_number))
        flat_states = state_indices.flatten()
        flat_actions = action_indices.flatten()
        
        flat_probs = self._sampling_probabilities.flatten()
        
        if flat_probs.sum() == 0:
            flat_probs = np.ones_like(flat_probs)
        
        flat_probs = flat_probs / flat_probs.sum()
        
        sampled_indices = np.random.choice(len(flat_probs), size=sample_number, p=flat_probs)
        
        sampled_states = flat_states[sampled_indices]
        sampled_actions = flat_actions[sampled_indices]
        
        return np.stack((sampled_states, sampled_actions), axis=1)
    
    def batch_predict(self, state_action_pairs):
        if state_action_pairs.ndim != 2 or state_action_pairs.shape[1] != 2:
            raise ValueError("state_action_pairs must be an iterable of (state, action) tuples.")
        
        states = state_action_pairs[:, 0]
        actions = state_action_pairs[:, 1]
        
        next_states = self._next_state_predictions[states, actions]
        average_rewards = self._reward_predictions[states, actions]
        
        return next_states, average_rewards
