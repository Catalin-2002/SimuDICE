import numpy as np
from agents.agent import Agent

from utils.tabular_world_model import TabularWorldModel
from utils.dual_dice import DualDICE

class SimuDICE(Agent): 
    def __init__(self, env, policy, world_model, configuration):
        self._env = env
        self._policy = policy
        self._world_model = world_model
        
        self._alpha = configuration.get('alpha', 0.1)
        self._gamma = configuration.get('gamma', 0.99)
        self._planning_steps = configuration.get('planning_steps', 10)
        self._iterations = configuration.get('iterations', 10)
        self._lambda = configuration.get('lambda', 100)
        
        self._sampling_strategy = configuration.get('sampling_strategy', 0)
        
        self.q_values = np.zeros((env.get_state_number(), env.get_action_number()))
        
    def set_policy(self, policy):
        self._policy = policy
        
    def online_learn(self, episodes_number, max_environment_steps, debug=False):
        pass
            
    def offline_learn(self, offline_data, debug=False):
        trajectories = [item for sublist in offline_data for item in sublist]
        np.random.shuffle(trajectories)
        
        # Learn initial Q-values
        for (state, action, reward, next_state) in trajectories:
            self.q_values[state, action] += self._alpha * (reward + self._gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])
            self._world_model.update(state, action, reward, next_state)
        
        # Estimate model confidence
        model_confidence_estimation = self.get_model_confidence_estimation(trajectories)
            
        # Plan
        planning_steps_per_iteration = self._planning_steps // self._iterations
        planning_steps_counter = 0
        
        for iteration in range(self._iterations):
            target_planning_steps = planning_steps_per_iteration if iteration != self._iterations - 1 else self._planning_steps - planning_steps_counter
            planning_steps_counter += target_planning_steps
            
            # Train DualDICE
            dice_estimator = DualDICE(self._env.get_state_number(), self._env.get_action_number(), self._gamma)
            zeta_values = dice_estimator.get_weight_estimates(offline_data, self.q_values, self._policy)
            
            zeta_values /= np.max(zeta_values) if np.max(zeta_values) > 0 else 1
            zeta_values *= self._lambda
            
            sampling_formula = self.get_sampling_formula(zeta_values, model_confidence_estimation)
            self._world_model.update_sampling_probabilities(sampling_formula)
            
            for _ in range(target_planning_steps * len(trajectories)):
                state, action = self._world_model.sample_state_action()
                reward, next_state = self._world_model.predict(state, action)
                self.q_values[state, action] += self._alpha * (reward + self._gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])
            
            if debug:
                print('Finished SimuDICE training iteration: {0}'.format(iteration + 1))
            
    def get_model_confidence_estimation(self, offline_data):
        state_action_confidence = np.zeros((self._env.get_state_number(), self._env.get_action_number()))
        for (state, action, reward, next_state) in offline_data:
            state_action_confidence[state, action] += 1
            
        state_action_confidence /= np.sum(state_action_confidence)
        return state_action_confidence
        
        
    def play(self, episodes_number, max_environment_steps, save_trajectories=False, debug=False):
        rewards = []
        trajectories = []
        
        for episode in range(episodes_number):
            state, _ = self._env.reset_environment()
            done = False
            environment_step = 0
            episode_trajectory = []
            
            while not done:
                action = self._policy.select_action(self.q_values[state])
                next_state, reward, done = self._env.step(action)
                
                episode_trajectory.append((state, action, reward, next_state))

                state = next_state
                environment_step += 1
                if environment_step == max_environment_steps:
                    break
                
            rewards.append(reward / environment_step)
            trajectories.append(episode_trajectory)
                
            if debug and episode % 100 == 0:
                print('Finished online play QLearningAgent for episode: {0}'.format(episode + 1))
                
        print('Finished playing with average reward: {0}'.format(np.mean(rewards)))
        if save_trajectories:
            return trajectories
        return None
    
    def get_policy_data(self):
        return {'q_values': self.q_values, 'epsilon': self._policy.get_epsilon()}
    
    def get_sampling_formula(self, zeta_values, model_confidence_estimation):
        state_number = self._env.get_state_number()
        action_number = self._env.get_action_number()
        if self._sampling_strategy == 0:
            # Uniform sampling
            return np.ones((state_number, action_number)) / (state_number * action_number)
        elif self._sampling_strategy == 1:
            # Formula 1 (Used by default by SimuDICE)
            softmax_zeta_values = np.exp(zeta_values) / np.sum(np.exp(zeta_values))
            shaped_softmax_zeta_values = np.reshape(softmax_zeta_values, (state_number, action_number))
            return model_confidence_estimation / (1 - self._policy.get_epsilon()) + shaped_softmax_zeta_values / self._lambda
        elif self._sampling_strategy == 2:
            # Formula 2
            model_confidence_estimation_flattened = np.reshape(model_confidence_estimation, (state_number * action_number))
            tmp_probability = model_confidence_estimation_flattened + zeta_values / self._lambda
            softmax = np.exp(tmp_probability) / np.sum(np.exp(tmp_probability))
            return np.reshape(softmax, (state_number, action_number))
        elif self._sampling_strategy == 3:
            # Formula 3
            softmax_zeta_values = np.exp(zeta_values) / np.sum(np.exp(zeta_values))
            return np.reshape(softmax_zeta_values, (state_number, action_number))
        
