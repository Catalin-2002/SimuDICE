import numpy as np
from agents.agent import Agent

from utils.dual_dice import DualDICE

class SimuDICE(Agent): 
    def __init__(self, env, policy, world_model, configuration):
        self._env = env
        self._policy = policy
        self._world_model = world_model
        
        self._alpha = configuration.get('alpha', 0.05)
        self._gamma = configuration.get('gamma', 0.99)
        self._planning_steps = configuration.get('planning_steps', 10)
        
        self._iterations = configuration.get('iterations', 1)
        self._lambda = configuration.get('lambda', 1000)
        
        self._sampling_strategy = configuration.get('sampling_strategy', 1)
        
        self.q_values = np.zeros((env.get_state_number(), env.get_action_number()))
        
    def set_policy(self, policy):
        self._policy = policy
        
    def online_learn(self, episodes_number, max_environment_steps, debug=False):
        pass
            
    def offline_learn(self, offline_data, debug=False):
        trajectories = [item for sublist in offline_data for item in sublist]
        
        np.random.seed(42)
        np.random.shuffle(trajectories)
        
        self._world_model.train(trajectories)
        
        self.q_values = np.zeros((self._env.get_state_number(), self._env.get_action_number()))
        
        for (state, action, reward, next_state) in trajectories:
            self.q_values[state, action] += self._alpha * (reward + self._gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])
        
        model_confidence_estimation = self.get_model_confidence_estimation(trajectories)
            
        planning_steps_per_iteration = self._planning_steps // self._iterations
        planning_steps_counter = 0
                
        for iteration in range(self._iterations):
            target_planning_steps = planning_steps_per_iteration if iteration != self._iterations - 1 else self._planning_steps - planning_steps_counter
            planning_steps_counter += target_planning_steps
            
            dice_estimator = DualDICE(self._env.get_state_number(), self._env.get_action_number(), self._gamma)
            zeta_values = dice_estimator.get_weight_estimates(offline_data, self.q_values, self._policy)
            
            sampling_formula = self.get_sampling_formula(zeta_values * self._lambda, model_confidence_estimation)
            self._world_model.update_sampling_probabilities(sampling_formula)
            
            num_samples = target_planning_steps * len(trajectories)
            sampled_pairs = self._world_model.batch_sample_state_actions(sample_number=num_samples)
            next_states, rewards = self._world_model.batch_predict(state_action_pairs=sampled_pairs)
            states = sampled_pairs[:, 0]
            actions = sampled_pairs[:, 1]
            
            for i in range(num_samples):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                self.q_values[state, action] += self._alpha * (reward + self._gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])

            if debug:
                print('Finished SimuDICE training iteration: {0}'.format(iteration + 1))
                
    def play(self, episodes_number, max_environment_steps, save_trajectories=False, debug=False, seed=42):
        rewards = []
        trajectories = []
        
        seed_generator = np.random.RandomState(seed)
            
        for episode in range(episodes_number):
            state = self._env.reset_environment(seed_generator.randint(0, 1000))
            done = False
            environment_step = 0
            episode_trajectory = []
            
            episode_reward = 0
            
            while not done:
                action = self._policy.select_action(self.q_values[state])
                next_state, reward, done = self._env.step(action)
                
                episode_reward += reward
                
                if save_trajectories:
                    episode_trajectory.append((state, action, reward, next_state))

                state = next_state
                environment_step += 1
                
                if environment_step == max_environment_steps or done:
                    break
            
            rewards.append(episode_reward / environment_step)
            
            if save_trajectories:
                trajectories.append(episode_trajectory)
                
            if debug and episode % 100 == 0:
                print('Finished online play SimuDICE for episode: {0}'.format(episode + 1))
                
        return np.mean(rewards), trajectories

    
    def get_policy_data(self):
        return {'q_values': self.q_values, 'epsilon': self._policy.get_epsilon()}
    
    def get_model_confidence_estimation(self, offline_data):
        state_action_confidence = np.zeros((self._env.get_state_number(), self._env.get_action_number()))
        for (state, action, _, _) in offline_data:
            state_action_confidence[state, action] += 1
            
        state_action_confidence /= np.sum(state_action_confidence)
        return state_action_confidence
    
    def get_sampling_formula(self, zeta_values, model_confidence_estimation):
        state_number = self._env.get_state_number()
        action_number = self._env.get_action_number()
    
        if self._sampling_strategy == 0:
            # Uniform sampling
            return np.ones((state_number, action_number)) / (state_number * action_number)
        elif self._sampling_strategy == 1:
            # Formula 1 (SimuDICE)
            maximum_zeta_values = np.max(zeta_values)
            softmax_zeta_values = np.exp(zeta_values - maximum_zeta_values) / np.sum(np.exp(zeta_values - maximum_zeta_values))
            shaped_softmax_zeta_values = np.reshape(softmax_zeta_values, (state_number, action_number))
            
            probs = model_confidence_estimation / (1 - self._policy.get_epsilon()) + shaped_softmax_zeta_values / self._lambda
            
            return probs / np.sum(probs)
        elif self._sampling_strategy == 2:
            # Formula 2
            maximum_zeta_values = np.max(zeta_values)
            softmax_zeta_values = np.exp(zeta_values - maximum_zeta_values) / np.sum(np.exp(zeta_values - maximum_zeta_values))
            shaped_softmax_zeta_values = np.reshape(softmax_zeta_values, (state_number, action_number))
            
            probs = model_confidence_estimation / (1 - self._policy.get_epsilon()) - shaped_softmax_zeta_values / self._lambda
            
            min_prob = np.min(probs)
            if min_prob < 0:
                probs += np.abs(min_prob)
                
            return probs / np.sum(probs)
        elif self._sampling_strategy == 3:
            # Formula 3
            maximum_zeta_values = np.max(zeta_values)
            softmax_zeta_values = np.exp(zeta_values - maximum_zeta_values) / np.sum(np.exp(zeta_values - maximum_zeta_values))
            shaped_softmax_zeta_values = np.reshape(softmax_zeta_values, (state_number, action_number))
            
            probs = np.ones((state_number, action_number)) / (state_number * action_number) + shaped_softmax_zeta_values / self._lambda
            
            return probs / np.sum(probs)
            
        
