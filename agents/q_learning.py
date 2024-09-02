import numpy as np
from agents.agent import Agent

class QLearningAgent(Agent): 
    def __init__(self, env, policy, configuration):
        self._env = env
        self._policy = policy
        
        self._alpha = configuration.get('alpha', 0.1)
        self._gamma = configuration.get('gamma', 0.99)
        
        self.q_values = np.zeros((env.get_state_number(), env.get_action_number()))
        
    def set_policy(self, policy):
        self._policy = policy
        
    def online_learn(self, episodes_number, max_environment_steps, debug=False):
        rewards = []
        
        for episode in range(episodes_number):
            state, _ = self._env.reset_environment()
            done = False
            environment_step = 0
            while not done:
                action = self._policy.select_action(self.q_values[state])
                next_state, reward, done = self._env.step(action)
                self.q_values[state, action] += self._alpha * (reward + self._gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])
                state = next_state
                
                environment_step += 1
                if environment_step == max_environment_steps:
                    break
                
            rewards.append(reward)
                
            if debug and episode % 100 == 0:
                print('Finished online play QLearningAgent for episode: {0}'.format(episode + 1))
        
        print('Finished training with average reward: {0}'.format(np.mean(rewards)))
            
    def offline_learn(self, offline_data, debug=False):
        trajectories = [item for sublist in offline_data for item in sublist]
        np.random.shuffle(trajectories)
        
        trajectory_count = 0
        for (state, action, reward, next_state) in trajectories:
            self.q_values[state, action] += self._alpha * (reward + self._gamma * np.max(self.q_values[next_state]) - self.q_values[state, action])
            
            if debug and trajectory_count % 100 == 0:
                print('Finished training QLearningAgent for trajectory: {0}'.format(len(trajectory_count)))
                trajectory_count += 1
        
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
                
            rewards.append(reward)
            trajectories.append(episode_trajectory)
                
            if debug and episode % 100 == 0:
                print('Finished online training QLearningAgent for episode: {0}'.format(episode + 1))
                
        print('Finished playing with last reward: {0}'.format(rewards[-1]))
        print('Finished playing with average reward: {0}'.format(np.mean(rewards)))
        if save_trajectories:
            return trajectories
        return None
    
    def get_policy_data(self):
        return {'q_values': self.q_values, 'epsilon': self._policy.get_epsilon()}
