import argparse
import os 
import pickle

from environments import create_environment
from policies.epsilon_greedy import EpsilonGreedyPolicy
from agents.q_learning import QLearningAgent

class ArgumentParser: 
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Behavioral Data Generator using Epsilon-Greedy Q-Learning')
        self._parser.add_argument('--env', type=str, default='Taxi', choices=['Taxi', 'CliffWalking', 'FrozenLake'], help='Environment to use')
        self._parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for epsilon-greedy policy')
        self._parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
        self._parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
        self._parser.add_argument('--train_episodes', type=int, default=1000, help='Number of episodes')
        self._parser.add_argument('--play_episodes', type=int, default=100, help='Number of episodes')
        self._parser.add_argument('--max_environment_steps', type=int, default=400, help='Maximum number of steps in the environment')
        self._parser.add_argument('--save_trajectories', action='store_true', help='Save trajectories')
        self._parser.add_argument('--save_dir', type=str, default='./datasets', help='Directory to save trajectories')
        self._parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
        self._parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    def parse(self):
        self._args = self._parser.parse_args()
        return self._args
    
    def print_args(self):
        print('-' * 50)
        print('Behavioral Data Generator using Epsilon-Greedy Q-Learning Configuration')
        print('Environment: {0}'.format(self._args.env))
        print('Epsilon: {0}'.format(self._args.epsilon))
        print('Alpha: {0}'.format(self._args.alpha))
        print('Gamma: {0}'.format(self._args.gamma))
        print('Train Episodes: {0}'.format(self._args.train_episodes))
        print('Play Episodes: {0}'.format(self._args.play_episodes))
        print('Max Environment Steps: {0}'.format(self._args.max_environment_steps))
        print('Save Trajectories: {0}'.format(self._args.save_trajectories))
        print('Save Directory: {0}'.format(self._args.save_dir))
        print('Seed: {0}'.format(self._args.seed))
        print('Debug: {0}'.format(self._args.debug))
        print('-' * 50)
        
def main():
    parser = ArgumentParser()
    args = parser.parse()
    parser.print_args()
    
    env = create_environment(args.env)
    
    agent = QLearningAgent(env, EpsilonGreedyPolicy(0.1), {'alpha': args.alpha, 'gamma': args.gamma})
    agent.online_learn(args.train_episodes, args.max_environment_steps, args.seed, args.debug)
    
    collection_policy = EpsilonGreedyPolicy(args.epsilon)
    agent.set_policy(collection_policy)
    
    mean_values, trajectories = agent.play(args.play_episodes, args.max_environment_steps, args.save_trajectories, args.debug, args.seed)
    policy_data = agent.get_policy_data()
    
    print('Mean reward: {0}'.format(mean_values))
    
    if args.save_trajectories and trajectories is not None:
        offline_dataset_name = '{0}_{1}_{2}_{3}_{4}_behavioral_data.pkl'.format(args.env, args.epsilon, args.train_episodes, args.play_episodes, args.max_environment_steps)
        offline_dataset_path = os.path.join(args.save_dir, offline_dataset_name)
        
        with open(offline_dataset_path, 'wb') as f:
            pickle.dump({'trajectories': trajectories, 'policy_data': policy_data}, f)
            print('Behavioral data saved at: {0}'.format(offline_dataset_path))
    else: 
        print('No data saved')
        
if __name__ == '__main__':
    main()