import argparse
import os 
import pickle

from environments import create_environment
from policies.epsilon_greedy import EpsilonGreedyPolicy
from agents.simu_dice import SimuDICE

from utils.tabular_world_model import TabularWorldModel

class ArgumentParser: 
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='SimuDICE: Offline Policy Optimization Through Iterative World Model Updates and DICE Estimation')
        self._parser.add_argument('--env', type=str, default='Taxi', choices=['Taxi', 'CliffWalking', 'FrozenLake'], help='Environment to use')
        self._parser.add_argument('--data_path', type=str, help='Path to offline data')
        
        self._parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
        self._parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
        self._parser.add_argument('--planning_steps', type=int, default=10, help='Number of planning steps')
        self._parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
        self._parser.add_argument('--lambda_value', type=int, default=100, help='Lambda parameter')
        self._parser.add_argument('--sampling_strategy', type=int, default=0, help='Sampling strategy')
        
        self._parser.add_argument('--play_episodes', type=int, default=100, help='Number of episodes to play')
        self._parser.add_argument('--max_environment_steps', type=int, default=100, help='Maximum number of environment steps')
        
        self._parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum number of episodes to use')
        
    def parse(self):
        self._args = self._parser.parse_args()
        return self._args
    
    def print_args(self):
        print('-' * 50)
        print('SimuDICE Configuration')
        print('Environment: {0}'.format(self._args.env))
        print('Data Path: {0}'.format(self._args.data_path))
        print('Alpha (Learning Rate): {0}'.format(self._args.alpha))
        print('Gamma (Discount Factor): {0}'.format(self._args.gamma))
        print('Planning Steps: {0}'.format(self._args.planning_steps))
        print('Iterations: {0}'.format(self._args.iterations))
        print('Lambda Parameter: {0}'.format(self._args.lambda_value))
        print('Sampling Strategy: {0}'.format(self._args.sampling_strategy))
        print('Play Episodes: {0}'.format(self._args.play_episodes))
        print('Max Environment Steps: {0}'.format(self._args.max_environment_steps))
        print('Max Episodes: {0}'.format(self._args.max_episodes))
        print('-' * 50)
        
def run_simu_dice(dataset_path, max_training_episodes, environment_name, alpha, gamma, planning_steps, iterations, lamba, sampling_strategy,
                  play_episodes, max_environment_steps, debug=False):
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)
    offline_data = data['trajectories'][:max_training_episodes]
    
    # Train agent
    env = create_environment(environment_name)
    
    # Train agent
    agent = SimuDICE(env, EpsilonGreedyPolicy(0.0), TabularWorldModel(), 
                     {'alpha': alpha, 'gamma': gamma, 'planning_steps': planning_steps, 'iterations': iterations, 'lambda': lamba, 'sampling_strategy': sampling_strategy})
    agent.offline_learn(offline_data, debug)
    
    # Play agent
    agent.set_policy(EpsilonGreedyPolicy(0.1))
    trajectories = agent.play(play_episodes, max_environment_steps, False, debug)
    
        
def main():
    parser = ArgumentParser()
    args = parser.parse()
    parser.print_args()
    
    run_simu_dice(args.data_path, args.max_episodes, args.env, args.alpha, args.gamma, args.planning_steps, args.iterations, args.lambda_value, args.sampling_strategy,
                  args.play_episodes, args.max_environment_steps, True)
        
if __name__ == '__main__':
    main()