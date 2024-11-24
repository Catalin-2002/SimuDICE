from main import run_simu_dice

config = {
    'dataset_path': './datasets/FrozenLake_0.7_10000_510_100_behavioral_data.pkl',
    'environment_name': 'Taxi',
    'alpha': 0.1,
    'gamma': 0.99,
    'planning_steps': 10,
    'iterations': 1,
    'lamba': 1000,
    'sampling_strategy': 1,
    'play_episodes': 500,
    'max_environment_steps': 100
}

max_training_episodes = list(range(25, 501, 50)) + [500]
print('Configuration:', config)

for max_training_episodes in max_training_episodes:
    print('Running SimuDICE for {0} episodes'.format(max_training_episodes))
    
    print('Q-learning')
    config['planning_steps'] = 0
    config['sampling_strategy'] = 0
    run_simu_dice(config['dataset_path'], max_training_episodes, config['environment_name'], config['alpha'], config['gamma'],
            config['planning_steps'], config['iterations'], config['lamba'], config['sampling_strategy'],
            config['play_episodes'], config['max_environment_steps'], debug=False)
    
    print('Offline Dyna-Q 10 PS')
    config['planning_steps'] = 10
    config['sampling_strategy'] = 0
    run_simu_dice(config['dataset_path'], max_training_episodes, config['environment_name'], config['alpha'], config['gamma'],
            config['planning_steps'], config['iterations'], config['lamba'], config['sampling_strategy'],
            config['play_episodes'], config['max_environment_steps'], debug=False)
    
    print('Offline Dyna-Q 20 PS')
    config['planning_steps'] = 20
    config['sampling_strategy'] = 0
    run_simu_dice(config['dataset_path'], max_training_episodes, config['environment_name'], config['alpha'], config['gamma'],
            config['planning_steps'], config['iterations'], config['lamba'], config['sampling_strategy'],
            config['play_episodes'], config['max_environment_steps'], debug=False)
    
    print('SimuDICE 10 PS')
    config['planning_steps'] = 10
    config['sampling_strategy'] = 1
    run_simu_dice(config['dataset_path'], max_training_episodes, config['environment_name'], config['alpha'], config['gamma'],
            config['planning_steps'], config['iterations'], config['lamba'], config['sampling_strategy'],
            config['play_episodes'], config['max_environment_steps'], debug=False)
    
    print('SimuDICE 20 PS')
    config['planning_steps'] = 20
    config['sampling_strategy'] = 1
    run_simu_dice(config['dataset_path'], max_training_episodes, config['environment_name'], config['alpha'], config['gamma'],
            config['planning_steps'], config['iterations'], config['lamba'], config['sampling_strategy'],
            config['play_episodes'], config['max_environment_steps'], debug=False)