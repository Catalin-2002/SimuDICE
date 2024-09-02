from environments.frozen_lake_env import FrozenLakeEnv
from environments.cliff_walking_env import CliffWalkingEnv
from environments.taxi_env import TaxiEnv

def create_environment(env_name):
    if env_name == 'FrozenLake':
        return FrozenLakeEnv()
    elif env_name == 'CliffWalking':
        return CliffWalkingEnv()
    elif env_name == 'Taxi':
        return TaxiEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")