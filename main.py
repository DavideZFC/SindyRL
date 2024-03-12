import gym
from classes.Linear_replay_buffer import Linear_replay_buffer

discretize = 10
numel = 1000

env = gym.make('Pendulum-v1')

LRB = Linear_replay_buffer(basis='legendre', approx_degree=4, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
LRB.linear_converter()

print(LRB.full_feature_map.shape)