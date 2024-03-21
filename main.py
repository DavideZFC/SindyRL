import gym
# from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.actors.SINDy import SINDy
from classes.environments.CartPoleContinuous import ContinuousCartPoleEnv
from functions.misc.make_experiment import make_experiment

discretize = 10
numel = 10000

# env = gym.make('LunarLanderContinuous-v2')
env = ContinuousCartPoleEnv()

agent1 = SINDy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, hor=2, trials=5)
agent2 = SINDy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, hor=2, trials=10)
agent3 = SINDy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, hor=2, trials=20)

make_experiment([agent1, agent2, agent3], env, seeds=4, K=20, labels=['sindyH1', 'sindyH2', 'sindyH4'])
