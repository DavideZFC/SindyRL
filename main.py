import gym
# from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.actors.SINDy import SINDy
from classes.environments.CartPoleContinuous import ContinuousCartPoleEnv
from classes.environments.Pendulum import Pendulum
from functions.misc.make_experiment import make_experiment

discretize = 10
numel = 10000

# env = gym.make('LunarLanderContinuous-v2')= Pendulum()#
env = ContinuousCartPoleEnv()

agent1 = SINDy(basis='sincos', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, hor=2, trials=20, alpha=0.001, lasso=True)#, lasso=True
agent2 = SINDy(basis='poly', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, hor=2, trials=20, alpha=0.001, lasso=True)
agent3 = SINDy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, hor=2, trials=20, alpha=0.001, lasso=True)

make_experiment([agent1, agent2, agent3], env, seeds=20, K=10, labels=['fourier', 'poly', 'legendre'])
