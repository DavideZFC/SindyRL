import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.environments.PendulumSimple import PendulumSimple
from sklearn.linear_model import Lasso
import numpy as np

discretize = 10
numel = 10000

env = gym.make('Pendulum-v1')
# env = PendulumSimple() non usare, ha qualcosa che non va

LRB = Linear_replay_buffer(basis='poly', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
LRB.linear_converter()

episodes = 10
state = env.reset()
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        LRB.memorize(state,action,new_state,reward)
        state = new_state

LRB.linear_converter()

X, y = LRB.get_SINDY_data()
numel = 15
eps = 10**(-8)
alpha_routine = [2**(-i) for i in range(numel)]
for alpha in alpha_routine:
    model = Lasso(alpha=alpha)
    model.fit(X,y)
    MSE = np.mean((model.predict(X)-y)**2)
    valid = np.sum(1*(model.coef_**2>eps))
    print('alpha = {}, MSE = {}, valid = {}, total = {}'.format(alpha,MSE,valid, model.coef_.shape))