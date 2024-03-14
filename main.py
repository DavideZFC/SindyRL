import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.environments.CartPoleContinuous import ContinuousCartPoleEnv
from sklearn.linear_model import Lasso
import numpy as np

discretize = 10
numel = 10000

# env = gym.make('LunarLanderContinuous-v2')
env = ContinuousCartPoleEnv()

LRB = Linear_replay_buffer(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
LRB.linear_converter()

transitions = 1000
state = env.reset()
t = 0
while t<transitions:
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        LRB.memorize(state,action,new_state,reward)
        state = new_state
        t += 1

LRB.linear_converter()

X, y = LRB.get_SINDY_reward_data()
print('training with {} data'.format(X.shape[0]))
numel = 6
eps = 0#10**(-8)
alpha_routine = [2**(-2*i-5) for i in range(numel)]
for alpha in alpha_routine:
    model = Lasso(alpha=alpha)
    model.fit(X,y)
    MSE = np.mean((model.predict(X)-y)**2)
    valid = np.sum(1*(model.coef_**2>eps))
    print('alpha = {}, MSE = {}, valid = {}, total = {}'.format(alpha,MSE,valid, model.coef_.shape))