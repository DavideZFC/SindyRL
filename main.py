import gym
# from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.actors.SINDy import SINDy
from classes.environments.CartPoleContinuous import ContinuousCartPoleEnv
from sklearn.linear_model import Lasso
import numpy as np

discretize = 10
numel = 10000

# env = gym.make('LunarLanderContinuous-v2')
env = ContinuousCartPoleEnv()

LRB = SINDy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
LRB.linear_converter()

transitions = 10000
state = env.reset()
t = 0
h = 5
N = 20
lim = 100
while t<transitions:
    state, _ = env.reset()
    done = False
    ret = 0
    while not done:
        if t>lim:
            action = LRB.planner(state,h=5,N=10)[0]
        else:
            action = env.action_space.sample()
        new_state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        LRB.memorize(state,action,new_state,reward)
        state = new_state
        t += 1

        ret += reward
    print('#\n siamo a t={} \n #'.format(t))
    print('episodic return {}'.format(ret))

    LRB.linear_converter()
    LRB.compute_models()

state, _ = env.reset()
print(LRB.planner(state,h=5,N=10))