import gym
# from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.actors.SINDy import SINDy
from classes.environments.CartPoleContinuous import ContinuousCartPoleEnv
from functions.misc.make_experiment import make_experiment

discretize = 10
numel = 10000

# env = gym.make('LunarLanderContinuous-v2')
env = ContinuousCartPoleEnv()

LRB2 = SINDy(basis='legendre', approx_degree=2, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
LRB3 = SINDy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
LRB4 = SINDy(basis='legendre', approx_degree=4, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)

make_experiment([LRB2, LRB3, LRB4], env, seeds=4, K=20, labels=['sindy2', 'sindy3', 'sindy4'])
'''
transitions = 10000
state = env.reset()
t = 0
h = 2
N = 10
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
'''