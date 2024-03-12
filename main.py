import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.environments.PendulumSimple import PendulumSimple

discretize = 10
numel = 10000

env = PendulumSimple()

LRB = Linear_replay_buffer(basis='legendre', approx_degree=4, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize)
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
print(LRB.full_feature_map.shape)
        