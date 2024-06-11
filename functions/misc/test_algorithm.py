import numpy as np
from copy import deepcopy

def test_algorithm(agent0, env, seeds, K=30, first_seed=1):
    '''
    Test a given policy on an environment and returns the regret estimated over some different random seeds

    Parameters:
        agent (specific class): policy to be tested
        env (class environment): environment over which to test the policy
        T (int): time horizon
        seeds (int): how many random seeds to use
        first seed (int): first seed to use

    Returns:
        regret_matrix (array): matrix having as rows the value of the cumulative regret for one random seed
    '''
    return_matrix = np.zeros((seeds, K))
    np.random.seed(first_seed)

    agent = deepcopy(agent0)

    for seed in range(seeds):

        for k in range(K):

            state = env.reset()[0]
            done = False
            ret = 0
            print('ep = {}'.format(k))

            while not done:
                # action = env.action_space.sample()
                action = agent.choose_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)

                ret += reward

                done = terminated or truncated

                agent.memorize(state, action, next_state, reward)
                state = next_state

            agent.train()
            return_matrix[seed, k] = ret

        
        agent.reset()
            
    return return_matrix

