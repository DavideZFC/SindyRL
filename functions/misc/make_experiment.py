import os
import datetime
from joblib import Parallel, delayed
import time
import json
import numpy as np
from functions.misc.test_algorithm import test_algorithm
from functions.misc.plot_data import plot_data
from functions.misc.confidence_bounds import bootstrap_ci
import matplotlib.pyplot as plt

def make_experiment(policies, env, seeds, K, labels, exp_name=''):
    '''
    Performs a RL experiment, estimating the reward curve and saving the data in a given folder

    Parameters:
        policies (list): list of policies to be tested
        env (class environment): environment over which to test the policies
        T (int): time horizon
        seeds (int): how many random seed to use in the experiment
        labels (list): list with the same length of policies giving a name to each one
        exp_name (string): string to be added to the filder created to same the data    
    '''

    # create folder
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
    dir = 'results/'+'_'+tail+exp_name

    os.mkdir(dir)
    dir = dir+'/'

    # in this dictionary, we store the running times of the algorithms
    running_times = {}
    par = {}
    
    for i in range(len(policies)):

        ####################################
        # actual algorithm simulation

        # evaluate running time of the algorithm
        t0 = time.time()

        # test the algorithm

        results = Parallel(n_jobs=seeds)(delayed(test_algorithm)(policies[i], env, seeds=1, K=K, first_seed=seed) for seed in range(seeds))

        # store time
        t1 = time.time()
        running_times[labels[i]] = t1 - t0
        par[labels[i]+'_trials'] = policies[i].trials
        par[labels[i]+'_alpha'] = policies[i].alpha
        par[labels[i]+'_lasso'] = policies[i].lasso
        
        print(labels[i] + ' finished')

        ####################################
        # part to save data

        results = np.concatenate(results, axis=0)

        # make nonparametric confidence intervals
        low, high = bootstrap_ci(results)

        # make plot
        # plot_data(np.arange(0,env.time_horizon*K), low, high, col='C{}'.format(i), label=labels[i])

        # save data in given folder
        np.save(dir+labels[i], results)

        # make nonparametric confidence intervals

        # make plot
        plot_data(np.arange(0,len(low)), low, high, col='C{}'.format(i+1), label=labels[i]+' smooth')

    with open(dir+"running_times.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(running_times, f)

    with open(dir+"params.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(par, f)
    
    plt.legend()
    plt.title('Reward curves')
    plt.savefig(dir+'reward_plot.pdf')