from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from sklearn.linear_model import Lasso
import numpy as np

class SINDy(Linear_replay_buffer):

    def __init__(self, basis, approx_degree, state_space_dim, action_space, numel, discretize=200):
        super().__init__(basis, approx_degree, state_space_dim, action_space, numel, discretize)

    def compute_models(self):
        X, y = self.get_SINDY_model_data()
        self.p_model = self.train_model(X,y)

        X, y = self.get_SINDY_reward_data()
        self.r_model = self.train_model(X,y)

    def train_model(self, X, y, alpha=1e-4):
        model = Lasso(alpha=alpha)
        model.fit(X,y)
        MSE = np.mean((model.predict(X)-y)**2)
        valid = np.sum(1*(model.coef_**2>0))
        print('MSE = {}, valid = {}, total = {}'.format(MSE, valid, model.coef_.shape))
        return model
    
    def eval_action_queue(self, state, action_queue):
        h = action_queue.shape[0]
        total = 0
        for i in range(h):
            action = action_queue[i,:]
            state_action = self.build_state_action_feature_map(state,action)
            new_state = self.p_model.predict(state_action)
            total += self.r_model.predict(state_action)
            state = new_state
        return total
    
    def planner(self, state, h, N):
        # generate random action queues
        action_queues = np.zeros((N,h,self.action_space_dim))
        for n in range(N):
            for i in range(h):
                action_queues[n,i,:] = self.action_space.sample()
        for n in range(N):
            print(self.eval_action_queue(state, action_queues[n,:,:]))






