import os
import time
import torch
from torch import Tensor
import math
from filelock import FileLock
from .benchmarks import BenchmarkBase
from botorch.utils.sampling import draw_sobol_samples
import yaml

# define the categorical choice or real interval for each hyperparameter
PPO_dict = {
    'categorical': {
        'hidden_dim': [8, 16, 32, 64, 128, 256, 512],
        'activation': [0, 1, 2], # ['tanh', 'relu', 'leaky_relu']
        'gamma': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        'gae_lambda': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
        'clip_param': [0.1, 0.2, 0.3, 0.4],
        'opt_epochs': [1, 5, 10, 20],
        'mini_batch_size': [32, 64, 128, 256],
        'rollout_steps': [100, 150, 200], # steps increment by rollout_steps * n_envs
        'max_env_steps': [30000, 72000, 114000, 156000, 216000],  # to make sure having the checkpoint at these steps [30000, 72000, 216000]
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'target_kl': [0.00000001, 0.8],
        'entropy_coef': [0.00000001, 0.1],
        'actor_lr': [math.log(1e-5, 10), math.log(0.5, 10)],
        'critic_lr': [math.log(1e-5, 10), math.log(0.5, 10)],
        'state_weight': [0.001, 20],
        'state_dot_weight': [0.001, 5],
        'action_weight': [0.001, 5],
    }
}
SAC_dict = {
    'categorical': {
        'hidden_dim': [32, 64, 128, 256, 512],
        'activation': [0, 1, 2], # ['tanh', 'relu', 'leaky_relu']
        'gamma': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        'train_interval': [10, 100, 1000],  # max_env_steps should be divisible by train_interval
        'train_batch_size': [32, 64, 128, 256, 512, 1024],
        'max_env_steps': [30000, 72000, 114000, 156000, 216000],  # to make sure having the checkpoint at these steps [30000, 72000, 216000]
        'warm_up_steps': [500, 1000, 2000, 4000],
        'max_buffer_size': [10000, 50000, 100000, 200000],
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'tau': [0.005, 1.0],
        'init_temperature': [0.01, 0.5],  # initial temperature for the policy
        'actor_lr': [math.log(1e-5, 10), math.log(0.5, 10)],
        'critic_lr': [math.log(1e-5, 10), math.log(0.5, 10)],
        'entropy_lr': [math.log(1e-5, 10), math.log(0.5, 10)],
        'state_weight': [0.001, 20],
        'state_dot_weight': [0.001, 5],
        'action_weight': [0.001, 5],
    }
}

GPMPC_dict = {
    'categorical': {
        'horizon': [10, 15, 20, 25, 30, 35, 40],
        'kernel': [0, 1], # ['Matern', 'RBF']
        'n_ind_points': [30, 40, 50],  # number should lower 0.8 * MIN(num_samples) if 0,2 is test_data_ratio
        'num_epochs': [4, 5, 6, 7, 8],
        'num_samples': [70, 75, 80, 85],
        'optimization_iterations': [2800, 3000, 3200],
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'learning_rate': [math.log(5e-4, 10), math.log(0.5, 10)],
        'state_weight': [0.001, 20],
        'state_dot_weight': [0.001, 5],
        'action_weight': [0.001, 5],
    }
}
HYPERPARAMS_DICT = {
    'ppo': PPO_dict,
    'sac': SAC_dict,
    'gp_mpc': GPMPC_dict,
}


class Quad2DBenchmark(BenchmarkBase):
    """
    Two-dimentional Quadrotor environment.
    """

    def __init__(self, algo, output_dir, metric, BO_algo, negate=True):
        super(Quad2DBenchmark, self).__init__()
        self.algo = algo
        self.output_dir = output_dir
        self.protocol = {'state': 'idle', 'metric': metric, 'algo': algo, 'BO_algo': BO_algo}
        # make self.output_dir
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.output_dir, BO_algo))
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # create a lock
        lock = FileLock(f'{self.path}/protocol.yaml.lock')
        with lock:
            with open(f'{self.path}/protocol.yaml', 'w')as f:
                yaml.dump(self.protocol, f, default_flow_style=False, sort_keys=False)
        # with open(f'{self.path}/tmp.yaml', 'w')as f:
        #     yaml.dump(self.protocol, f, default_flow_style=False, sort_keys=False)
        # os.rename(f'{self.path}/tmp.yaml', f'{self.path}/protocol.yaml')
        print(f'output_dir: {self.path}')
        self.negate = negate
        self.count = 0

    def evaluate(self, x: Tensor, repeat_eval, return_eval_times=True, max_processes=2, mode='train'):
        # prepare protocol
        self.count += 1
        self.protocol['raw_x'] = x.tolist()
        x_copy, indx = self.transform_search_space(x)
        hp_configs = self._to_hp_configs(x_copy, indx)
        self.protocol['state'] = 'request'
        self.protocol['hps'] = hp_configs
        self.protocol['repeat_eval'] = repeat_eval
        self.protocol['max_processes'] = max_processes
        self.protocol['mode'] = mode
        # create a lock
        lock = FileLock(f'{self.path}/protocol.yaml.lock')
        with lock:
            with open(f'{self.path}/protocol.yaml', 'w')as f:
                yaml.dump(self.protocol, f, default_flow_style=False, sort_keys=False)
        # with open(f'{self.path}/tmp.yaml', 'w')as f:
        #     yaml.dump(self.protocol, f, default_flow_style=False, sort_keys=False)
        # os.rename(f'{self.path}/tmp.yaml', f'{self.path}/protocol.yaml')
        
        while self.protocol['state'] != 'done':
            try:
                # create a lock
                lock = FileLock(f'{self.path}/protocol.yaml.lock')
                with lock:
                    with open(f'{self.path}/protocol.yaml', 'r')as f:
                        self.protocol = yaml.safe_load(f)
                # with open(f'{self.path}/protocol.yaml', 'r')as f:
                #     self.protocol = yaml.safe_load(f)
            except:
                pass
            time.sleep(.5)
            if self.protocol:
                if self.protocol['state'] == 'done':
                    y = torch.tensor(self.protocol['y'], dtype=float).reshape(x_copy.shape[0], -1)
                    if self.negate == True:
                        y = -torch.log(y)
                    self.protocol['tran_y'] = y.tolist()
                    with open(f'{self.path}/protocol_y_{self.count}.yaml', 'w')as f:
                        yaml.dump(self.protocol, f, default_flow_style=False, sort_keys=False)
            else:
                self.protocol = {'state': 'request'}
        
        return x_copy, y, repeat_eval

    def get_domain(self):
        """Get bounds based on HYPERPARAMS_DICT"""
        bound_list = []
        for key, value in HYPERPARAMS_DICT[self.algo].items():
            if key == 'categorical':
                for k, v in value.items():
                    bound_list.append([v[0], v[-1]])
            elif key == 'float':
                for k, v in value.items():
                    bound_list.append([v[0], v[1]])
        return torch.Tensor(bound_list).T

    def get_random_initial_points(self, num_points, seed):

        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).squeeze()
        tran_x, indx =self.transform_search_space(x)

        return tran_x
    
    def transform_search_space(self, x):
        
        hyperparams_dict = HYPERPARAMS_DICT[self.algo]

        # copy x
        x_copy = x.clone()
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape(1, -1)
        batch_size = x_copy.shape[0]
        # init indx the same shape as x_copy
        indx = torch.zeros_like(x_copy, dtype=torch.int64)

        # get index that nearest to the value
        for key, value in hyperparams_dict.items():
            if key == 'categorical':
                for i, (k, v) in enumerate(value.items()):
                    for j in range(batch_size):
                        indx[j, i] = torch.argmin(torch.abs(torch.tensor(v) - x_copy[j, i]))
                        x_copy[j, i] = v[indx[j, i].item()]
            elif key == 'float':
                continue
        # covert shape of x_copy to the shape of x
        # x_copy = x_copy.reshape(x.shape)
        return x_copy, indx
    
    def _to_hp_configs(self, x, indx):
        x_copy = x.clone()
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape(1, -1)
        batch_size = x_copy.shape[0]
        hyperparams_dict = HYPERPARAMS_DICT[self.algo]
        hp_configs = [{} for _ in range(batch_size)]
        i = 0
        for key, value in hyperparams_dict.items():
            if key == 'categorical':
                for k, v in value.items():
                    for j in range(batch_size):
                        if k == 'activation':
                            hp_configs[j][k] = ['tanh', 'relu', 'leaky_relu'][indx[j, i].item()]
                        elif k == 'kernel':
                            hp_configs[j][k] = ['Matern', 'RBF'][indx[j, i].item()]
                        else:
                            hp_configs[j][k] = v[indx[j, i].item()]
                    i += 1
            elif key == 'float':
                for k, v in value.items():
                    for j in range(batch_size):
                        hp_configs[j][k] = x_copy[j, i].item()
                    i += 1
        return hp_configs

    def close(self):
        protocol = {'state': 'end'}
        # create a lock
        lock = FileLock(f'{self.path}/protocol.yaml.lock')
        with lock:
            with open(f'{self.path}/protocol.yaml', 'w')as f:
                yaml.dump(protocol, f, default_flow_style=False, sort_keys=False)
        # with open(f'{self.path}/tmp.yaml', 'w')as f:
        #     yaml.dump(protocol, f, default_flow_style=False, sort_keys=False)
        # os.rename(f'{self.path}/tmp.yaml', f'{self.path}/protocol.yaml')
        return
