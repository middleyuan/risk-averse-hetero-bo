import os
import time
import torch
from torch import Tensor
import math
from filelock import FileLock
from .benchmarks import BenchmarkBase
from botorch.utils.sampling import draw_sobol_samples
import yaml
import subprocess
from multiprocessing import Process, Manager
import numpy as np

# define the categorical choice or real interval for each hyperparameter
PPO_dict = {
    'categorical': {
        'N_WARMUP_STEPS_CRITIC': [0, 10, 100, 1000],
        'N_WARMUP_STEPS_ACTOR': [0, 10, 100, 1000],
        'ON_POLICY_RUNNER_STEPS_PER_ENV': [256, 512, 1024, 2048],
        'BATCH_SIZE': [32, 64, 128, 256],
        'TOTAL_STEP_LIMIT': [100000, 150000, 300000, 350000, 400000],
        'EPISODE_STEP_LIMIT': [100, 200, 300, 400, 500],
        'ACTOR_HIDDEN_DIM': [8, 16, 32, 64, 128],
        'CRITIC_HIDDEN_DIM': [8, 16, 32, 64, 128],
        'ACTOR_NUM_LAYERS': [2, 3, 4],
        'CRITIC_NUM_LAYERS': [2, 3, 4],
        'ACTOR_ACTIVATION_FUNCTION': [1, 2, 3, 4, 5], # # RELU 1,GELU 2,TANH 3,FAST_TANH 4,SIGMOID 5
        'CRITIC_ACTIVATION_FUNCTION': [1, 2, 3, 4, 5], # # RELU 1,GELU 2,TANH 3,FAST_TANH 4,SIGMOID 5
        'N_EPOCHS': [1, 2, 3, 4, 5],
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'GAMMA': [0.9, 0.9999],
        'LAMBDA': [0.8, 1.0],
        'EPSILON_CLIP': [0.1, 0.4],
        'INITIAL_ACTION_STD': [0.2, 0.8],
        'ACTION_ENTROPY_COEFFICIENT': [0.001, 0.05],
        'POLICY_KL_EPSILON': [math.log(0.00001, 10), math.log(0.1, 10)],
        'ACTOR_ALPHA': [math.log(1e-5, 10), math.log(0.5, 10)],
        'CRITIC_ALPHA': [math.log(1e-5, 10), math.log(0.5, 10)],
    }
}

HYPERPARAMS_DICT = {
    'ppo': PPO_dict,
}


class Pendulum(BenchmarkBase):
    """
    Two-dimentional Pendulum environment.
    """

    def __init__(self, algo, output_dir, BO_algo, max_processes, rescale=True):
        super(Pendulum, self).__init__()
        self.algo = algo
        self.bo_algo = BO_algo
        self.output_dir = output_dir
        self.max_processes = max_processes
        self.cur_dir = os.getcwd()
        # make self.output_dir
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.output_dir, BO_algo))
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        print(f'output_dir: {self.path}')
        self.rescale = rescale

    def evaluate(self, x: Tensor, repeat_eval, return_eval_times=True, mode='train'):
        x_copy, indx = self.transform_search_space(x)
        hp_configs = self._to_hp_configs(x_copy, indx)

        def objective(seed, actor_alpha, critic_alpha, result_metrics, i):
            args = ['./my_pendulum']
            args.append(f'--seed={seed}')
            actor_alpha = math.pow(10, actor_alpha)
            critic_alpha = math.pow(10, critic_alpha)
            args.append(f'--actor_alpha={actor_alpha}')
            args.append(f'--critic_alpha={critic_alpha}')
            result = subprocess.run(args, capture_output=True, text=True)
            final_reward = float(result.stdout.strip().split('\n')[-2].split(' ')[-1])
            initial_reward = float(result.stdout.strip().split('\n')[0].split(' ')[-1])
            # check if final_reward is a valid float number
            if math.isnan(final_reward) or math.isinf(final_reward):
                if math.isnan(initial_reward) or math.isinf(initial_reward):
                    print(f'Invalid reward: {final_reward}, replace with {initial_reward}')
                    final_reward = initial_reward
                else:
                    print(f'Invalid reward: {final_reward}, replace with -1000')
                    final_reward = -1000
            result_metrics[i] = final_reward
            # print reward given seed and alphas
            print(f'bo_algo={self.bo_algo}, seed={seed}, actor_alpha={actor_alpha}, critic_alpha={critic_alpha}, reward={final_reward}')

        aggregate_results = []
        for hps in hp_configs:
            os.chdir(f'{self.cur_dir}')
            processes = []
            result_metrics = Manager().list([None] * repeat_eval)

            # configure cmake
            cmake_args = ['cmake']
            for key, value in hps.items():
                if key == 'ACTOR_ALPHA' or key == 'CRITIC_ALPHA':
                    continue
                if key == 'POLICY_KL_EPSILON':
                    value = math.pow(10, value)
                cmake_args.append(f'-DHYPERPARAM_{key}={value}')
            cmake_args += ['-DCMAKE_BUILD_TYPE=Release', '../..']
            if not os.path.exists(f'./example/build/{self.bo_algo}'):
                os.makedirs(f'./example/build/{self.bo_algo}')
            os.chdir(f'./example/build/{self.bo_algo}')
            cmake_process = subprocess.run(cmake_args, capture_output=True, text=True)
            if cmake_process.returncode != 0:
                print("CMake configuration failed:")
                print(cmake_process.stdout)
                print(cmake_process.stderr)
                return
            
            # build
            build_args = ['cmake', '--build', '.']
            build_process = subprocess.run(build_args, capture_output=True, text=True)
            if build_process.returncode != 0:
                print("Build failed:")
                print(build_process.stdout)
                print(build_process.stderr)
                return
            
            for i in range(repeat_eval):
                if mode == 'train':
                    seed = np.random.randint(0, 10000)
                else:
                    seed = np.random.randint(10000, 20000)
                p = Process(target=objective, args=(seed, hps['ACTOR_ALPHA'], hps['CRITIC_ALPHA'], result_metrics, i))
                processes.append(p)

            step = 0
            while step < len(processes):
                begin = int(step * self.max_processes)
                end = min(begin + self.max_processes, len(processes))
                for p in processes[begin:end]:
                    p.start()
                for p in processes[begin:end]:
                    p.join()
                step += 1
             # Collect results from all processes
            results = []
            for i in range(repeat_eval):
                results.append(result_metrics[i])
            aggregate_results.append(results)
        
        y = torch.tensor(aggregate_results, dtype=float).reshape(x_copy.shape[0], -1)
        if mode == 'train':
            if self.rescale == True:
                # y = -torch.log(y)
                y = -torch.log(-y)
        
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
                        hp_configs[j][k] = v[indx[j, i].item()]
                    i += 1
            elif key == 'float':
                for k, v in value.items():
                    for j in range(batch_size):
                        hp_configs[j][k] = x_copy[j, i].item()
                    i += 1
        return hp_configs
