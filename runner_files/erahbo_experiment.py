import os
import pickle
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import time
import torch
import datetime
from rahbo.test_functions.quadrotor_2D import Quad2DBenchmark
from rahbo.acquisition.acquisition import RiskAverseUpperConfidenceBound, LowerConfidenceBound
from rahbo.optimization.bo_step import bo_step_risk_averse, bo_step_adaptive_risk_averse
from rahbo.optimization.bo_step import bo_step
from rahbo.optimization.bo_loop import bo_loop_learn_rho, evaluate_rho_mean
from rahbo.test_functions.benchmark_factory import build_benchmark
from runner_utils.preprocess_config import parse_config
from runner_utils.postprocess_results import dump_exp_results
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from copy import deepcopy

import argparse

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Start erahbo benchmarking')
parser.add_argument('--algo', type=str, default='ppo',
                    help='algorithm to run')
parser.add_argument('--BO_algo', type=str, default='gpucb')
parser.add_argument('--gamma', type=int, default=1,
                    help='risk-tolerance constant for rahbo (corresponds to \alpha in the RAHBO paper)')
parser.add_argument('--seed', type=int, default=24,
                    help='random seed for initial evaluations generation')
parser.add_argument('--n_budget', type=int, default=5,
                    help='number of BO iterations')
parser.add_argument('--repeat_eval', type=int, default=2,
                    help='number of evaluations at the same point')
parser.add_argument('--beta', type=int, default=2,
                    help='hyperparameter for UCB acquisition function')
parser.add_argument('--n_bo_restarts', type=int, default=1,
                    help='number of BO restarts')
parser.add_argument('--n_initial', type=int, default=2,
                    help='number of initial points for BO')
parser.add_argument('--min_repeat_eval', type=int, default=2,
                    help='minimum number of evaluations at the same point')
parser.add_argument('--output_dir', type=str, default='../../../examples/hpo/quadrotor_2D_attitude',
                    help='output directory relative to the path to quadrotor_2D.py')
parser.add_argument('--metric', type=str, default='average_rmse')
parser.add_argument('--max_processes', type=int, default=2)


def main(args):
    benchmark = Quad2DBenchmark(args.algo, args.output_dir, args.metric, args.BO_algo, args.max_processes)
    # print args
    print(args)
    objective = benchmark.evaluate
    bounds = benchmark.get_domain()

    results_all = []
    for restart in range(args.n_bo_restarts):
        
        results = {}
        idxs = []
        
        x = benchmark.get_random_initial_points(num_points=args.n_initial, seed=args.seed + restart)
        _, y, _ = objective(x, repeat_eval=args.repeat_eval)
        yvar = y.var(dim=1, keepdim=True)
        ymean = y.mean(dim=1, keepdim=True)
        
        inputs = x
        input_queries = torch.cat([torch.cat([input_x.reshape(1, -1)] * args.repeat_eval) for input_x in inputs])
        scores = ymean
        scores_var = yvar
        gps_state_dict = [None]*len(inputs)
        gps_var_state_dict = [None]*len(inputs)
        state_dict = None
        current_best = 0

        best_observed = []
        best_observed.append(scores.max())
        
        total_iters = args.n_budget
        for iteration in range(1, total_iters + 1):
            if args.BO_algo == 'gpucb':
                GP = FixedNoiseGP
                acquisition = lambda gp: UpperConfidenceBound(gp, beta=args.beta)


                inputs, scores, gp, scores_var, input_queries, eval_times = bo_step(inputs, scores, input_queries, objective, bounds,
                                                                                    args.repeat_eval, GP=GP, acquisition=acquisition, q=1,
                                                                                    state_dict=state_dict, input_query=True,
                                                                                    train_Yvar = scores_var)
                lcb_function = LowerConfidenceBound(gp, args.beta, maximize=False)
                # means = lcb_function.forward(test_x.reshape((len(test_x), 1, -1)), posterior_mean=True)                                                                
                # idxs.extend([int(means.argmax())] * eval_times)
                lcbs = lcb_function.forward(inputs.reshape((len(inputs), 1, -1)))                                                                
                idxs.extend([int(lcbs.argmax())] * eval_times)
                
                state_dict = gp.state_dict()
                gps_state_dict.append(gp.state_dict())
                best_observed.append(scores.max())
            elif args.BO_algo == 'rahbo':
                GP = FixedNoiseGP
                acquisition = lambda gp, gp_var: RiskAverseUpperConfidenceBound(gp, gp_var,
                                                                                beta=args.beta, 
                                                                                beta_varproxi=args.beta,
                                                                                gamma=args.gamma
                                                                                )

                inputs, scores, gp, scores_var, gp_var, input_queries, eval_times = bo_step_risk_averse(inputs, scores, input_queries, objective, bounds,
                                                                                            args.repeat_eval, GP=GP, acquisition=acquisition, q=1,
                                                                                            state_dict=state_dict, input_query=True,
                                                                                            train_Yvar=scores_var)
                ralcb_function = RiskAverseUpperConfidenceBound(gp, gp_var, args.beta, args.beta, args.gamma, maximize=False)
                lcbs = ralcb_function.forward(inputs.reshape((len(inputs), 1, -1)))
                idxs.extend([int(lcbs.argmax())] * eval_times)
                # reporting, _ = optimize_acqf(ralcb_function, bounds=bounds, q=1, num_restarts=1, raw_samples=1000)
                # if reportings is None:
                #     reportings = torch.cat([reporting] * eval_times)
                # else:
                #     reportings = torch.cat([reportings, torch.cat([reporting] * eval_times)])
                        
                state_dict = gp.state_dict()
                gps_state_dict.append(state_dict)
                gps_var_state_dict.append(gp_var.state_dict())
                best_observed.append(scores.max())
            elif args.BO_algo == 'erahbo':
                GP = FixedNoiseGP
                acquisition = lambda gp, gp_var: RiskAverseUpperConfidenceBound(gp, gp_var,
                                                                                beta=args.beta, 
                                                                                beta_varproxi=args.beta,
                                                                                gamma=args.gamma
                                                                                )

                inputs, scores, gp, scores_var, gp_var, input_queries, eval_times = bo_step_adaptive_risk_averse(inputs, scores, input_queries, args.gamma, current_best, 
                                                                                                    args.min_repeat_eval, args.repeat_eval,
                                                                                                    (args.n_initial+args.n_budget)*args.repeat_eval,
                                                                                                    objective, bounds, GP=GP, acquisition=acquisition,
                                                                                                    q=1, state_dict=state_dict, input_query=True,
                                                                                                    train_Yvar=scores_var)
                ralcb_function = RiskAverseUpperConfidenceBound(gp, gp_var, args.beta, args.beta, args.gamma, maximize=False)
                lcbs = ralcb_function.forward(inputs.reshape((len(inputs), 1, -1)))
                idxs.extend([int(lcbs.argmax())] * eval_times)
                # reporting, _ = optimize_acqf(ralcb_function, bounds=bounds, q=1, num_restarts=1, raw_samples=1000)
                # if reportings is None:
                #     reportings = torch.cat([reporting] * eval_times)
                # else:
                #     reportings = torch.cat([reportings, torch.cat([reporting] * eval_times)])

                rule = ralcb_function.forward(inputs.reshape((len(inputs), 1, -1)), rule=True)
            
                if int(rule.max()) > current_best:
                    current_best = int(rule.max())
                state_dict = gp.state_dict()
                gps_state_dict.append(state_dict)
                gps_var_state_dict.append(gp_var.state_dict())
                best_observed.append(scores.max())
        
        results['inputs'] = inputs
        results['input_queries'] = input_queries
        results['reporting'] = inputs[idxs]
        results['reporting_idx'] = idxs
        results['scores'] = scores
        results['scores_var'] = scores_var
        results['scores_best'] = best_observed
        results['gps'] = gps_state_dict
        results['gps_var'] = gps_var_state_dict
        results_all.append(results)

    # save to pickle
    with open(f'{benchmark.path}/{args.BO_algo}_results.pkl', 'wb')as f:
        pickle.dump(results_all, f)

    benchmark.close()



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)