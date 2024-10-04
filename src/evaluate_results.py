import json
import argparse
import sys

from utils.utils import load_json, save_json
from evaluation.eval import RewardBenchEvaluation, BoNEvaluation, RewardMATHEvaluation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--result_dir2", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--eval_mode', type=str, default='answer_acc', choices=["reward_bench_prm", "best_of_n", "best_of_n_prm", "our_reward"])
    parser.add_argument("--split_type", type=str, default=None, choices=['extract_answer'])
    parser.add_argument("--prm_function", type=str, default=None, choices=['min','max','prod','mean','mean_logit','mean_odd','last', "geometric_mean"])
    parser.add_argument('--prm_mode', action='store_true')
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--except_model', type=str, default=None,)
    args = parser.parse_args()

    return args

def do_eval(args):
    dataset = load_json(args.result_dir)

    if args.eval_mode=="reward_bench_prm":
        func_list = ["min", "max", "prod", "geometric_mean", "mean", "mean_logit", "mean_odd", "last"]
        eval_func = RewardBenchEvaluation(dataset, func_list)
        results = eval_func.get_results()
    elif args.eval_mode=="best_of_n":
        eval_func = BoNEvaluation(dataset, args.eval_mode, None)
        results = eval_func.get_results()
    elif args.eval_mode=="best_of_n_prm":
        eval_func = BoNEvaluation(dataset, args.eval_mode, args.prm_function)
        results = eval_func.get_results()
    elif args.eval_mode=="our_reward":
        if args.prm_mode:
            func_list = ["min", "max", "prod", "geometric_mean", "mean", "mean_logit", "mean_odd", "last"]
        elif args.pairwise:
            func_list = ["pairwise"]
        else:
            func_list = ["normal"]
        eval_func = RewardMATHEvaluation(dataset, func_list, args.except_model)
        results = eval_func.get_results()


if __name__ == "__main__":
    args = parse_args()
    do_eval(args)