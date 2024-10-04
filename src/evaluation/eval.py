import random
from typing import Optional
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from utils.utils import load_json
from evaluation.latex_answer_check import latex_answer_check

random.seed(0)

class BaseEvaluation(ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def get_results(self):
        pass

class AnswerEvaluation(BaseEvaluation):
    def __init__(self, dataset: list, split_type: str=None):
        super().__init__(dataset)
        self.split_type = split_type
    
    def print_results(self, total_acc, each_avg_acc):
        print(f"Total Accuracy: {round(100 * total_acc / len(self.dataset), 2)}%")
        print(f"Avg Accuracy: {round(100 * each_avg_acc / len(self.dataset), 2)}%")
    
    def check_correctness(self, output, answer, config_for_type, split):
        """Refactored to handle exception and split logic."""
        if len(output) == 0:
            try:
                output = output.split("boxed")[1]
            except IndexError:
                return False
        try:
            return latex_answer_check(output, answer, eval_policy=config_for_type["eval_policy"], extract_policy=config_for_type["extract_policy"], split=split)
        except Exception as e:
            print(f"Error checking correctness: {e}")
            return False

    def get_results(self):
        config = load_json("src/evaluation/eval_config.json")
        data_type = "MATH"
        config_for_type = config["policy"][data_type]

        total_acc = 0
        each_avg_acc = 0
        for i, data in enumerate(tqdm(self.dataset)):
            correctness = []
            each_acc = 0
            outputs = data["output"]["answer"] if self.split_type == "extract_answer" else data["output"]["solution"]
            split = None if self.split_type == "extract_answer" else config["extract_pattern"]
            for output in outputs:
                result = self.check_correctness(output, data["answer"], config_for_type, split)
                each_acc += int(result)
                correctness.append(result)

            total_acc += int(each_acc > 0)
            each_avg_acc += each_acc / len(outputs)
            data["output"]['correctness'] = correctness

        self.print_results(total_acc, each_avg_acc)

        return self.dataset
    

class RewardBenchEvaluation(BaseEvaluation):
    def __init__(self, dataset: list, func_list: list):
        super().__init__(dataset)
        self.func_list = func_list

    def _calculate_final_reward(self, output, func_name):
        if func_name=="min":
            return min(output)
        elif func_name=="max":
            return max(output)
        elif func_name=="prod":
            prod = 1
            for out in output:
                prod*=out
            return prod
        elif func_name=="mean":
            return sum(output)/len(output)
        elif func_name=="mean_logit":
            p = np.array(output)
            logit = np.log(p / (1 - p))
            mean_logit = 1 / (1 + np.exp(-np.mean(logit)))
            return mean_logit
        elif func_name=="mean_odd":
            p = np.array(output)
            odds = p / (1 - p)
            mean_odd = np.maximum(0, np.mean(odds))
            return mean_odd
        elif func_name=="last":
            return output[-1]
        else:
            return None
        
    def get_results(self):
        ### Calculate total results
        tmp_results = {}
        for data in self.dataset:
            final_reward = {}
            for func_name in self.func_list:
                final_reward[func_name] = self._calculate_final_reward(data["output"]["step_scores"], func_name)

            problem_id = data["problem_id"]
            if str(problem_id) in tmp_results.keys():
                tmp_results[str(problem_id)].append({
                    "final_reward" : final_reward,
                    "solution_type": data["solution_type"]
                })
            else:
                tmp_results[str(problem_id)] = [{
                    "final_reward" : final_reward,
                    "solution_type": data["solution_type"]
                }]

        ### Compare the reward of chosen and rejected solution
        compare_results = {}
        for func_name in self.func_list:
            compare_results[func_name] = []
            for problem in tmp_results:
                chosen_reward=0; rejected_reward=0
                for d in tmp_results[problem]:
                    if d["solution_type"]=="chosen":
                        chosen_reward = d["final_reward"][func_name]
                    else:   # rejected
                        rejected_reward = d["final_reward"][func_name]
                if chosen_reward > rejected_reward:
                    compare_results[func_name].append(1)
                else:
                    compare_results[func_name].append(0)
        
        print("### Reward Accuracy ###")
        for func_name in self.func_list:
            print(round(100*sum(compare_results[func_name])/len(compare_results[func_name]),3))

        return compare_results

class BoNEvaluation(BaseEvaluation):
    def __init__(self, dataset: list, eval_type: str=None, prm_func: str=None):
        super().__init__(dataset)
        self.eval_type = eval_type
        self.prm_func = prm_func
    
    def _calculate_final_reward(self, output, func_name):
        if len(output)==0:
            return 0.5
        if func_name=="min":
            return min(output)
        elif func_name=="max":
            return max(output)
        elif func_name=="prod":
            prod = 1
            for out in output:
                prod*=out
            return prod
        elif func_name=="geometric_mean":
            g_mean = 1
            for out in output:
                g_mean*=out
            return g_mean**(1/len(output))
        elif func_name=="mean":
            return sum(output)/len(output)
        elif func_name=="mean_logit":
            p = np.array(output)
            logit = np.log(p / (1 - p))
            mean_logit = 1 / (1 + np.exp(-np.mean(logit)))
            return mean_logit
        elif func_name=="mean_odd":
            p = np.array(output)
            odds = p / (1 - p)
            mean_odd = np.maximum(0, np.mean(odds))
            return mean_odd
        elif func_name=="last":
            return output[-1]
        else:
            return None
        
    def check_correctness(self, output, answer):
        if len(output) == 0:
            try:
                output = output.split("boxed")[1]
            except IndexError:
                return False
        try:
            return latex_answer_check(output, answer, eval_policy="aggressive", extract_policy="flex", split=None)
        except Exception as e:
            print(f"Error checking correctness: {e}")
            return False

    def self_consistency(self, answer_list):
        index_dict = {}
        tmp_answer_list = []
        for idx, answer in enumerate(answer_list):
            is_unique = True
            idx_list = [ i for i in index_dict]
            if len(answer)==0:
                answer = "None"     ### for empty answer
            for k, tmp_answer in enumerate(tmp_answer_list):
                if latex_answer_check(answer, tmp_answer, eval_policy="aggressive", extract_policy="flex", split=None):
                    index_key = idx_list[k]
                    index_dict[index_key] += 1
                    is_unique = False
                    break
            if is_unique:
                tmp_answer_list.append(answer)
                index_dict[str(idx)] = 1

        max_value = max(index_dict.values())
        keys_with_max_value = [key for key, value in index_dict.items() if value == max_value]
        if len(keys_with_max_value)>1:
            return keys_with_max_value[0]
        else:
            return keys_with_max_value[0]
        
    def extract_top_response(self, answer_list, scores):
        if self.eval_type=="best_of_n":
            max_score = max(scores)
            max_score_index = [i for i, score in enumerate(scores) if score == max_score]
            if len(max_score_index)>1:
                output_list = [answer_list[i] for i in max_score_index]
                max_voting_key = self.self_consistency(output_list)
                top_response = output_list[int(max_voting_key)]
            else:
                top_response = answer_list[max_score_index[0]]
        elif self.eval_type=="best_of_n_prm":
            score_list = [self._calculate_final_reward(score, func_name=self.prm_func) for score in scores]
            max_score = max(score_list)
            max_score_index = [i for i, score in enumerate(score_list) if score == max_score]
            if len(max_score_index)>1:
                output_list = [answer_list[i] for i in max_score_index]
                max_voting_key = self.self_consistency(output_list)
                top_response = output_list[int(max_voting_key)]
            else:
                top_response = answer_list[max_score_index[0]]
        else:
            raise ValueError("Invalid eval type")
        return top_response

    def get_results(self):
        data_source = ["MATH", "agieval_gaokao", "agieval_sat"]

        sampling_num_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # customize the list

        total_acc = {}
        for sampling_num in sampling_num_list:
            total_acc[str(sampling_num)] = {}
            for source in data_source:
                total_acc[str(sampling_num)][source] = []

        for data in tqdm(self.dataset):
            for sampling_num in sampling_num_list:
                # sampling the top response
                if "score" in data:
                    top_response = self.extract_top_response(data["output"]["answer"][:sampling_num], data["score"][:sampling_num])
                else:
                    top_response = self.extract_top_response(data["output"]["answer"][:sampling_num], None)
                # checking correctness
                result = self.check_correctness(top_response, data["answer"])
                total_acc[str(sampling_num)][data["data_source"]].append(result)

        print("######## Results ########")
        for sampling_num in sampling_num_list:
            for source in data_source: 
                print(round(100*sum(total_acc[str(sampling_num)][source])/len(total_acc[str(sampling_num)][source]),3), end=", ")
            print()
        return self.dataset

class RewardMATHEvaluation(BaseEvaluation):
    def __init__(self, dataset: list, func_list: list, except_model: str=None):
        super().__init__(dataset)
        self.func_list = func_list
        self.except_model = except_model

    def _calculate_final_reward(self, output, func_name):
        if func_name=="min":
            return min(output)
        elif func_name=="max":
            return max(output)
        elif func_name=="prod":
            prod = 1
            for out in output:
                prod*=out
            return prod
        elif func_name=="geometric_mean":
            g_mean = 1
            for out in output:
                g_mean*=out
            return g_mean**(1/len(output))
        elif func_name=="mean":
            return sum(output)/len(output)
        elif func_name=="mean_logit":
            p = np.array(output)
            logit = np.log(p / (1 - p))
            mean_logit = 1 / (1 + np.exp(-np.mean(logit)))
            return mean_logit
        elif func_name=="mean_odd":
            p = np.array(output)
            odds = p / (1 - p)
            mean_odd = np.maximum(0, np.mean(odds))
            return mean_odd
        elif func_name=="last":
            return output[-1]
        else:
            return None
        
    def _calculate_MRR(self, chosen_score, rejected_score, mode="mrr"):
        tmp_list = []
        for rejected_s in rejected_score:
            tmp_list.append((rejected_s, "rej"))
        for chosen_s in chosen_score:
            tmp_list.append((chosen_s, "cho"))
        sorted_list = sorted(tmp_list, key=lambda x: (x[0], x[1]), reverse=True)
        first_chosen_idx = None
        for i, s in enumerate(sorted_list):
            if s[1]=="cho":
                first_chosen_idx = i+1
                break
        if mode=="mrr":
            return 1/first_chosen_idx
        elif mode=="rank":
            return (len(tmp_list) - first_chosen_idx)/(len(tmp_list)-1)
    
    def processing_results(self, data):
       
        chosen_score = {func_name: [] for func_name in self.func_list}
        rejected_score = {func_name: [] for func_name in self.func_list}

        for func_name in self.func_list:
            if func_name == "pairwise":
                win_result = {
                    "total" : [],
                    "easy" : [],
                    "hard": []
                }
            for idx, (data_source, score) in enumerate(zip(data["solution_reference"], data["score"])):
                if func_name == "pairwise":
                    win_result["total"].append(score==data["chosen_position"][idx])
                else:
                    if func_name == "normal":
                        tmp_score = score
                    else:
                        tmp_score =  self._calculate_final_reward(score, func_name=func_name)
                    if data_source == "human_to_GPT-4":
                        chosen_score[func_name].append(tmp_score)
                    else:
                        if data_source!=self.except_model:
                            rejected_score[func_name].append(tmp_score)

            if func_name == "pairwise":
                # chosen > rejected -> reward_rej = 1, chosen < rejected  -> reward_rej : largest reward
                largest_reward = len(win_result["total"])+2
                chosen_score[func_name].append(sum(win_result["total"])+1)
                rejected_score[func_name] = [1 if x else largest_reward for x in win_result["total"]]
                
        return {
            "problem" : data["problem"],
            "chosen_score" : chosen_score,
            "rejected_score" : rejected_score
        }
        
    def get_results(self):
        final_data_score = [self.processing_results(data) for data in self.dataset]

        metric_list = ["reward Acc", "reward MRR"]
        final_results = {}
        for func_name in self.func_list:
            final_results[func_name] = {metric_name: [] for metric_name in metric_list}
            final_results[func_name]["reward Acc (w/ tie)"] = []
            for data_score in final_data_score:
                final_results[func_name]["reward Acc (w/ tie)"].append(min(data_score["chosen_score"][func_name]) >= max(data_score["rejected_score"][func_name]))
                final_results[func_name]["reward Acc"].append(min(data_score["chosen_score"][func_name]) > max(data_score["rejected_score"][func_name]))
                final_results[func_name]["reward MRR"].append(self._calculate_MRR(data_score["chosen_score"][func_name], data_score["rejected_score"][func_name]))

        print("### Total Results ###")
        for func_name in self.func_list:
            if func_name=="normal":
                print("w/ tie Acc: ", round(100*sum(final_results[func_name]["reward Acc (w/ tie)"])/len(final_results[func_name]["reward Acc (w/ tie)"]),3))
            for metric in final_results[func_name]:
                if metric=="reward Acc (w/ tie)":
                    pass
                print(round(100*sum(final_results[func_name][metric])/len(final_results[func_name][metric]),3), end=", ")
            print()
                    

        return final_results