import re
from abc import ABC, abstractmethod
from copy import deepcopy

from utils.utils import parsing_steps, load_json, save_json

class ResultCollector(ABC):
    def __init__(self, prompt_key):
        self.prompt_key = prompt_key

    @abstractmethod
    def data_processing(self, base_dataset, dataset):
        pass


class BaseResultCollector(ResultCollector):
    def __init__(self, prompt_key):
        super().__init__(prompt_key)
    
    def data_processing(self, base_dataset, dataset):
        final_dataset=[]
        for base_data, data in zip(base_dataset, dataset):
            if base_data["problem"]!=data["problem"]:
                print("Something is worng!!!")
                break
            final_dataset.append({
                "problem": base_data["problem"],
                "solution": base_data["solution"],
                "answer": base_data["answer"],
                "subject": base_data["subject"],
                "level": base_data["level"],
                "output": data["output"]
            })
        return final_dataset


class SolutionCollector(ResultCollector):
    def __init__(self, prompt_key):
        super().__init__(prompt_key)
    
    def _find_first_number(self, text):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        else:
            return None 

    def _make_solution_answer(self, results, mode=None):
        sol_list = []
        ans_list = []
        if mode=="WizardMath":
            for tmp_result in results:
                splits = re.split(r'(?<=\n\n)(?=[0-9A-Z])', tmp_result[1:])
                cleaned_splits = [s[:-2] for s in splits[:-1]]
                tmp_solution = []
                for cleaned_split in cleaned_splits:
                    if "# Answer" in cleaned_split:
                        break
                    if "Step" in cleaned_split:
                        tmp_solution.append(re.split(r'Step \d+: ', cleaned_split)[-1].replace("\n\n","\n").strip())
                    else:
                        tmp_solution.append(cleaned_split.replace("\n\n","\n").strip())
                sol_list.append(tmp_solution)
                if "The answer is: " in tmp_result:
                    ans_list.append(tmp_result.split("The answer is: ")[-1][:-1].strip())
                else:
                    ans_list.append("None")
        elif mode=="Mistral_MetaMATH":
            for tmp_result in results:
                cleaned_splits = tmp_result.split("The answer is:")[0].split("ки\n")    # "ки" is special step tag
                tmp_solution = []
                for cleaned_split in cleaned_splits:
                    if "Step" in cleaned_split:
                        tmp_solution.append(re.split(r'Step \d+: ', cleaned_split)[-1].replace("\n\n","\n").strip())
                    else:
                        tmp_solution.append(cleaned_split.replace("\n\n","\n").strip())
                sol_list.append(tmp_solution)
                if "The answer is: " in tmp_result:
                    ans_list.append(tmp_result.split("The answer is: ")[-1].replace("ки", "").strip())
                else:
                    ans_list.append("None")
        else:
            for tmp_result in results:
                try:
                    solution_list = tmp_result.split("\n### Answer:")[0].split("Solution:")[1]
                except:
                    solution_list = tmp_result.split("\n### Answer:")[0]
                sol_list.append(parsing_steps(solution_list))
                ans_list.append(tmp_result.split("\n### Answer:")[-1].split("\n")[0].strip())
        return sol_list, ans_list
    
    def _group_dataset(self, final_dataset):
        final_dict = {}
    
        for data in final_dataset:
            problem = data["problem"]
            if problem not in final_dict:
                final_dict[problem] = data
            else:
                final_dict[problem]["output"]["solution"].extend(data["output"]["solution"])
                final_dict[problem]["output"]["answer"].extend(data["output"]["answer"])
        
        return list(final_dict.values())

    def data_processing(self, base_dataset, dataset):
        final_dataset = []
        for base_data, data in zip(base_dataset, dataset):
            if base_data["problem"]!=data["problem"]:
                print("Something is worng!!!")
                break
            
            if self.prompt_key=="WizardMath":
                sol_list, ans_list = self._make_solution_answer(data["output"], mode="WizardMath")
            elif self.prompt_key=="Mistral_MetaMATH":
                sol_list, ans_list = self._make_solution_answer(data["output"], mode="Mistral_MetaMATH")
            else:
                sol_list, ans_list = self._make_solution_answer(data["output"])

            reasoning_prompt = ["WizardMath", "Mistral_MetaMATH"]
            if self.prompt_key in reasoning_prompt:
                final_dataset.append({
                    "problem": base_data["problem"],
                    "answer": base_data["answer"],
                    "data_source": base_data["data_source"],
                    "feature": base_data["feature"],
                    "output": {
                        "solution": sol_list,
                        "answer": ans_list,
                    }
                })
            else:
                final_dataset.append({
                    "problem": base_data["problem"],
                    "solution": base_data["solution"],
                    "answer": base_data["answer"],
                    "subject": base_data["subject"],
                    "level": base_data["level"],
                    "output": {
                        "solution": sol_list,
                        "answer": ans_list,
                    }
                })
        return final_dataset



class RewardBenchCollector(ResultCollector):
    def __init__(self, prompt_key):
        super().__init__(prompt_key)
        
    def _make_results(self, output):
        if "prm" in self.prompt_key:
            results = {
                "step_scores" : output
            }
        else:
            results = None
        return results

    def data_processing(self, base_dataset, dataset):
        final_dataset = []
        for base_data, data in zip(base_dataset, dataset):
            if base_data["problem"]!=data["problem"]:
                print("Something is worng!!!")
            else:
                results = self._make_results(data["output"][0])
                final_dataset.append({
                    "problem": base_data["problem"],
                    "eval_solution": base_data["eval_solution"],
                    "subject": base_data["subject"],
                    "level": base_data["level"],
                    "output" : results,
                    "problem_id" : base_data["problem_id"],
                    "solution_type": base_data["solution_type"]
                })
            
        return final_dataset
    

class RewardCollector(ResultCollector):
    def __init__(self, prompt_key):
        super().__init__(prompt_key)
    
    def _find_first_number(self, text):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        else:
            return 1    ## parsing error -> get score "1"
    
    def _make_results(self, output):
        if "Rating:" in output:
            return self._find_first_number(output.split("Rating:")[-1])
        else:
            return 1    ## parsing error -> get score "1"


    def _group_dataset(self, final_dataset):
        final_dict = {}
    
        for data in final_dataset:
            problem = data["problem"]
            if problem not in final_dict:
                final_dict[problem] = data
            else:
                final_dict[problem]["output"]["solution"].extend(data["output"]["solution"])
                final_dict[problem]["output"]["answer"].extend(data["output"]["answer"])
                final_dict[problem]["output"]["reward"].extend(data["output"]["reward"])
        
        return list(final_dict.values())

    def data_processing(self, base_dataset, dataset):
        final_dataset = []
        for base_data, data in zip(base_dataset, dataset):
            if base_data["problem"]!=data["problem"]:
                print("Something is worng!!!")
            else:
                if self.prompt_key=="classifier":     # classifier model
                    reward = self._make_results(data["output"][0])
                elif self.prompt_key=="prm":     # process reward model
                    reward = self._make_results(data["output"][0])
                else:
                    reward = self._make_results(data["output"][0])
                final_dataset.append({
                    "problem": base_data["problem"],
                    "answer": base_data["answer"],
                    "data_source": base_data["data_source"],
                    "feature": base_data["feature"],
                    "output": {
                        "solution": base_data["tmp_solution"],
                        "answer": base_data["tmp_answer"],
                        "reward": reward
                    }
                })

        final_dataset = self._group_dataset(final_dataset)
        return final_dataset