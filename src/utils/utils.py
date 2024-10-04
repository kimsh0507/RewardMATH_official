import json
import re
import yaml

def parsing_steps(steps):
    ptn = r'\n\d+\.'
    parsed_steps = re.split(ptn, "\n"+ steps)
    
    s = [p.strip() for p in parsed_steps if p.strip()]

    return s

def numbered_list_prompt(solution, strat_idx=1):
    '''
    solution : list
    return prompt -> numbered list format
    '''
    prompt = ""
    for i, sol in enumerate(solution):
        prompt += f"{i+strat_idx}. {sol}\n"
    return prompt[:-1]

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_prompt(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)
    return prompt