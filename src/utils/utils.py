import importlib
import json
import re
from sys import version
import warnings
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

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True