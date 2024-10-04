import random
from abc import ABC, abstractmethod
from copy import deepcopy

# from fastchat.conversation import get_conv_template

from utils.conversation import get_conv_template
from utils.utils import load_prompt, load_json, numbered_list_prompt
from utils.rewardbench_utils import check_tokenizer_chat_template

PROMPT_STYLE = {
    "gpt-3.5-turbo-0125": "default",
    "gpt-3.5-turbo-1106": "default",
    "gpt-3.5-turbo": "default",
    "gpt-4": "default",
    "gpt-4-0613": "default",
    "gpt-4-0125-preview": "default",
    "gpt-4o": "default",
    "gpt-4o-2024-05-13": "default",
    "claude-3-opus-20240229": "default",
    "claude-3-sonnet-20240229": "default",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3",
    "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral",
    "google/gemma-7b-it": "gemma",
    "google/gemma-2-27b-it": "gemma",
    "allenai/tulu-2-dpo-7b": "tulu",
    "allenai/tulu-2-dpo-70b": "tulu",
    "microsoft/Phi-3-medium-4k-instruct": "phi",
    "EleutherAI/llemma_7b": "default",
    "EleutherAI/llemma_34b": "default",
    "WizardLMTeam/WizardMath-7B-V1.1": "default",
    "deepseek-ai/deepseek-math-7b-instruct": "deepseek-chat",
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "deepseek-chat",
    "Qwen/Qwen1.5-7B": "qwen-7b-chat",
    "prometheus-eval/prometheus-7b-v2.0": "mistral",
    "prometheus-eval/prometheus-8x7b-v2.0": "mistral"
}


class PromptHandler(ABC):
    '''
    Base PromptHandler
    '''
    def __init__(self, dataset, model_name, prompt_dir, prompt_key, few_shot_dir):
        super().__init__()
        self.dataset = dataset
        self.model_name = model_name
        self.prompt_template = load_prompt(prompt_dir)[prompt_key]
        self.examplar = load_json(few_shot_dir) 
        if 'prompt' in self.prompt_template:
            self.system_message = self.prompt_template['prompt']
        else:
            self.system_message = None
    
    def _create_conversation(self, messages, style):
        if style=="default":
            if len(messages)<=2:
                message_list = [message['content'] for message in messages]
                prompt = "\n\n".join(message_list)
            else:
                prompt = ""
                for i, message in enumerate(messages):
                    if message["role"] == "system":
                        prompt += message["content"] + "\n\n"
                    elif message["role"] == "user":
                        prompt += f"[Example {i//2+1}]\n" + message["content"] + "\n"
                    else:
                        prompt += message["content"] + "\n"
            return prompt
        else:
            conv = get_conv_template(style)
            for i, message in enumerate(messages):
                role = message['role']
                content = message['content']
                if role == "system":
                    conv.set_system_message(content)
                else:
                    # conv.append_message(conv.roles[role == "assistant"], content)
                    if i%2 == 1:
                        conv.append_message(conv.roles[0], content)
                    else:
                        conv.append_message(conv.roles[1], content)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
    
    def _set_exampler(self, num_shots):
        if len(self.examplar) > num_shots:
            self.examplar = random.sample(self.examplar, num_shots)
        elif len(self.examplar) < num_shots:
            raise ValueError("Insufficient exemplars: requested number of shots exceeds available exemplars.")

    def generate_prompt(self, num_shots=None):
        final_dataset = []
        for data in self.dataset:
            messages = [{"role": "system", "content": self.system_message}]
            ## For new_solution
            if "solution" not in data:
                data['solution'] = numbered_list_prompt(data["new_solution"])
            ## few shot
            few_shot_message = self._get_few_shot(num_shots) if num_shots else None
            messages.extend(few_shot_message or [])
            ## Data to inference
            messages.append({"role": "user", "content": self.prompt_template['format']['user'].format(**data)})
            data["prompt"] = self._create_conversation(messages, PROMPT_STYLE.get(self.model_name, "default"))
            final_dataset.append(data)
        return final_dataset


class SolutionPromptHandler(PromptHandler):
    '''
    PromptHandler for generating solution
    '''
    def __init__(self, dataset, model_name, prompt_dir, prompt_key, few_shot_dir):
        super().__init__(dataset, model_name, prompt_dir, prompt_key, few_shot_dir)

    def _get_few_shot(self, num_shots):
        examples = []
        for x in self.examplar:
            examples.append({
                "problem" : x["problem"],
                "solution": numbered_list_prompt(x["solution"]),
                "answer": x["answer"],
            })
        few_shot_message = []
        for i, ex in enumerate(examples[:num_shots]):
            few_shot_message.append({"role": "user", "content": self.prompt_template['format']['user'].format(**ex)})
            few_shot_message.append({"role": "assistant", "content": self.prompt_template['format']['assistant'].format(**ex)})
        return few_shot_message
    
    def generate_prompt(self, num_shots=None):
        final_dataset = []
        for data in self.dataset:
            if self.system_message:
                messages = [{"role": "system", "content": self.system_message}]
            else:
                messages = []
            ## few shot
            if num_shots:
                few_shot_message = self._get_few_shot(num_shots)
                messages.extend(few_shot_message)
                
            messages.append({"role": "user", "content": self.prompt_template['format']['user'].format(**data)})
            if "gpt" in self.model_name or "claude" in self.model_name:
                data["prompt"] = messages
            else:
                data["prompt"] = self._create_conversation(messages, PROMPT_STYLE.get(self.model_name, "default"))
            final_dataset.append(data)
        return final_dataset


class ExperimentPromptHandler:
    '''
    PromptHandler for main experiments
    '''
    def __init__(self, dataset, model_name, prompt_dir, prompt_key):
        self.dataset = dataset
        self.model_name = model_name
        self.prompt_key = prompt_key
        if prompt_dir:
            self.prompt_template = load_prompt(prompt_dir)[prompt_key]
            if 'prompt' in self.prompt_template:
                self.system_message = self.prompt_template['prompt']
            else:
                self.system_message = None
        else:
            self.prompt_template = None
            self.system_message = None
    
    def _create_conversation(self, messages, style):
        if style=="default":
            if len(messages)<=2:
                message_list = [message['content'] for message in messages]
                prompt = "\n\n".join(message_list)
            else:
                prompt = ""
                for i, message in enumerate(messages):
                    if message["role"] == "system":
                        prompt += message["content"] + "\n\n"
                    elif message["role"] == "user":
                        prompt += f"[Example {i//2+1}]\n" + message["content"] + "\n"
                    else:
                        prompt += message["content"] + "\n"
            return prompt
        else:
            conv = get_conv_template(style)
            for i, message in enumerate(messages):
                role = message['role']
                content = message['content']
                if role == "system":
                    conv.set_system_message(content)
                else:
                    if i%2 == 1:
                        conv.append_message(conv.roles[0], content)
                    else:
                        conv.append_message(conv.roles[1], content)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
    
    def generate_prompt(self):
        final_dataset = []
        for data in self.dataset:
            if self.system_message:
                messages = [{"role": "system", "content": self.system_message}]
            else:
                messages = []
            messages.append({"role": "user", "content": self.prompt_template['format']['user'].format(**data)})
            if self.prompt_key=="Mistral_MetaMATH":
                data["prompt"] = f"{messages[0]['content']} "
            else:
                data["prompt"] = self._create_conversation(messages, PROMPT_STYLE.get(self.model_name, "default"))
            final_dataset.append(data)
        return final_dataset
    
class RewardModelPromptHandler(ExperimentPromptHandler):
    '''
    PromptHandler for reward model (Generative Reward Model, classifier reward model, process reward model)
    '''
    def __init__(self, dataset, model_name, prompt_dir, prompt_key, tokenizer, chat_template, model_type="generative", custom_dialogue_formatting=None):
        super().__init__(dataset, model_name, prompt_dir, prompt_key)
        self.tokenizer=tokenizer
        if tokenizer:
            self.usable_tokenizer=check_tokenizer_chat_template(tokenizer)
        else:
            self.usable_tokenizer=None
        self.chat_template=chat_template
        self.model_type=model_type
        self.custom_dialogue_formatting=custom_dialogue_formatting
    
    def generate_prompt(self, prm_type=None):
        final_dataset = []
        for data in self.dataset:
            for solution, answer in zip(data["output"]["solution"],data["output"]["answer"]):
                tmp_data = deepcopy(data)
                tmp_data["tmp_solution"] = solution
                tmp_data["tmp_answer"] = answer
                del tmp_data["output"]

                if self.model_type=="generative":
                    messages = [{"role": "system", "content": self.system_message}]
                    tmp_data["eval_solution"] = numbered_list_prompt(solution)
                    messages.append({"role": "user", "content": self.prompt_template['format']['user'].format(**tmp_data)})
                    if "gpt" in self.model_name or "claude" in self.model_name:
                        tmp_data["prompt"] = messages
                    else:
                        tmp_data["prompt"] = self._create_conversation(messages, PROMPT_STYLE.get(self.model_name, "default"))
                elif self.model_type=="classifier":
                    messages = [
                        {"role": "user", "content": data["problem"]},
                        {"role": "assistant", "content": "\n".join(solution)},
                    ]
                    if not self.custom_dialogue_formatting:
                        if self.usable_tokenizer:
                            tmp_data["prompt"] = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                            )
                        else:
                            tmp_data["prompt"] = self._create_conversation(messages, self.chat_template)
                    else:
                        tmp_data["prompt"] = messages
                elif self.model_type=="prm":
                    if prm_type=="ProcessRewardModel":
                        solution_text = "\n\n".join(solution)
                        PROMPT_FORMAT = "# Question\n\n{question}\n\n# Solution\n\n{output}"
                        tmp_data["prompt"] = PROMPT_FORMAT.format(question=data["problem"], output=solution_text)
                    elif prm_type=="MathShepherd":
                        PROMPT_FORMAT = "{question} {output}"
                        step_tag = 'ки'
                        solution_text = ""
                        for i, step in enumerate(solution):
                            solution_text += f"Step {i+1}: {step} {step_tag}\n"
                        tmp_data["prompt"] = PROMPT_FORMAT.format(question=data["problem"], output=solution_text[:-1])
                    elif prm_type=="ReasonEval":
                        PROMPT_FORMAT = "Question:\n{question}\nAnswer:\nLet's think step by step.\n"
                        step_separator = f"{self.tokenizer.pad_token}"
                        combined_steps=""
                        reasoning_steps = []
                        for i, step in enumerate(solution):
                            reasoning_steps.append(f"{i+1}. {step} ")
                        for steps in reasoning_steps:
                            combined_steps += steps + step_separator
                        tmp_data["prompt"] = PROMPT_FORMAT.format(question=data["problem"]) + step_separator + combined_steps
                    else:
                        raise ValueError("Insufficient Process Reward Model Type")
                else:
                    raise ValueError("Insufficient Model Type")
                final_dataset.append(tmp_data)

        return final_dataset

    
class RewardExperimentPromptHandler(ExperimentPromptHandler):
    '''
    PromptHandler for reward model (Generative Reward Model, classifier reward model, process reward model)
    '''
    def __init__(self, dataset, model_name, prompt_dir, prompt_key, tokenizer, chat_template, model_type="generative", custom_dialogue_formatting=None, pairwise_exp=False):
        super().__init__(dataset, model_name, prompt_dir, prompt_key)
        self.tokenizer=tokenizer
        if tokenizer:
            self.usable_tokenizer=check_tokenizer_chat_template(tokenizer)
        else:
            self.usable_tokenizer=None
        self.chat_template=chat_template
        self.model_type=model_type
        self.custom_dialogue_formatting=custom_dialogue_formatting
        self.pairwise_exp = pairwise_exp
    
    def generate_prompt(self, prm_type=None):
        final_dataset = []
        for data in self.dataset:
            tmp_data = deepcopy(data)
            if self.pairwise_exp:
                solution_A = data["eval_solution_A"]
                solution_B = data["eval_solution_B"]
            else:
                solution = data["eval_solution"]
                tmp_data["tmp_solution"] = solution

            if self.model_type=="generative":
                messages = [{"role": "system", "content": self.system_message}]
                if self.pairwise_exp:
                    tmp_data["eval_solution_A"] = numbered_list_prompt(solution_A)
                    tmp_data["eval_solution_B"] = numbered_list_prompt(solution_B)
                else:
                    tmp_data["eval_solution"] = numbered_list_prompt(solution)
                messages.append({"role": "user", "content": self.prompt_template['format']['user'].format(**tmp_data)})
                if "gpt" in self.model_name or "claude" in self.model_name:
                    tmp_data["prompt"] = messages
                else:
                    tmp_data["prompt"] = self._create_conversation(messages, PROMPT_STYLE.get(self.model_name, "default"))
            elif self.model_type=="classifier":
                messages = [
                    {"role": "user", "content": data["problem"]},
                    {"role": "assistant", "content": "\n".join(solution)},
                ]
                if not self.custom_dialogue_formatting:
                    if self.usable_tokenizer:
                        tmp_data["prompt"] = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                        )
                    else:
                        tmp_data["prompt"] = self._create_conversation(messages, self.chat_template)
                else:
                    tmp_data["prompt"] = messages
            elif self.model_type=="prm":
                if prm_type=="ProcessRewardModel":
                    solution_text = "\n\n".join(solution)
                    PROMPT_FORMAT = "# Question\n\n{question}\n\n# Solution\n\n{output}"
                    tmp_data["prompt"] = PROMPT_FORMAT.format(question=data["problem"], output=solution_text)
                elif prm_type=="MathShepherd":
                    PROMPT_FORMAT = "{question} {output}"
                    step_tag = 'ки'
                    solution_text = ""
                    for i, step in enumerate(solution):
                        solution_text += f"Step {i+1}: {step} {step_tag}\n"
                    tmp_data["prompt"] = PROMPT_FORMAT.format(question=data["problem"], output=solution_text[:-1])
                elif prm_type=="ReasonEval":
                    PROMPT_FORMAT = "Question:\n{question}\nAnswer:\nLet's think step by step.\n"
                    step_separator = f"{self.tokenizer.pad_token}"
                    combined_steps=""
                    reasoning_steps = []
                    for i, step in enumerate(solution):
                        reasoning_steps.append(f"{i+1}. {step} ")
                    for steps in reasoning_steps:
                        combined_steps += steps + step_separator
                    tmp_data["prompt"] = PROMPT_FORMAT.format(question=data["problem"]) + step_separator + combined_steps
                else:
                    raise ValueError("Insufficient Process Reward Model Type")
            else:
                raise ValueError("Insufficient Model Type")
            final_dataset.append(tmp_data)

        return final_dataset