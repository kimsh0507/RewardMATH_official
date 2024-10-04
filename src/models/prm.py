from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.reasoneval import ReasonEval_7B, ReasonEval_34B

class PRM(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def _set_model(self):
        pass

    @abstractmethod
    def get_results(self, question, reasoning_steps_list):
        pass


class ProcessRewardModel(PRM):
    def __init__(self, model_name):
       super().__init__(model_name)
       self.model = self._set_model()
       self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
    
    def _set_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    def get_results(self, prompt):
        begin_solution_tokens = self.tokenizer.encode("\n\n# Solution", add_special_tokens=False)[1:]
        scoring_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)[1:]
        eos_token = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(prompt)

        begin_solution_flag = False
        candidate_positions = []
        for start_idx in range(len(input_ids)):
            if tuple(input_ids[start_idx:start_idx+len(begin_solution_tokens)]) == tuple(begin_solution_tokens):
                begin_solution_flag = True

            if begin_solution_flag and tuple(input_ids[start_idx:start_idx+len(scoring_tokens)]) == tuple(scoring_tokens):
                candidate_positions.append(start_idx)

            if input_ids[start_idx] == eos_token:
                candidate_positions.append(start_idx)
                break
        del candidate_positions[0]
        device = self.model.device

        input_tensor = torch.tensor([input_ids]).to(device)
        candidate_positions = torch.tensor(candidate_positions)

        with torch.no_grad():
            logits = self.model(input_tensor).logits
            scores = logits.mean(dim=-1)
            step_scores = scores[0][candidate_positions]
            step_probs = torch.sigmoid(step_scores).tolist()
        
        return step_probs


class MathShepherd(PRM):  
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = self._set_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _set_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto").eval()
    
    def _convert_reasoning_step(self, reasoning_steps_list):
        step_tag = 'ки'
        text = ""
        for i, step in enumerate(reasoning_steps_list):
            text += f"Step {i+1}: {step} {step_tag}\n"
        return text[:-1]
    
    def get_results(self, prompt):
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
        step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        
        device = self.model.device
        input_id = torch.tensor([self.tokenizer.encode(prompt)]).to(device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            step_scores = scores[input_id == step_tag_id].tolist()

        return step_scores


class ReasonEval(PRM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = self._set_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _set_model(self):
        print(self.model_name)
        if self.model_name=="GAIR/ReasonEval-7B":
            return ReasonEval_7B.from_pretrained(self.model_name, device_map="auto")
        elif self.model_name=="GAIR/ReasonEval-34B":
            return ReasonEval_34B.from_pretrained(self.model_name, device_map="auto")
        else:
            raise ValueError(f"Invalid model name")

    def _process_tokenized_result(self, tokenized_result, step_separator):
        separator_token_id = self.tokenizer(step_separator)['input_ids'][-1]
        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)
        
        device = self.model.device
        if self.model_name=="GAIR/ReasonEval-7B":
            adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
            adjusted_token_ids=torch.tensor([adjusted_token_ids]).to(device)
            labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the problems)
        elif self.model_name=="GAIR/ReasonEval-34B":
            adjusted_token_ids=torch.tensor([adjusted_token_ids]).to(device)
            labeled_token_indices = labeled_token_indices[1:]  # Adjusting to skip the first separator (endding of the problems)
        else:
            raise ValueError(f"Invalid model size")

        return adjusted_token_ids, labeled_token_indices

    def get_results(self, prompt):
        step_separator = f"{self.tokenizer.pad_token}"
        tokenized_result = self.tokenizer(prompt)['input_ids']
        
        ## Separating labels and adjusting token IDs
        adjusted_token_ids, labeled_token_indices = self._process_tokenized_result(tokenized_result, step_separator)
        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)

        # Evaluating reasoning steps using ReasonEval
        with torch.no_grad():
            reasoning_scores = self.model(adjusted_token_ids, attention_mask)[0,labeled_token_indices , :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()

        # Calculating the validity and redundancy scores
        ## score: [p_{negative}, p_{neutral}, p_{positive}]
        ## S_{validity} = p_{neutral} + p_{positive}
        step_level_validity_scores =  [(score[1] + score[2]) for score in scores]

        return step_level_validity_scores