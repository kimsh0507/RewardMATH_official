import random
from typing import List

import torch


class ArmoRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        random.seed(0)

    def __call__(self, samples: List[str], **kwargs):
        """
        samples: List[str]
        """
        device = self.model.device
        out = []
        with torch.no_grad():
            for sample in samples:
                input_ids = self.tokenizer.apply_chat_template(sample, return_tensors="pt").to(device)
                output = self.model(input_ids)
                score = output.score.float().item()
                out.append(score)
        return torch.Tensor(out)
