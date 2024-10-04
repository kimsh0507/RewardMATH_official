import random
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

SUPPORTED_SKYWORK_MODELS = ["Skywork/Skywork-Reward-Llama-3.1-8B", "Skywork/Skywork-Reward-Gemma-2-27B"]


def build_skywork_rm(model_name, **kwargs):
    if model_name == "Skywork/Skywork-Reward-Llama-3.1-8B":
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            num_labels=1,
        )
    elif model_name == "Skywork/Skywork-Reward-Gemma-2-27B":
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
    else:
        raise ValueError(
            f"Model {model_name} not found in Skywork reward models. Supported are {SUPPORTED_SKYWORK_MODELS}"
        )

    reward_model.eval().requires_grad_(False)
    return reward_model

class SkyworkRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs).logits

        return outputs