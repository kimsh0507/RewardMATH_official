# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    MixtralForCausalLM,
    T5ForConditionalGeneration,
)

from .armorm import ArmoRMPipeline
from .beaver import BeaverCostPipeline, BeaverPipeline, LlamaForScore
from .internlm import InternLMPipeline
from .openassistant import *  # noqa
from .openbmb import LlamaRewardModel, OpenBMBPipeline
from .pipeline import RewardBenchPipeline
from .grm import GRewardModel, GRMPipeline
from .skywork import (
    SkyworkRMPipeline,
    build_skywork_rm,
)
from .starling import (
    LlamaForSequenceClassification,
    StarlingPipeline,
    build_starling_rm,
)
from .prm import ProcessRewardModel, MathShepherd, ReasonEval


ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
)

API_MODEL_LIST = OPENAI_MODEL_LIST + ANTHROPIC_MODEL_LIST


# Please open a PR if you need to add more custom modeling code / utilize existing code for you model
REWARD_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "tulu",
    },
    "openbmb/UltraRM-13b": {
        "model_builder": LlamaRewardModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "openbmb",
    },
    "openbmb/Eurus-RM-7b": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "mistral",
    },
    "PKU-Alignment/beaver-7b-v1.0-reward": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "pku-align",
    },
    "PKU-Alignment/beaver-7b-v1.0-cost": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverCostPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "pku-align",
    },
    "PKU-Alignment/beaver-7b-v2.0-reward": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "pku-align",
    },
    "PKU-Alignment/beaver-7b-v2.0-cost": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverCostPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "pku-align",
    },
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": ArmoRMPipeline,
        "quantized": False,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
        "torch_dtype": torch.bfloat16,
        "chat_template": "tulu",
    },
    "internlm/internlm2-20b-reward": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": InternLMPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "tulu",
    },
    "internlm/internlm2-7b-reward": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": InternLMPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "tulu",
    },
    "internlm/internlm2-1_8b-reward": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": InternLMPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "chat_template": "tulu",
    },
    "Nexusflow/Starling-RM-34B": {
        "model_builder": LlamaForSequenceClassification.from_pretrained,
        "pipeline_builder": StarlingPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Ray2333/GRM-Gemma-2B-sftreg": {
        "model_builder": GRewardModel.from_pretrained,
        "pipeline_builder": GRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Ray2333/GRM-llama3-8B-sftreg": {
        "model_builder": GRewardModel.from_pretrained,
        "pipeline_builder": GRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Skywork/Skywork-Reward-Llama-3.1-8B": {
        "model_builder": build_skywork_rm,
        "pipeline_builder": SkyworkRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Skywork/Skywork-Reward-Gemma-2-27B": {
        "model_builder": build_skywork_rm,
        "pipeline_builder": SkyworkRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
}

DPO_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForCausalLM.from_pretrained,
        "tokenizer_builder": AutoTokenizer.from_pretrained,
    },
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
        "model_builder": MixtralForCausalLM.from_pretrained,
        "tokenizer_builder": LlamaTokenizer.from_pretrained,
    },
}

PRM_MODEL_CONFIG = {
    "default" : {
        "model_class" : ProcessRewardModel,
        "prm_type" : "ProcessRewardModel",
        "model_type": "prm",
        "tokenizer_name": "EleutherAI/llemma_7b"
    },
    "peiyi9979/math-shepherd-mistral-7b-prm" : {
        "model_class" : MathShepherd,
        "prm_type" : "MathShepherd",
        "model_type": "prm",
        "tokenizer_name": "peiyi9979/math-shepherd-mistral-7b-prm"
    },
    "ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf" : {
        "model_class" : ProcessRewardModel,
        "prm_type" : "ProcessRewardModel",
        "model_type": "prm",
        "tokenizer_name": "EleutherAI/llemma_7b"
    },
    "GAIR/ReasonEval-7B" : {
        "model_class" : ReasonEval,
        "prm_type" : "ReasonEval",
        "model_type": "prm",
        "tokenizer_name": "GAIR/ReasonEval-7B"
    },
    "GAIR/ReasonEval-34B" : {
        "model_class" : ReasonEval,
        "prm_type" : "ReasonEval",
        "model_type": "prm",
        "tokenizer_name": "GAIR/ReasonEval-34B"
    },
}