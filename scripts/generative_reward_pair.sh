#!/bin/bash

export OPENAI_ORGANIZATION="YOUR_OPENAI_ORGANIZATION"
API_KEY="YOUR_API_KEY" 

# meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, gpt-3.5-turbo-0125, gpt-4-0125-preview, claude-3-opus-20240229, claude-3-5-sonnet-20240620, gpt-4o-2024-05-13
# llama3_8B, llama3_70B, chatgpt, gpt4, opus, sonnet-3-5, prometheus_7B, prometheus_8x7B
MODEL_NAME="gpt-3.5-turbo-0125"
SAVE_NAME="chatgpt"


SAVE_PATH="results/RewardMATH/${SAVE_NAME}_pairwise_reward.json"

### API 
python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_pairwise.json \
    --save_path=$SAVE_PATH \
    --model_name=$MODEL_NAME \
    --api_key=$API_KEY \
    --pairwise_exp \
    --prompt_dir=prompt/experiments_prompts.yaml \
    --prompt_key=llm_judgement_pair \
    --model_type=generative \
    --num_sample=20

# ### vLLM
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_pairwise.json \
#     --save_path=$SAVE_PATH \
#     --model_name=$MODEL_NAME \
#     --api_key=$API_KEY \
#     --pairwise_exp \
#     --prompt_dir=prompt/experiments_prompts.yaml \
#     --prompt_key=llm_judgement_pair \
#     --model_type=generative \
#     --num_gpus 4 \
#     # --num_sample=22

## Prometheus  (promethus_pair)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_pairwise.json \
#     --save_path=$SAVE_PATH \
#     --model_name=$MODEL_NAME \
#     --api_key=$API_KEY \
#     --prompt_dir=prompt/experiments_prompts.yaml \
#     --pairwise_exp \
#     --prompt_key=promethus_pair \
#     --model_type=generative \
#     --num_gpus 4 \
#     # --num_sample=22