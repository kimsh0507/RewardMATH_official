#!/bin/bash
export OPENAI_ORGANIZATION="YOUR_OPENAI_ORGANIZATION"
API_KEY="YOUR_API_KEY" 

# meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, gpt-3.5-turbo-0125, gpt-4-0125-preview, claude-3-opus-20240229, claude-3-5-sonnet-20240620, gpt-4o-2024-05-13
# llama3_8B, llama3_70B, chatgpt, gpt4, opus, sonnet-3-5, prometheus_7B, prometheus_8x7B
MODEL_NAME="gpt-4-0125-preview"
SAVE_NAME="gpt4"

SAVE_PATH="results/RewardMATH/${SAVE_NAME}_reward.json"

### API 
# python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=$SAVE_PATH \
#     --model_name=$MODEL_NAME \
#     --api_key=$API_KEY \
#     --prompt_dir=prompt/experiments_prompts.yaml \
#     --prompt_key=llm_judgement \
#     --model_type=generative \
#     # --num_sample=10


# ### vLLM
# CUDA_VISIBLE_DEVICES=4,5,6,7 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=$SAVE_PATH \
#     --model_name=$MODEL_NAME \
#     --api_key=$API_KEY \
#     --prompt_dir=prompt/experiments_prompts.yaml \
#     --prompt_key=llm_judgement \
#     --model_type=generative \
#     --num_gpus 4 \
#     # --num_sample=22


## Prometheus  (promethus_direct)
# CUDA_VISIBLE_DEVICES=4,5,6,7 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=$SAVE_PATH \
#     --model_name=$MODEL_NAME \
#     --api_key=$API_KEY \
#     --prompt_dir=prompt/experiments_prompts.yaml \
#     --prompt_key=promethus_direct \
#     --model_type=generative \
#     --num_gpus 4 \
#     --num_sample=22