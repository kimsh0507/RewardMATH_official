#!/bin/bash

# CUDA_VISIBLE_DEVICES=4 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=results/RewardMATH/ArmoRM_reward.json \
#     --model_name=RLHFlow/ArmoRM-Llama3-8B-v0.1 \
#     --model_type=classifier \
#     --trust_remote_code \
#     --batch_size=8 \
#     # --num_sample=20

# CUDA_VISIBLE_DEVICES=4 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=results/RewardMATH/internlm2_reward.json \
#     --model_name=internlm/internlm2-7b-reward \
#     --model_type=classifier \
#     --trust_remote_code \
#     --batch_size=2 \
#     # --num_sample=8

# CUDA_VISIBLE_DEVICES=4 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=results/RewardMATH/Eurus-RM_reward.json \
#     --model_name=openbmb/Eurus-RM-7b \
#     --model_type=classifier \
#     --chat_template=mistral \
#     --trust_remote_code \
#     --batch_size=8 \
#     # --num_sample=8

# CUDA_VISIBLE_DEVICES=4 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=results/RewardMATH/beaver_reward.json \
#     --model_name=PKU-Alignment/beaver-7b-v2.0-reward \
#     --model_type=classifier \
#     --chat_template=pku-align \
#     --trust_remote_code \
#     --batch_size=8 \
#     # --num_sample=8

# CUDA_VISIBLE_DEVICES=4 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=results/RewardMATH/oasst-rm_reward.json \
#     --model_name=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
#     --model_type=classifier \
#     --trust_remote_code \
#     --chat_template=oasst_pythia \
#     --batch_size=8 \
    # --num_sample=8

# CUDA_VISIBLE_DEVICES=1 python src/inference_reward.py \
#     --input_path=dataset/benchmark/RewardMATH_direct.json \
#     --save_path=results/RewardMATH/internlm2_20b_reward.json \
#     --model_name=internlm/internlm2-20b-reward \
#     --model_type=classifier \
#     --trust_remote_code \
#     --batch_size=1 \
#     # --num_sample=8