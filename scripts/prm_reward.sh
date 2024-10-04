#!/bin/bash

#  peiyi9979/math-shepherd-mistral-7b-prm,  ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf,  GAIR/ReasonEval-7B, GAIR/ReasonEval-34B
# "Math_Shepherd", "Easy_to_hard", "ReasonEval_7B", "ReasonEval_34B"

MODEL_NAME="peiyi9979/math-shepherd-mistral-7b-prm"
SAVE_NAME="Math_Shepherd"

SAVE_PATH="results/RewardMATH/${SAVE_NAME}_reward.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_direct.json \
    --save_path=$SAVE_PATH \
    --model_name=$MODEL_NAME \
    --model_type=prm \
    --num_sample=2