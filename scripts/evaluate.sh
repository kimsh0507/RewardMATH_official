#!/bin/bash

############ Your Parameters ############
# Generative : llama3_8B, llama3_70B, prometheus_7B, prometheus_8x7B
# Classifier : ArmoRM, internlm2, Eurus-RM, beaver, oasst-rm
# PRM : Math_Shepherd, Easy_to_hard, ReasonEval_7B, ReasonEval_34B
MODEL_NAME="ArmoRM"
#########################################

### direct (default)
python src/evaluate_results.py \
    --result_dir "results/RewardMATH/${MODEL_NAME}_reward.json" \
    --eval_mode our_reward \

### direct (for PRM)
# python src/evaluate_results.py \
#     --result_dir results/RewardMATH/${MODEL_NAME}_reward.json \
#     --eval_mode our_reward \
#     --prm_mode \

### pairwise
# python src/evaluate_results.py \
#     --result_dir "results/RewardMATH/${MODEL_NAME}_pairwise_reward.json" \
#     --eval_mode our_reward \
#     --pairwise