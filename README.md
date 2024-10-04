<h1 align="center"> Evaluating Robustness of Reward Models for Mathematical Reasoning </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2410.01729"><img src="https://img.shields.io/badge/arXiv-2410.01729-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/RewardMATH"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
</p>

Full dataset can be found at: https://huggingface.co/datasets/RewardMATH/RewardMATH


## Installation

[![python](https://img.shields.io/badge/Python-3.10.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
```shell
pip install -r requirements.txt
```

To evaluate results, [MARIO EVAL](https://github.com/MARIO-Math-Reasoning/MARIO_EVAL) needs to be installed. 
### Install MARIO EVAL as Python package
```shell
git clone https://github.com/MARIO-Math-Reasoning/MARIO_EVAL.git
cd MARIO_EVAL
cd latex2sympy && pip install . && cd ..
pip install -e .
```


## Evaluating Reward Models
### Generative Reward Model (LLM-as-a-judge)
#### Direct Assessment
To run api models (e.g., claude-3-5-sonnet-20240620) using direct assessment, run:
```
python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_direct.json \
    --save_path=YOUR_SAVE_PATH \
    --model_name=claude-3-5-sonnet-20240620 \
    --api_key=YOUR_API_KEY \
    --prompt_dir=prompt/experiments_prompts.yaml \
    --prompt_key=llm_judgement \
    --model_type=generative \
    # --num_sample=10
```

To run models with vllm (e.g., meta-llama/Meta-Llama-3-70B-Instruct), run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_direct.json \
    --save_path=YOUR_SAVE_PATH \
    --model_name=meta-llama/Meta-Llama-3-70B-Instruct \
    --prompt_dir=prompt/experiments_prompts.yaml \
    --prompt_key=llm_judgement_pair \
    --model_type=generative \
    --num_gpus 4 \
    # --num_sample=10
```

#### Pairwise Comparison
To run api models (e.g., claude-3-5-sonnet-20240620) using pairwise comparison, run:
```
python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_pairwise.json \
    --save_path=YOUR_SAVE_PATH \
    --model_name=claude-3-5-sonnet-20240620 \
    --api_key=YOUR_API_KEY \
    --prompt_dir=prompt/experiments_prompts.yaml \
    --prompt_key=llm_judgement \
    --model_type=generative \
    # --num_sample=10
```

To run models with vllm (e.g., meta-llama/Meta-Llama-3-70B-Instruct), run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_pairwise.json \
    --save_path=YOUR_SAVE_PATH \
    --model_name=meta-llama/Meta-Llama-3-70B-Instruct \
    --pairwise_exp \
    --prompt_dir=prompt/experiments_prompts.yaml \
    --prompt_key=llm_judgement_pair \
    --model_type=generative \
    --num_gpus 4 \
    # --num_sample=10
```

### Classifier-based Reward Model
To run classifier-based reward models, run:
```
CUDA_VISIBLE_DEVICES=0 python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_direct.json \
    --save_path=YOUR_SAVE_PATH \
    --model_name=RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --model_type=classifier \
    --trust_remote_code \
    --batch_size=8 \
    # --num_sample=10
```

### Process Reward Model (PRM)
To run PRMs, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_direct.json \
    --save_path=YOUR_SAVE_PATH \
    --model_name=peiyi9979/math-shepherd-mistral-7b-prm \
    --model_type=prm \
    # --num_sample=10
```


## Getting Benchmark Scores
```shell
### Results of direct assessment (default)
python src/evaluate_results.py \
    --result_dir=YOUR_RESULTS_FILES \
    --eval_mode=our_reward \

### Results of direct assessment (for PRM)
python src/evaluate_results.py \
    --result_dir=YOUR_RESULTS_FILES \
    --eval_mode=our_reward \
    --prm_mode \

### Results of pairwise comparison
python src/evaluate_results.py \
    --result_dir=YOUR_RESULTS_FILES \
    --eval_mode=our_reward \
    --pairwise
```


## Repository Structure
```
├── dataset/                    <- Stores the datasets used in the project. These may include training, validation, and test sets.
├── prompt/                     <- Contains files related to input prompts or configurations used to guide or configure the processes or models.
├── results/                    <- Holds output files from simulations or model evaluations, such as tables, figures, and logs.
├── scripts/                    <- Includes various scripts used for batch processing, data manipulation, and auxiliary tasks.
├── src/                        <- Source code directory for the project.
|   ├── evaluation/                  ├── Contains scripts and modules for model evaluation, such as performance metrics and test routines.
|   ├── models/                      ├── Includes model definitions and possibly pre-trained models or their configurations.
|   ├── utils/                       ├── Utility scripts and helper functions used across the project.
|   └── *.py                         └── Other Python scripts that do not necessarily fit into the above subdirectories.
└── tests.py                    <- Unit tests for the project's modules, ensuring the correctness of the code.
```

## 👏 Acknowledgements

The underlying codebase for evaluating reward model from [RewardBench](https://github.com/allenai/reward-bench).


## Citation

```bibtex
@misc{kim2024evaluatingrobustnessrewardmodels,
      title={Evaluating Robustness of Reward Models for Mathematical Reasoning}, 
      author={Sunghwan Kim and Dongjin Kang and Taeyoon Kwon and Hyungjoo Chae and Jungsoo Won and Dongha Lee and Jinyoung Yeo},
      year={2024},
      eprint={2410.01729},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01729}, 
}
```