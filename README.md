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

## Example
To obtain the benchmark results of the reward model, you need to execute bash file in root directory.
<br>
For a different reward model, you should modify bash file.

1. Classifier reward model
```bash
bash scripts/classifier_reward.sh
```

2. Generative reward model with pairwise result
```bash
bash scripts/generative_reward_pair.sh
```

3. Generative reward model 
```bash
bash scripts/generative_reward.sh
```

4. Process reward model
```bash
bash scripts/prm_reward.sh
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