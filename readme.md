This repository contains the official implementation for our paper: [Sequential Stochastic Combinatorial Optimization Using Hierarchal Reinforcement Learning](https://arxiv.org/abs/2305.19450).

In this work, we propose a novel framework — Wake-Sleep Option (WS-Option) — to solve Sequential Stochastic Combinatorial Optimization (SSCO) problems.
Our method leverages hierarchical reinforcement learning to efficiently explore the solution space and handle the combinatorial nature of sequential decisions.

## Getting Started
To reproduce our results, navigate to the specific environment folder (IM or RPP) depending on the task:
```
cd IM  # or cd RPP
```
Then simply run:
```bash
python Main.py
```
Note: Each task environment may have slight differences in configuration or setup.

## Citation
If you find this repository helpful in your research, please consider citing our paper:
```
@article{feng2025sequential,
  title={Sequential Stochastic Combinatorial Optimization Using Hierarchical Reinforcement Learning},
  author={Feng, Xinsong and Yu, Zihan and Xiong, Yanhai and Chen, Haipeng},
  journal={arXiv preprint arXiv:2305.19450},
  year={2025}
}
```