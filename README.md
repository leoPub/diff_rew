# READ ME

> *Source code for paper "A Differentiated Reward Method for Reinforcement Learning Based Multi-Vehicle Cooperative Decision-Making Algorithms"*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img
    src="./assets/banner.png" width="800"
  />
</p>

We provide the source code implementation of the reward-differential method for the QMIX algorithm herein. Due to substantial subsequent modifications in our extended work, implementations for MADQN and MAPPO could be adapted from the repositories we provide below. For any reproduction issues, please contact leohancnjs@outlook.com.

## 🚀 Getting Started

1. Clone this repository

2. Install [Flow](https://flow-project.github.io/) to [`./flow/`](./flow/) and config SUMO binary
   
   Flow is a computational framework for deep RL and control experiments for traffic microsimulation.
   To install Flow and the compatible SUMO binary, you should follow
   [Flow Installation instructions](http://flow.readthedocs.io/en/latest/flow_setup.html). Make sure the test case could run.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up algorithm parameters in [`./config.py`](./config.py)

6. Run experiment from [`./highway_exp.py`](./highway_exp.py)

## 🤗 Repo Links

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.
