# READ ME

> *Source code for paper "A Differentiated Reward Method for Reinforcement Learning Based Multi-Vehicle Cooperative Decision-Making Algorithms"*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img
    src="./assets/banner.png" width="800"
  />
</p>

We provide the source code implementation of the reward-differential method for the QMIX algorithm herein. Due to substantial subsequent modifications in our extended work, implementations for all methodologies cannot be comprehensively provided. The MADQN and MAPPO implementations were adapted from the following repositories respectively. For any reproduction issues, please contact leohancnjs@outlook.com.

## 🚀 Getting Started

1. Clone this repository

2. Install [Flow](https://flow-project.github.io/) to [`./flow/`](./flow/)
   
   Flow is a computational framework for deep RL and control experiments for traffic microsimulation.
   Follow [Flow Installation instructions](http://flow.readthedocs.io/en/latest/flow_setup.html) to finish installation. Make sure the test case could run.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up algorithm parameters in [`./config.py`](./config.py)

5. Run experiment from [`./highway_exp.py`](./highway_exp.py)
