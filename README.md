# Multi-Agent-Security
This is the code for paper Advanced Persistent Threat in Cooperative Multi-Agent Reinforcement Learning. 

**Note**: Environments include requirements.txt file in each folder. 

In order to install virtual environments:

- `cd` into the corresponding directories 
- type `pip install -r requirements.txt`

This project utilized the following GitHub as referenced, and if encountered any problem regarding installment of the environment, user should follow the corresponding GitHub in the following: 

MADDPG: https://github.com/openai/maddpg

Multi-agent-particle: https://github.com/openai/multiagent-particle-envs

DDPG/PPO: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

AIRL: https://github.com/yangmuzhi/airl

Qmix on STARCRAFT II: https://github.com/deepmind/pysc2

Detection Dataset are avaialble in this Google Drive: FIXMEE!!!!!!!!

## Directories Structure
- `./Attacks for MADDPG/` contains code for all attacks in cooeprative navigation and physical deception
- `./Attacks for Qmix/` contains code for all attacks in StarCraft II
- `./MARL_Detection/` contains code for the detection model and FGSM attack
- `./Demo Video/` contains the visualization of the attack in different environments

## Cooperative Navigation and Physical Deception
### Randomly-timed attack & Strategically-timed attack
- code are available in `./experiments/predicted_agent_0_attacks.py
(For Joe)

### Counterfactual attack & Zero-sum attack
- packages and dependencies are available in requirements.txt
- all code for attacks are in the folder `./experiments/`
- CN: stands for cooeprative navigation
- PD: stands for physical deception (simple adversary)
- KL: stands for counterfactual attack (simple spread)
- Adv: stands for zero-sum attack
- all trained weights are availble in corresponding folders
- For example: if you want to run counterfactual, blackbox, in cooperative navigation, you should type
`python train_PPO_CF_KL_CN_blackbox.py --scenario simple_spread`
- attack rate can be specified by variable `attack_rate`, range from 0 to 1

## StarCraft II
- 

