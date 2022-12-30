This is the code for our paper: Tongtong Liu,Joe McCalmon, Cameron Lischke, Asif Rahman, Talal Halabi, and Sarra Alqahtani, On Weaponizing Actions in Multi-Agent Reinforcement Learning: Theoretical and Empirical Study on Security and Robustness, The 24th International Conference on Principles and Practice of Multi-Agent Systems, 2022, accepted


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

Python version should be 3.6

## Directories Structure
- `./Attacks for MADDPG/` contains code for all attacks in cooeprative navigation and physical deception
- `./Attacks for Qmix/` contains code for all attacks in StarCraft II
- `./MARL_Detection/` contains code for the detection model and FGSM attack
- `./Demo Video/` contains the visualization of the attack in different environments

## Cooperative Navigation and Physical Deception
### Randomly-timed attack & Strategically-timed attack
- code are available in `./experiments/predicted_agent_0_attacks.py`
- Whitebox, blackbox, random and strategically-timed attacks can be performed for cooperative navigation and physical deception from the 'attack' function in predicted_agent_0_attacks.py
- Parameters threshold, attack_rate, random, and black_box control which attack is run.
- threshold is the threshold of c(s_t) for the strategically timed attack
- To run cooperative navigation, set the flag --scenario=simple_spread. For physical deception, set --scenario=simple_adversary

### Counterfactual attack & Zero-sum attack
- packages and dependencies are available in requirements.txt
- all code for attacks are in the folder `./experiments/`
- CN: stands for cooperative navigation
- PD: stands for physical deception (simple adversary)
- KL: stands for counterfactual attack (simple spread)
- Adv: stands for zero-sum attack
- all trained weights are availble in corresponding folders
- For example: if you want to run counterfactual, blackbox, in cooperative navigation, you should type
`python train_PPO_KL_CN_Blackbox.py --scenario simple_spread`
- attack rate can be specified by variable `attack_rate`, range from 0 to 1
- For PPO, please go back to `PPO.py` to change the state dimension
- 23 for CN (18 + 5)
- 15 for PD (15 + 5)

## StarCraft II
- to install the StarCraft II game, please refer back to StarCraft II's original GitHub for installment procedure in different operating system
- to install the dependency, please check the requirements.txt
- all implementations are in `rollout.py` under function`generate_episode()`
- attack name can be changed in `main.py` using variable args.attack_name:
- `'inverse_glob' for zero sum`
- `'counterfactual_RL' for counterfactual`
- `'strategic' for strategic`
- `'random' for random`
- attack rate can be specified in `rollout.py` using variable name `attack_rate`
- white box/black box can be specified in arguments.py under white_box (True for WB, False for BB)
- other important output info can be found in `runner.py` under function `evaluate()`

## Detection Model
- please follow the "Anomalous behavior detection in c-MARL.pdf" in the folder 

## Appendix
- Please see Appendix.pdf for more supplmentary details for this paper.
