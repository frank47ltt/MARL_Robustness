Readme for multi agent attacks on the qmix starcraft environment.

Packages found in requirements.txt.
pip install -r requirements.txt

Code is run from main.py. Attacks and attack thresholds are specified in the arguments in arguments.py, or dynamically through setting args.###

Most of the influential code is in rollout.py, in generate_episode()

Project has capabilities for white box and black box random, strategically timed, and counterfactual attacks