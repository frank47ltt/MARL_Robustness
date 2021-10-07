import numpy as np
import torch
from torch.distributions import Categorical
import sys
from adv_qmix import ADV_QMIX
sys.path.insert(0,"/root/qmix")

class Adversary:
  def __init__(self, args):
    self.n_actions = args.n_actions
    self.n_agents = args.n_agents
    self.state_shape = args.state_shape
    self.obs_shape = args.obs_shape
    self.policy = ADV_QMIX(args)
    self.args = args
    # self.agents
    # self.env
      
  def random_attack(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
    action_ind_list = np.nonzero(avail_actions)[0]
    print("random attack here is " + str(action_ind_list))
    action = np.random.choice(action_ind_list)
    return action
  
  def random_time_attack(self, q_val, avail_actions):
    q_value = q_val
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)   # 增加一个维度 从 (available action) --> (1, available action)
    q_value[avail_actions == 0.0] =  float("inf")
    # if np.random.uniform() < epsilon:   #EXPLORATION
    #   action = np.random.choice(avail_actions_ind)  
    # else:                             #EXPLIOTATION
    action = torch.argmin(q_value)  #USES ARGMAX FOR ACTION
    return action

  def strategic_time_attack(self, q_val, avail_action, epsilon,  threshold):
    q_value = q_val
    avail_actions_ind = np.nonzero(avail_action)[0]
    avail_action = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
    q_value[avail_action == 0.0] = -1* float("inf")
    maximum = torch.max(q_value)
    q_value[avail_action == 0.0] = float("inf")
    minimum = torch.min(q_value)
    diff = maximum - minimum 
    #print("Difference =", diff)
    #print("Available actions: {}".format(avail_action))
    if diff > threshold:
      action = torch.argmin(q_value)
      attacked = 1
      if action == 0:
        attacked = 0
    else:
      q_value[avail_action == 0.0] = -1* float("inf")
      action = torch.argmax(q_value)
      attacked = 0
    return action, diff, attacked



  # def strategic_attack(self, agents, env):
  #   self.env = env
  #   self.agents = agents


  
  
    

 

