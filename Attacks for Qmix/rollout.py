import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from numpy import save
import sys
from scipy.special import softmax

sys.path.insert(0, "/root/qmix")
import tensorflow as tf
from ddpg import DDPG
import copy

import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf
from algo import Discriminator, AIRL_wrapper

#####################  hyper parameters  ####################


LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


def kl_divergence(p, q):
    epsilon = 0.0000001
    p = p + epsilon
    q = q + epsilon
    ans = sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)))
    return ans


# specifically for blackbox counterfactual RL
def build_model(scope, fname=None):
    with tf.variable_scope(scope):
        visible = Input(shape=(3, 93))
        hidden1 = LSTM(128, return_sequences=True, name='firstLSTMLayer')(visible)
        hidden2 = LSTM(64, name='secondLSTMLayer')(hidden1)
        hidden3 = Dense(64, name='firstDenseLayer')(hidden2)
        output = Dense(90)(hidden3)
        model = Model(inputs=visible, outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        # print(model.summary())

        if fname is not None:
            model.load_weights(fname)

    return model


def get_logits(model, obs, action):
    obs = np.reshape(obs, [-1, 3, 90])
    action = np.reshape(action, [-1, 3, 3])
    model_input = np.concatenate([obs, action], axis=2)
    logits = np.asarray(model.predict(model_input.astype('float64'), verbose=0))
    return logits


###############################  DDPG  ####################################

class RolloutWorker:
    def __init__(self, env, agents, args, adversarial, simu_env):
        self.env = env
        self.simu_env = simu_env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.adversarial = adversarial

        print('Successful Init of RolloutWorker')

    # ===============>entry 1

    def generate_transition(self, obs, actions, next_obs, attacked, valid_step, able_to_attack):

        obs = np.asarray(obs).flatten()
        next_obs = np.asarray(next_obs).flatten()
        actions = np.asarray(actions).flatten()
        # print(obs.shape)
        # print(actions.shape)
        # print(next_obs.shape)
        transition = (obs, actions[0], actions[1], actions[2], next_obs, attacked, valid_step, able_to_attack)
        return transition

    def generate_episode(self, episode_num=None, evaluate=False, black_box=True, bb_net=None, total_timestep = 0):
        tf.reset_default_graph()
        if black_box:
            print("This is black box")
        if black_box and not self.args.attack_name == 'counterfactual_RL' and not self.args.attack_name == 'inverse_glob':
            assert bb_net is not None, "Specify black box network"

        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        a, adv_o, adv_action, threshold = [], [], [], []
        cor_obs = []
        all_obs = []


        self.env.reset()
        if self.args.attack_name == 'counterfactual_RL' and self.args.white_box:
            self.simu_env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards

        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        not_attack_last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        self.agents.policy.init_hidden(1)

        transitions = []

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        s_dim = 32  # agent 0's observation with agent 1 and 2's action
        a_dim = 9
        a_bound = 1
        ddpg = DDPG(a_dim, s_dim, a_bound)

        DDPG_obs = np.zeros(32)
        DDPG_rew = 0

        attack_rate = 0.25

        # AIRL for zero-sum blackbox
        D = Discriminator("save_model_AIRL", (30,), 9)  # initialize the AIRL discriminator

        expert_observations = np.genfromtxt('SC_Obs_Good.txt')
        expert_actions = np.genfromtxt('SC_Act_Good_dist.txt', dtype=np.int32)

        obs_his = []
        act_his = []

        ep_obs_his = []
        ep_act_his = []



        collect_AIRL = False

        if collect_AIRL:
            file_obs_good = open("SC_Obs_Good.txt", "a")
            file_act_good_disc = open("SC_Act_Good_dist.txt", "a")

        transit_obs = []
        transit_act = []

        transition_fname = "Prediction_models/transition_tf1_weights/"
        # create prediction model for counterfactual_RL in blackbox
        transition_model = build_model("Transition", fname=transition_fname)

        # main loop for training
        while not terminated and step < self.episode_limit:

            obs = self.env.get_obs()  # (3, 30) - agent observations
            DDPG_obs[:30] = obs[2]
            state = self.env.get_state()  # (48,) Global state - should not be passed during execution

            all_obs.append(np.reshape(np.asarray(obs), [1, 90]))  # (1,90) by concatinate three agent's observation

            attacked = 0
            able_to_attack = 1
            other_actions = []

            actions, avail_actions, actions_onehot = [], [], []
            not_attack_actions, not_attack_avail_actions, not_attack_actions_onehot = [], [], []

            if self.args.attack_name == 'counterfactual_RL' and self.args.white_box:
                # update the world state
                self.simu_env.update_world(self.env)

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)

                if self.args.attack_name == 'counterfactual_RL' and self.args.white_box:
                    simu_avail_action = self.simu_env.get_avail_agent_actions(agent_id)

                # victim_agent = 2
                if self.args.victim_agent == agent_id:
                    cor_obs.append(obs[agent_id])
                    # print(np.shape(cor_obs))

                # start to attack our victim agent
                if self.args.victim_agent == agent_id and self.args.adversary:

                    if self.args.attack_name == "random" and np.random.uniform() <= self.args.attack_rate:
                        if black_box:
                            pass

                        # white box randomly timed
                        else:
                            action = self.adversarial.random_attack(obs[agent_id], last_action[agent_id], agent_id,
                                                                    avail_action, epsilon, evaluate)
                        attacked = 1

                    # randomly timed
                    elif self.args.attack_name == "random_time" and np.random.uniform() <= self.args.attack_rate:
                        # self.adversarial.policy.init_hidden(1)
                        # black box
                        if black_box:
                            if step < 2:
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                                attacked = 0
                                able_to_attack = 0
                            else:

                                input_obs = np.asarray(cor_obs[-3:])
                                input_obs = np.expand_dims(input_obs, axis=0)
                                probs = bb_net.predict_on_batch(input_obs)
                                avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                                probs[avail == 0.0] = float("inf")
                                action = np.argmin(probs)
                                able_to_attack = 1
                                attacked = 1
                        # whitebox
                        else:
                            q_val = self.agents.get_qvalue(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                           epsilon, evaluate)
                            action = self.adversarial.random_time_attack(q_val, avail_action)
                            attacked = 1

                    # strategic
                    elif self.args.attack_name == "strategic":

                        # blackbox
                        if black_box:
                            if step < 2:
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                                attacked = 0
                                able_to_attack = 0
                            else:
                                input_obs = np.asarray(cor_obs[-3:])
                                input_obs = np.expand_dims(input_obs, axis=0)
                                probs = bb_net.predict_on_batch(input_obs)
                                avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                                probs[avail == 0.0] = -1 * float("inf")
                                maximum = np.max(probs)
                                probs[avail == 0.0] = float("inf")
                                minimum = np.min(probs)
                                c = maximum - minimum
                                if c > self.args.strategic_threshold:
                                    action = np.argmin(probs)
                                    attacked = 1
                                else:
                                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                       avail_action, epsilon, evaluate)
                                    attacked = 0
                                able_to_attack = 1
                        # whitebox
                        else:
                            demo_thrs = self.args.strategic_threshold
                            q_val = self.agents.get_qvalue(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                           epsilon, evaluate)
                            action, diff, attacked = self.adversarial.strategic_time_attack(q_val, avail_action,
                                                                                            epsilon,
                                                                                            demo_thrs)
                            threshold.append(diff)
                            if action == 0:
                                able_to_attack = 0
                    # zero-sum
                    elif self.args.attack_name == 'inverse_glob':
                        DDPG_act = ddpg.choose_action(DDPG_obs)
                        DDPG_act = softmax(DDPG_act)
                        avail_to_choose = []
                        for i in range(9):
                            if avail_action[i] == 1:
                                avail_to_choose.append(DDPG_act[i])
                            else:
                                avail_to_choose.append(0.0)

                        if np.random.random() < attack_rate:
                            action = np.argmax(avail_to_choose)
                            attacked = 1
                            able_to_attack = 1
                        else:
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)
                            attacked = 0
                            able_to_attack = 1
                        if collect_AIRL:
                            for item in obs[1]:
                                file_obs_good.write("{} ".format(item))
                            file_obs_good.write("\n")
                            file_act_good_disc.write("{} \n".format(action))

                    elif self.args.attack_name == 'counterfactual_RL':
                        if black_box:
                            """
                          At time t:
                          1. Output agent 0, 1, 2's action from Qmix's policy, named not_attack_action
                          2. Output agent 2's action from DDPG's policy, assign it to action_n[2] to make it action
                          3. Feed not_attack_action in prediction network, get next state w/o attack
                          4. Feed action in the env's step function, get next state w/ attack
                          5. Feed both version of next_state into Qmix's optimal policy to get the q_value array (action should be argmax)
                          6. Normalize the q_value
                          7. Compute KL divergence among, assign it to DDPG's reward
                          8. do regular training as other methods
                          """
                            DDPG_act = ddpg.choose_action(DDPG_obs)
                            DDPG_act = softmax(DDPG_act)  # 9个random 数字
                            avail_to_choose = []
                            for i in range(9):
                                if avail_action[i] == 1:
                                    avail_to_choose.append(DDPG_act[i])
                                else:
                                    avail_to_choose.append(0.0)

                            if np.random.random() < attack_rate:
                                attacked = 1
                                able_to_attack = 1
                                action = np.argmax(avail_to_choose)  # DDPG's action
                                not_attack_action = self.agents.choose_action(obs[agent_id], last_action[agent_id],
                                                                              agent_id,
                                                                              avail_action, epsilon, evaluate)
                            else:  # not attack
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                                not_attack_action = action
                                attacked = 0
                                able_to_attack = 1

                        else:  # white box counterfactual RL
                            DDPG_act = ddpg.choose_action(DDPG_obs)
                            DDPG_act = softmax(DDPG_act)  # 9个random 数字
                            avail_to_choose = []
                            for i in range(9):
                                if avail_action[i] == 1:
                                    avail_to_choose.append(DDPG_act[i])
                                else:
                                    avail_to_choose.append(0.0)

                            if np.random.random() < attack_rate:
                                action = np.argmax(avail_to_choose)
                                attacked = 1
                            else:
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                                attacked = 0
                            able_to_attack = 1

                            not_attack_action = self.agents.choose_action(obs[agent_id], last_action[agent_id],
                                                                          agent_id,
                                                                          simu_avail_action, epsilon, evaluate)
                    #############################################################################################
                    # optimal attack
                    else:
                        action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                           avail_action, epsilon, evaluate)
                # not attack or not attacking the first one
                else:
                    if self.args.white_box:
                        action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                           avail_action, epsilon, evaluate)
                        if self.args.attack_name == 'counterfactual_RL':
                            not_attack_action = self.agents.choose_action(obs[agent_id], last_action[agent_id],
                                                                          agent_id, simu_avail_action, epsilon,
                                                                          evaluate)
                    # blackbox
                    else:
                        if self.args.attack_name == 'counterfactual_RL':
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)
                            not_attack_action = action

                        elif self.args.victim_agent == agent_id:
                            if step < 2:
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                            else:
                                cor_obs.append(obs[agent_id])
                                input_obs = np.asarray(cor_obs[-3:])
                                input_obs = np.expand_dims(input_obs, axis=0)
                                probs = bb_net.predict_on_batch(input_obs)
                                avail = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)
                                probs[avail == 0.0] = -1 * float("inf")
                                action = np.argmax(probs)
                        else:
                            action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                               avail_action, epsilon, evaluate)
                if self.args.attack_name == 'counterfactual_RL':
                    not_attack_actions.append(not_attack_action)
                    # print("not attack action is " + str(not_attack_actions))
                    not_attack_action_onehot = np.zeros(self.args.n_actions)
                    not_attack_action_onehot[not_attack_action] = 1
                    not_attack_actions_onehot.append(not_attack_action_onehot)
                    not_attack_last_action[agent_id] = not_attack_action_onehot
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                if self.args.victim_agent != agent_id and self.args.adversary:
                    other_actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
                if agent_id == self.args.victim_agent:
                    adv_o.append(obs[agent_id])
                    adv_action.append(action)
            if self.args.attack_name == 'counterfactual_RL':
                if black_box:
                    reward, terminated, info = self.env.step(actions)

                    if step < 3:
                        obs_n_all = np.concatenate((obs[0], obs[1], obs[2])).reshape(1, -1)
                        action_n_all = np.array(actions).reshape((1, -1))
                        transit_obs.append(obs_n_all)
                        transit_act.append(action_n_all)
                    else:
                        not_attack_next_obs = get_logits(transition_model, transit_obs[-3:], transit_act[-3:])
                        not_attack_next_obs = np.reshape(not_attack_next_obs, (3, 30))
                else:  # whitebox
                    _, _, _ = self.simu_env.step(not_attack_actions)
                    reward, terminated, info = self.env.step(actions)
            else:
                reward, terminated, info = self.env.step(actions)

            if action == 0:
                able_to_attack = 0
                attacked = 0

            # negative global reward attack
            if self.args.attack_name == 'inverse_glob':
                if self.args.white_box:
                    DDPG_rew = -reward
                else: #black box, using AIRL
                    AIRL_act = actions[2]

                    DDPG_rew = D.get_rewards(obs[2].reshape((1, -1)), AIRL_act.reshape((-1, 1)))
                    DDPG_rew = np.squeeze(DDPG_rew)
                    DDPG_rew *= -2
                    ep_obs_his.append(np.squeeze(obs[2]))
                    ep_act_his.append(AIRL_act)

            elif self.args.attack_name == 'counterfactual_RL' and not black_box:
                next_obs = self.env.get_obs()
                not_attack_next_obs = self.simu_env.get_obs()
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    not_attack_avail_action = self.simu_env.get_avail_agent_actions(agent_id)

                    next_action = self.agents.get_qvalue(next_obs[agent_id], last_action[agent_id], agent_id,
                                                         avail_action, epsilon, evaluate)

                    not_attack_next_action = self.agents.get_qvalue(not_attack_next_obs[agent_id],
                                                                    not_attack_last_action[agent_id], agent_id,
                                                                    not_attack_avail_action, epsilon, evaluate)
                    next_action.detach().numpy()
                    not_attack_next_action.detach().numpy()
                    p = np.zeros(9)
                    q = np.zeros(9)
                    for i in range(9):
                        if next_action[0, i] < -10000:
                            next_action[0, i] = 0.0001
                        p[i] = next_action[0, i]
                        if not_attack_next_action[0, i] < -10000:  # -inf:
                            not_attack_next_action[0, i] = 0.0001
                        q[i] = not_attack_next_action[0, i]
                    p_max = np.max(p)
                    p_min = np.min(p)
                    q_max = np.max(q)
                    q_min = np.min(q)
                    p = (p - p_min) / (p_max - p_min)
                    q = (q - q_min) / (q_max - q_min)
                    DDPG_rew = kl_divergence(p, q)
            elif self.args.attack_name == 'counterfactual_RL' and black_box:
                next_obs = self.env.get_obs()
                if step < 3:
                    not_attack_next_obs = next_obs
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    not_attack_avail_action = avail_action
                    next_action = self.agents.get_qvalue(next_obs[agent_id], last_action[agent_id], agent_id,
                                                         avail_action, epsilon, evaluate)
                    # print(np.array(not_attack_next_obs).shape)
                    not_attack_next_action = self.agents.get_qvalue(not_attack_next_obs[agent_id],
                                                                    not_attack_last_action[agent_id], agent_id,
                                                                    not_attack_avail_action, epsilon, evaluate)
                    next_action.detach().numpy()
                    not_attack_next_action.detach().numpy()
                    p = np.zeros(9)
                    q = np.zeros(9)
                    for i in range(9):
                        if next_action[0, i] < -10000:
                            next_action[0, i] = 0.0001
                        p[i] = next_action[0, i]
                        if not_attack_next_action[0, i] < -10000:  # -inf:
                            not_attack_next_action[0, i] = 0.0001
                        q[i] = not_attack_next_action[0, i]
                    p_max = np.max(p)
                    p_min = np.min(p)
                    q_max = np.max(q)
                    q_min = np.min(q)
                    p = (p - p_min) / (p_max - p_min)
                    q = (q - q_min) / (q_max - q_min)
                    DDPG_rew = kl_divergence(p, q)
            else:
                pass
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            a.append(actions)

            if step < 25:
                valid_step = 1
                next_obs = self.env.get_obs()
                DDPG_new_obs = np.append(next_obs[2], other_actions)
                transition = self.generate_transition(obs, actions, next_obs, attacked, valid_step, able_to_attack)
                transitions.append(transition)
                if self.args.attack_name == 'counterfactual_RL' or self.args.attack_name == 'inverse_glob':
                    ddpg.store_transition(DDPG_obs, DDPG_act, DDPG_rew, DDPG_new_obs)
                DDPG_obs = DDPG_new_obs
                if ddpg.pointer > MEMORY_CAPACITY and (total_timestep + step) % 100 == 0:
                    ddpg.learn()
                    _, dloss = D.train(expert_observations, expert_actions, obs_his, act_his)
                    obs_his = obs_his[-MEMORY_CAPACITY:]
                    act_his = act_his[-MEMORY_CAPACITY:]

            if terminated and step < 24:
                valid_step = 0
                able_to_attack = 0
                zero_obs = np.zeros((90,))
                zero_acts = np.zeros((3,))
                attacked = 0
                for i in range(24 - step):
                    transition = self.generate_transition(zero_obs, zero_acts, zero_obs, attacked, valid_step,
                                                          able_to_attack)
                    transitions.append(transition)

            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        total_timestep += step
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            # a.append(np.zeros((self.n_agents, self.n_actions)))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )

        # ===========EPISODE SAVED FOR CAM=======================
        data_set = dict(
            state=s.copy(),
            observation=o.copy(),
            action=a.copy(),
            next_state=s_next.copy(),
            observation_next=o_next.copy(),
        )

        adv_data = dict(
            adv_observation=adv_o.copy(),
            adv_action=adv_action.copy()
        )

        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        transitions = np.asarray(transitions)
        # print(transitions.shape)
        return episode, episode_reward, win_tag, step, data_set, adv_data, threshold, transitions, total_timestep
