import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from algo import Discriminator, AIRL_wrapper

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

from DDPG import DDPG

from scipy.special import softmax


# global variables for DDPG

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.90  # reward discount   originally 0.9
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False


# ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=2000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./weights_CN_save/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=200,
                        help="save model once every time this many episodes are completed")  # originally 1000
    parser.add_argument("--load-dir", type=str, default="./weights_CN_final/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session() as marl_sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        s_dim = 20  # 18 + 2
        a_dim = 5
        a_bound = 1  # upper limit of the action

        var = 3  # control exploration
        t1 = time.time()

        ##################### DDPG #######################

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()

        loaded = True
        attack_rate = 0.05

        collect_AIRL = False

        if collect_AIRL:
            # collect observation for AIRL in black box
            file_obs_good = open("CN_Obs_Good.txt", "w")
            file_act_good_disc = open("CN_Act_Good_dist.txt", "w")

        occ_landmarks = 0
        avg_occ_landmarks = []

        rew_sum = 0
        counter = 0

        D = Discriminator("./disc_saved_model", (18,), 5)  # initialize the AIRL discriminator


        ddpg = DDPG(a_dim, s_dim, a_bound)  # define DDPG policy trainer here
        U.initialize()
        ddpg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor') + tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic')
        saver_DDPG = tf.train.Saver(ddpg_vars)

        if loaded:
            print("loading-------------------------------------------------------------")
            saver_DDPG.restore(marl_sess, './bb_weights_DDPG_Adv_CN_final/')
            # saver_DDPG.restore(marl_sess, './backup/')

        transition = []
        label = 0

        episode_step = 0
        train_step = 0
        t_start = time.time()

        #### DDPG parameters initialization ####

        DDPG_obs = obs_n[2]  # (10,)
        DDPG_obs = np.append(DDPG_obs, np.zeros((2,)))
        DDPG_rew = 0
        DDPG_ep_rw = 0

        reward_good = [0.0]

        expert_observations = np.genfromtxt('AIRL data/CN_Obs_Good.txt')
        expert_actions = np.genfromtxt('AIRL data/CN_Act_Good_dist.txt', dtype=np.int32)

        obs_his = []
        act_his = []

        ep_obs_his = []
        ep_act_his = []

        ep_rew = [0]

        print('Starting iterations...')
        while True:
            label = 0
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,
                                                                obs_n)]  # (3,5), three agent, each with a probability distribution of action #NONE, UP, DOWN, LEFT, RIGHT
            # environment step
            DDPG_act = ddpg.choose_action(DDPG_obs)
            # get DDPG action (agent 3)

            ###### modified MADDPG env with output of DDPG network #####

            DDPG_act = softmax(DDPG_act)
            if np.random.random() < attack_rate:
                action_n[2] = DDPG_act  # compromised agent attack
                label = 1

            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            occ_landmarks += info_n['n'][0][3]


            # DDPG_rew = -rew_n[2] * 2
            reward_good[-1] += rew_n[1]

            AIRL_act = np.argmax(action_n[2])

            DDPG_rew = D.get_rewards(obs_n[2].reshape((1, -1)), AIRL_act.reshape((-1, 1)))
            DDPG_rew = np.squeeze(DDPG_rew)
            DDPG_rew *= -2
            ep_rew[len(episode_rewards)-1] += DDPG_rew
            ep_obs_his.append(np.squeeze(obs_n[2]))
            ep_act_his.append(AIRL_act)

            # DDPG_rew = -(rew_n[1] + rew_n[2])  # version 2

            action_taken_0 = np.argmax(action_n[0])
            action_taken_1 = np.argmax(action_n[1])

            DDPG_new_obs = np.append(new_obs_n[2], [action_taken_0, action_taken_1])
            ddpg.store_transition(DDPG_obs, DDPG_act, DDPG_rew, DDPG_new_obs)

            # transition appending
            o = np.concatenate(obs_n).ravel()
            o_next = np.concatenate(new_obs_n).ravel()
            o = np.reshape(o, [1, -1])
            o_next = np.reshape(o_next, [1, -1])
            action_0 = np.argmax(action_n[0])
            action_1 = np.argmax(action_n[1])
            action_2 = np.argmax(action_n[2])

            # transition.append((o, a_n[controlled_agent], o_next, label))
            transition.append((o, action_0, action_1, action_2, o_next, label))

            if collect_AIRL:
                for item in obs_n[1]: # good agent
                    file_obs_good.write("{} ".format(item))
                file_obs_good.write("\n")
                file_act_good_disc.write("{} \n".format(action_1))

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            if not loaded: # no, it just print in the consol that rew   no the reward nan is the good agent reward
                if ddpg.pointer > MEMORY_CAPACITY and len(episode_rewards) % 100 == 0:
                    ddpg.learn()
                    _, dloss = D.train(expert_observations, expert_actions, obs_his, act_his)
                    obs_his = obs_his[-MEMORY_CAPACITY:]
                    act_his = act_his[-MEMORY_CAPACITY:]

            DDPG_obs = DDPG_new_obs
            DDPG_ep_rw += DDPG_rew

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                ep_rew.append(0)
                obs_his.extend(ep_obs_his)
                act_his.extend(ep_act_his)
                ep_obs_his = []
                ep_act_his = []
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                reward_good.append(0)
                avg_occ_landmarks.append(occ_landmarks / arglist.max_episode_len)
                occ_landmarks = 0



            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            """
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue
            """
            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                print("episode reward for DDPG agent: {}".format(DDPG_ep_rw / arglist.save_rate))
                DDPG_ep_rw = 0
                print("AIRL DDPG bb agent reward: {}".format(np.mean(ep_rew)))
                U.save_state(arglist.save_dir, saver=saver)

                if not loaded:
                    saver_DDPG.save(marl_sess, './bb_weights_DDPG_Adv_CN_save/')

                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                    print("good reward is {}".format(np.mean(reward_good[-arglist.save_rate:])))
                    rew_sum += np.mean(reward_good[-arglist.save_rate:])
                    counter += 1
                    print("Average Occupied Landmarks: {}".format(sum(avg_occ_landmarks) / len(avg_occ_landmarks)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                print("final ans is " + str(rew_sum / counter))

                """
                print("Saving final transitions")
                print(np.array(transition).shape)
                np.save('CN_ZS_WB_100_train', transition)
                """

                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
