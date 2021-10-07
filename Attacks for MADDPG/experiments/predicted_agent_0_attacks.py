import tensorflow as tf
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.envs.multiagent.make_coop_env import *
import maddpg_implementation.maddpg.common.tf_util as U
from maddpg_implementation.experiments.test import get_trainers
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sandbox.rocky.tf.envs.base import TfEnv
import time
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_maddpg_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="coop_nav", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./weights_new/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="../maddpg_implementation/Good_agent_weights/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=-1,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./data/oop_nav_attack/",
                        help="directory where plot data is saved")

    parser.add_argument("--att-benchmark-dir", type=str, default="Attack/benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--att-plots-dir", type=str, default="Attack/learning_curves/",
                        help="directory where plot data is saved")

    return parser.parse_args()


def get_logits(model, obses, in_length):
    obs = np.reshape(obses, [1, 3, in_length])
    logits = np.asarray(model.predict_on_batch(obs.astype('float64')))
    return logits

def build_model(scope, fname=None):
    with tf.variable_scope(scope):
        # build functional model
        visible = Input(shape=(3, 10))
        hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
        hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)

        hidden_final = LSTM(10, name='leftBranch')(hidden2)
        agent0 = Dense(5, activation='softmax', name='agent1classifier')(hidden_final)


        model = Model(inputs=visible, outputs=agent0)

        model.compile(optimizer='adam',
                      loss={'agent1classifier': 'categorical_crossentropy'},
                      metrics={'agent1classifier': ['acc']})

        model.summary()

        if fname is not None:
            model.load_weights(fname)

    return model


def attack(arglist, threshold=1, random=True, attack_rate = 1, test=False, black_box=True):
    """
    RUN THIS FOR RANDOM AND STRATEGICALLY TIMED
    """

    tf.reset_default_graph()

    scenario = arglist.scenario #Either simple_adversary or simple_spread
    env = make_env(scenario, benchmark=True)

    marl_sess = U.single_threaded_session()

    attacking = True
    timed_attack = random #False is random, true is strategic
    attack_threshold = threshold #rate of attacking for random, or c threshold for strategic

    with marl_sess as sess:

        act_model = build_model("Prediction")
        U.initialize()
        if scenario == 'simple_spread':
            act_model.load_weights("Prediction_weights/agent0_policy_predictor")
        else:
            act_model.load_weights("Prediction_weights/adv_agent1_policy_predictor")

        obs_shape_n = [env.all_obs_space[i].shape for i in range(env.n)]
        num_adversaries = 1
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        trainer_vars = []
        for i in range(env.n):
            trainer_vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_%d" % i))
        trainer_vars = np.asarray(trainer_vars)
        trainer_vars = trainer_vars.flatten()
        trainer_vars = list(trainer_vars)
        trainer_saver = tf.train.Saver(trainer_vars)
        if scenario == 'simple_spread':
            U.load_state(arglist.load_dir + "nav_policy", saver=trainer_saver)
            controlled_agent = 0
        else:
            U.load_state(arglist.load_dir + "adv_policy", saver=trainer_saver)
            controlled_agent = 1

        env = TfEnv(env)

        episode_rewards = [0.0]
        adv_rewards = [0.0]
        coop_rewards = [0.0]
        iter_rewards = []
        train_step = 0
        episode_step = 0

        obs = env.reset()

        episode_obs = []
        episode_attacks = 0
        avg_episode_attacks = []

        collisions = 0
        min_dists = 0
        occ_landmarks = 0
        avg_collisions = []
        avg_min_dists = []
        avg_occ_landmarks = []

        adv_dist = 0
        coop_dist = 0
        avg_adv_dist = []
        avg_coop_dist = []

        label = 0

        transition = []

        while True:
            label = 0

            episode_obs.append(obs[controlled_agent])

            probs = [agent.action(o) for (o, agent) in zip(obs, trainers)]

            actions = []

            for j in range(len(trainers)):
                actions.append(np.random.choice(np.arange(len(probs[0])), p=probs[j]))

            if episode_step > 2:

                if black_box:
                    agent0_probs = np.squeeze(get_logits(act_model, episode_obs[-3:], 10))
                else:
                    agent0_probs = probs[0]

                actions[controlled_agent] = np.random.choice(np.arange(len(agent0_probs)), p=agent0_probs)

                if attacking:
                    if not random:
                        c = np.max(agent0_probs) - np.min(agent0_probs)
                        if c > attack_threshold:
                            actions[controlled_agent] = np.argmin(agent0_probs)
                            episode_attacks += 1
                            label = 1
                    else:
                        if np.random.random() < attack_threshold:
                            actions[controlled_agent] = np.argmin(agent0_probs)
                            episode_attacks += 1
                            label = 1





            new_obs, rew, done, info_n = env.step(actions)

            o = np.concatenate(obs).ravel()
            o_next = np.concatenate(new_obs).ravel()
            o = np.reshape(o, [1, -1])
            o_next = np.reshape(o_next, [1, -1])

            # transition.append((o, a_n[controlled_agent], o_next, label))
            transition.append((o, actions[0], actions[1], actions[2], o_next, label))



            if scenario == 'simple_spread':
                collisions += max([info_n['n'][0][1], info_n['n'][1][1], info_n['n'][2][1]]) - 1
                min_dists += info_n['n'][0][2]
                occ_landmarks += info_n['n'][0][3]

            else:
                adv_dist += info_n['n'][0]
                coop_dist += min(info_n['n'][1][2], info_n['n'][2][2])

            if scenario == 'simple_adversary':
                for i, r in enumerate(rew):
                    if i == 0:
                        adv_rewards[-1] += r
                    else:
                        coop_rewards[-1] += r
                    episode_rewards[-1] += r
            else:
                episode_rewards[-1] += rew[0]


            episode_step += 1
            train_step += 1
            done = all(done)
            terminal = (episode_step >= arglist.max_episode_len)


            obs = new_obs

            if done or terminal:
                episode_obs = []
                avg_episode_attacks.append(episode_attacks/22)
                episode_attacks = 0

                avg_collisions.append(collisions / arglist.max_episode_len)
                avg_min_dists.append(min_dists / arglist.max_episode_len)
                avg_occ_landmarks.append(occ_landmarks / arglist.max_episode_len)

                avg_adv_dist.append(adv_dist / arglist.max_episode_len)
                avg_coop_dist.append(coop_dist / arglist.max_episode_len)

                collisions = 0
                min_dists = 0
                occ_landmarks = 0

                adv_dist = 0
                coop_dist = 0

                label = 0

                obs = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                adv_rewards.append(0)
                coop_rewards.append(0)



            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            if terminal and len(episode_rewards) % arglist.save_rate == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}".format(
                    train_step, len(episode_rewards),
                    sum(episode_rewards[-arglist.save_rate - 1:-1]) / len(episode_rewards[-arglist.save_rate - 1:-1]))
                )
                iter_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))


            if len(episode_rewards) > arglist.num_episodes:
                break

        """
        print("Saving transition...")
        transition = np.asarray(transition)
        print("Transition shape: {}".format(transition.shape))
        if not random:
            string = "timed"
        else:
            string = "random"


        if test:
            np.save("phys_decept_blackbox_{}_{}_test".format(string, int(attack_rate * 100)), np.asarray(transition))
        else:
            np.save("phys_decept_blackbox_{}_{}".format(string, int(attack_rate * 100)), np.asarray(transition))
        """

        if arglist.scenario == 'simple_adversary':
            good_agent_rewards = (sum(coop_rewards) / len(coop_rewards))
            print("AVERAGE GOOD AGENT REWARD: {}".format(good_agent_rewards))
            print("AVERAGE ADV AGENT REWARD: {}".format(sum(adv_rewards) / len(adv_rewards)))
        else:
            print("AVERAGE REWARD: {}".format(sum(episode_rewards) / len(episode_rewards)))
        print("ATTACK RATE: {}".format(sum(avg_episode_attacks) / len(avg_episode_attacks)))

        if arglist.scenario == 'simple_spread':
            print("Average collisions: {}".format(sum(avg_collisions) / len(avg_collisions)))
            print("Average total min dist to targets: {}".format(sum(avg_min_dists) / len(avg_min_dists)))
            print("Average Occupied Landmarks: {}".format(sum(avg_occ_landmarks) / len(avg_occ_landmarks)))
        else:
            print("Average Adv Dist to Target: {}".format(sum(avg_adv_dist) / len(avg_adv_dist)))
            print("Average Best Coop Dist to Target: {}".format(sum(avg_coop_dist) / len(avg_coop_dist)))

        #print("Transition positive rate: {}".format(np.sum(transition[:, 5]) / (22 * arglist.num_episodes)))


if __name__ == "__main__":
    arglist = parse_maddpg_args()
    attack(arglist)
