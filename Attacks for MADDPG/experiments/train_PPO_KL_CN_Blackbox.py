
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

from PPO import PPO

from scipy.special import softmax
from scipy.special import kl_div


# global variables for PPO

#####################  hyper parameters  ####################



GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 5
C_UPDATE_STEPS = 5
S_DIM, A_DIM = 23, 5
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


###############################  PPO  ####################################



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
    parser.add_argument("--save-dir", type=str, default="./weights_CN_save/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=25, help="save model once every time this many episodes are completed") #originally 1000
    parser.add_argument("--load-dir", type=str, default="./weights_CN_final/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def kl_divergence(p, q):
    epsilon = 0.0000001
    p = p + epsilon
    q = q + epsilon
    ans = sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))
    return ans

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
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def get_logits(model, obses, in_length, actions=None, model_type="other_policy"):
    if model_type == "other_policy":
        obs = np.reshape(obses, [-1, 3, in_length])
        action = np.reshape(actions, [-1, 3, 1])
        model_input = np.concatenate([obs, action], axis=2)
        logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
        return logits
    elif model_type == "transition":
        obs = np.reshape(obses, [-1, 3, in_length])   ## why this is 3 * 28
        action = np.reshape(actions, [-1, 3, 3])      ## why this is 3 * 3
        model_input = np.concatenate([obs, action], axis=2)
        logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
        return logits
    elif model_type == "policy":
        obs = np.reshape(obses, [-1, 3, in_length])
        logits = np.asarray(model.predict_on_batch(obs.astype('float64')))
        return logits
    else:
        raise NotImplementedError

def build_model(scope, in_length, fname=None, model_type="other_policy"):
    if model_type == "other_policy":
        with tf.variable_scope(scope):
            # build functional model
            visible = Input(shape=(3, in_length))
            hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
            hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)
            # left branch decides second agent action
            hiddenLeft = LSTM(10, name='leftBranch')(hidden2)
            agent2 = Dense(5, activation='softmax', name='agent2classifier')(hiddenLeft)
            # right branch decides third agent action
            hiddenRight = LSTM(10, name='rightBranch')(hidden2)
            agent3 = Dense(5, activation='softmax', name='agent3classifier')(hiddenRight)

            model = Model(inputs=visible, outputs=[agent2, agent3])

            model.compile(optimizer='adam',
                          loss={'agent2classifier': 'categorical_crossentropy',
                                'agent3classifier': 'categorical_crossentropy'},
                          metrics={'agent2classifier': ['acc'],
                                   'agent3classifier': ['acc']})

            model.summary()

            U.initialize()

            if fname is not None:
                model.load_weights(fname)

        return model

    elif model_type == "transition":
        visible = Input(shape=(3, in_length))
        hidden1 = LSTM(100, return_sequences=True)(visible)
        hidden2 = LSTM(64, return_sequences=True)(hidden1)
        hiddenObservation = LSTM(64, name='observationBranch')(hidden2)
        observation = Dense(in_length-3, name='observationScalar')(hiddenObservation)

        # model = Model(inputs=visible,outputs=[agent1,agent2,agent3,observation])
        model = Model(inputs=visible, outputs=observation)
        model.compile(optimizer='adam',
                      loss={'observationScalar': 'mse'},
                      metrics={'observationScalar': ['mae']})

        model.summary()

        U.initialize()

        if fname is not None:
            model.load_weights(fname)

        return model

    elif model_type == "policy":

        with tf.variable_scope(scope):
            # build functional model
            visible = Input(shape=(3, in_length))
            hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
            hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)

            hidden_final = LSTM(10, name='leftBranch')(hidden2)
            agent0 = Dense(5, activation='softmax', name='agent0classifier')(hidden_final)

            model = Model(inputs=visible, outputs=agent0)

            model.compile(optimizer='adam',
                          loss={'agent0classifier': 'categorical_crossentropy'},
                          metrics={'agent0classifier': ['acc']})

            model.summary()

            U.initialize()

            if fname is not None:
                model.load_weights(fname)

        return model

    else:
        raise NotImplementedError


def train(arglist):
    # with U.single_threaded_session() as marl_sess:
    tf.reset_default_graph()
    # Create environment

    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    marl_sess = U.single_threaded_session()
    
    in_length = 54
    cor_agent_in_length = 18


    other_act_model_fname = "Prediction_weights/actionMultiClassNetwork"
    transition_model_fname = "Prediction_weights/StateTransitionNetwork"
    corrupted_act_model_fname = "Prediction_weights/agent0_policy_predictor"
        

    with marl_sess as sess:

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        
        
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
        

        loaded = True  # load previously trained weight
        attack_rate = 1  # attack threshold
        occ_landmarks = 0
        avg_occ_landmarks = []

        counter = 0   # counter to help final average
        num_attack = 0  # serve as a double-checking index


        ppo = PPO() #PPO policy here
        transition_model = build_model("Transition", in_length + 3, fname=transition_model_fname, model_type="transition")
        U.initialize()
        ppo_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PPO')
        saver_PPO = tf.train.Saver(ppo_vars)
        if loaded:
            print("loading-------------------------------------------------------------")
            saver_PPO.restore(marl_sess, './bb_weights_PPO_KL_CN_final/')
        
        episode_step = 0
        train_step = 0
        t_start = time.time()

        

        PPO_obs = obs_n[2]  # (18,)
        PPO_obs = np.append(PPO_obs, np.zeros((5,)))
        PPO_rew = 0
        PPO_ep_rw = 0
        ep_obs = []
        ep_act = []
        reward_good = [0.0]
        buffer_s, buffer_a, buffer_r = [], [], []

        transition = []
        label = 0


        recorded_sum = 0

        print('Starting iterations...')
        while True:
            label = 0
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]   # (3,5), three agent, each with a probability distribution of action #NONE, UP, DOWN, LEFT, RIGHT

            
            PPO_act = ppo.choose_action(PPO_obs)
            PPO_act = softmax(PPO_act)
            attack_action_n = copy.deepcopy(action_n)
            if np.random.random() < attack_rate:
                attack_action_n[2] = PPO_act  # compromised agent attack

        


            if np.random.random() < attack_rate:
                #attack
                num_attack += 1
                label = 1
                new_obs_n, rew_n, done_n, info_n = env.step(attack_action_n)
            else:
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            occ_landmarks += info_n['n'][0][3]

            action_taken_0 = np.random.choice(5, p=action_n[0])
            action_taken_1 = np.random.choice(5, p=action_n[1])
            action_taken_2 = np.random.choice(5, p=action_n[2])
            action_to_simulate = np.array([action_taken_0, action_taken_1, action_taken_2])
            action_to_simulate = np.array(action_to_simulate)
            action_to_simulate = np.reshape(action_to_simulate,(1,-1))
            #print(action_to_simulate.shape)
            obs_n = np.array(obs_n)
            obs_n_all = np.concatenate((obs_n[0],obs_n[1],obs_n[2])).reshape(1,-1)
            #print(obs_n_all.shape)
            ep_obs.append(obs_n_all)
            ep_act.append(action_to_simulate)

            if episode_step > 2:
                MADDPG_new_obs_n = get_logits(transition_model, ep_obs[-3:], in_length, ep_act[-3:], model_type="transition")


                
                # shape from transition model is (18, 18, 18)
                MADDPG_agent0_obs = MADDPG_new_obs_n[-1,0:18]
                MADDPG_agent1_obs = MADDPG_new_obs_n[-1,18:36]
                MADDPG_agent2_obs = MADDPG_new_obs_n[-1,36:54]
                MADDPG_new_obs_n = [MADDPG_agent0_obs, MADDPG_agent1_obs, MADDPG_agent2_obs]
                # get MADDPG_new_obs_n


            

                MADDPG_next_action_n = [agent.action(obs) for agent, obs in zip(trainers, MADDPG_new_obs_n)]
                MADDPG_next_action_n_atk = [agent.action(obs) for agent, obs in zip(trainers, new_obs_n)]

                # KL_0 = sum(kl_div(MADDPG_next_action_n[0], MADDPG_next_action_n_atk[0]))
                KL_1 = kl_divergence(MADDPG_next_action_n[1], MADDPG_next_action_n_atk[1])


            #***************************************************************

                PPO_rew = KL_1 * 10
                if train_step % (arglist.max_episode_len - 1) == 0:
                    PPO_rew += 80 * -reward_good[-1]
            else:
                PPO_rew = 0.001

            PPO_new_obs = np.append(new_obs_n[2], attack_action_n[1])
            buffer_s.append(PPO_obs)
            buffer_a.append(PPO_act)
            buffer_r.append(PPO_rew)

            
            # transition appending
            o = np.concatenate(obs_n).ravel()
            o_next = np.concatenate(new_obs_n).ravel()
            o = np.reshape(o, [1, -1])
            o_next = np.reshape(o_next, [1, -1])
            action_0 = np.random.choice(5, p=action_n[0])
            action_1 = np.random.choice(5, p=action_n[1])
            action_2 = np.random.choice(5, p=action_n[2])

            # transition.append((o, a_n[controlled_agent], o_next, label))
            transition.append((o, action_0, action_1, action_2, o_next, label))  



            """
            agent 0 = (8,) 
            agent 1 = (10,)
            agent 2 - (10,)

            state_shape = (28)
            actions (5,)

            input o, action2(victm agent) o_next, reward   [list, int, list, int]

            AIRL: input o, it output action 0

            trainsition network: INPUT: o, OUTPUT: o_next, reward
            reward network


            1000000 10^6, 8GB
            """


            reward_good[-1] += rew_n[0]
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            
            PPO_obs = PPO_new_obs
            PPO_ep_rw += PPO_rew 
            if not loaded:
                if (episode_step+1) % BATCH == 0 or episode_step == arglist.max_episode_len - 1:
                    v_s_ = ppo.get_v(PPO_new_obs)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br)
                

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                reward_good.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                # buffer_s, buffer_a, buffer_r = [], [], []
                avg_occ_landmarks.append(occ_landmarks / arglist.max_episode_len)
                occ_landmarks = 0

            # increment global step counter
            train_step += 1
            """
            # for benchmarking learned policies
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
                
                print("episode reward for PPO agent: {}".format(PPO_ep_rw/arglist.save_rate))
                #wandb.log({'Reward for PPO': PPO_ep_rw/arglist.save_rate})
                PPO_ep_rw = 0
                if not loaded:
                    saver_PPO.save(marl_sess, './bb_weights_PPO_KL_CN_save/')
                
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    print("occ land avg {}".format(sum(avg_occ_landmarks) / len(avg_occ_landmarks)))
                    print("good reward is {}".format(np.mean(reward_good[-arglist.save_rate:])))
                    recorded_sum += np.mean(reward_good[-arglist.save_rate:])
                    counter += 1
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                recorded_sum = recorded_sum / counter
                print("final ans for attack rate " + str(attack_rate) + " is " + str(recorded_sum))
                print("number of attack is " + str(num_attack))

                print("Saving final transitions")
                print(np.array(transition).shape)
                np.save('CN_CF_BB_100_train', transition)



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
