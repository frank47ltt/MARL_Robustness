import numpy as np
from matplotlib import pyplot 
import os
import sys
sys.path.insert(0,"/root/qmix")

from rollout import RolloutWorker  
from agent import Agents 
from adversary import Adversary
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

from tensorflow import keras


class Runner:
    def __init__(self, env, args, simu_env):
        self.env = env
        self.simu_env = simu_env
        self.agents = Agents(args)
        self.adversarial = Adversary(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args, self.adversarial, simu_env)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.data_set = []
        self.adv_data =[]
        self.episode = []
        self.q_diff = []
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map_name
        self.bb_net = None
        
        if self.args.attack_name == "strategic":
          self.save_path1 = 'Saved_data/result' + '/' + args.alg + '/' + args.map_name+'/attack_data'
          self.save_path1 = self.save_path1+'/'+self.args.attack_name+'/'+'atk_threshold_{}'.format(self.args.strategic_threshold)
        else:
          self.save_path1 = 'Saved_data/result' + '/' + args.alg + '/' + args.map_name+'/attack_data'
          self.save_path1 = self.save_path1+'/'+self.args.attack_name+'/'+'atk_rate_{}'.format(self.args.attack_rate)

        self.save_path2 = 'Saved_data/result' + '/' + args.alg + '/' + args.map_name+'/normal'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            #print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                print('Time_steps {}:'.format(time_steps),"Win rate:", win_rate,"Episode reward:", episode_reward)
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
            episodes = []
         
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps,_,_,_, _ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
           
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
              mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
              self.agents.train(mini_batch, train_steps)
              train_steps += 1
        win_rate, episode_reward, _ = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)


    def sort_thresholds(self):
        self.q_diff = sorted(self.q_diff)
        t1 = self.q_diff[int(.25 * len(self.q_diff))]
        t2 = self.q_diff[int(.5 * len(self.q_diff))]
        t3 = self.q_diff[int(.75 * len(self.q_diff))]
        t4 = self.q_diff[-1]
        return t1, t2, t3, t4


    def evaluate(self):
        # situations of blackbox + attack
        if not self.args.white_box and self.args.adversary:

            # load blackbox weights for prediction purpose
            if self.args.attack_name == "random_time" or self.args.attack_name == "random" or self.args.attack_name == "strategic":
                self.bb_net = keras.models.load_model("Prediction_models/Model_weights/SC_corrupted_agent_predictions.keras")
            elif self.args.attack_name == "counterfactual":

                self.bb_net = [keras.models.load_model("Prediction_models/Model_weights/SC_corrupted_agent_predictions.keras"),
                               keras.models.load_model("Prediction_models/Model_weights/SC_other_agent_predictions.keras"),
                               keras.models.load_model("Prediction_models/Model_weights/SCTransition_model.keras")]

        # situations of blackbox + optimal
        elif not self.args.white_box and not self.args.adversary:
            # load blackbox weights for prediction purpose
            self.bb_net = keras.models.load_model("Prediction_models/Model_weights/SC_corrupted_agent_predictions.keras")


        #declare variables
        transitions = []
        win_number = 0
        episode_rewards = 0

        save_rate = 25
        graphed_reward = 0
        counter = 0

        time_step = 0
        # run episodes
        x = []
        y = []
        for epoch in range(self.args.evaluate_epoch):
            self.episode, episode_reward, win_tag, _, self.data_set, self.adv_data, self.q_diff, transition, curr_step = self.rolloutWorker.generate_episode(epoch, evaluate=True, black_box=not self.args.white_box, bb_net=self.bb_net, total_timestep= time_step)
            time_step = curr_step
            episode_rewards += episode_reward
            graphed_reward += episode_reward
            counter += 1
            transitions.append(transition)
            if (epoch + 1) % self.args.evaluate_epoch == 0:
                print("Epoch {} Finished".format(epoch+1))

            if epoch % save_rate == 0 and epoch != 0:
                print("This is epoch " + str(epoch) + " with a reward of " + str(graphed_reward/counter))
                x.append(epoch)
                y.append(graphed_reward/counter)
                plt.plot(x, y, 'b')
                plt.title('Qmix with inverse global')
                plt.xlabel('Epoch')
                plt.ylabel('Starcraft Average Rewards')
                plt.show()
                graphed_reward = 0
                counter = 0

            if win_tag:
                win_number += 1
        transitions = np.asarray(transitions)
        transitions = np.reshape(transitions, [-1, 8])
        print(transitions.shape)  # (2500, 7)

        # win_rate, avg_reward, transitions = runner.evaluate()
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, transitions
    
  
    def plt(self, num):
        # plt.figure()
        # plt.ylim([0, 120])
        # plt.cla()
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('Win rates')
        # plt.subplot(2, 1, 2)
        # plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('episode_rewards')
        fig = pyplot.figure()
        fig.subplots_adjust(hspace=0.5)
        # fig.ylim([0, 120])
        # fig.cla()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(range(len(self.win_rates)), self.win_rates)
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax1.set_ylabel('Win rates')

        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(range(len(self.episode_rewards)), self.episode_rewards)
        ax2.set_ylim([0, 50])
        ax2.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax2.set_ylabel('Episode Rewards')

        #====================================================
        fig2 = pyplot.figure()
        fig2.subplots_adjust(hspace=0.5)

        """
        ax1 = fig2.add_subplot(3,1,1)
        self.q_diff = self.q_diff.detach().numpy()
        ax1.plot(range(len(np.asarray(self.q_diff))), np.asarray(self.q_diff))
        ax1.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax1.set_ylabel('Max-Min: Qvalue')
        """
        
        ax2 = fig2.add_subplot(3,1,2)
        ax2.plot(range(len(self.win_rates)), self.win_rates)
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax2.set_ylabel('Win rates')

        ax3 = fig2.add_subplot(3,1,3)
        ax3.plot(range(len(self.episode_rewards)), self.episode_rewards)
        ax3.set_ylim([0, 50])
        ax3.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax3.set_ylabel('Episode Rewards')
        #====================================================
        if self.args.adversary:
          if not os.path.exists(self.save_path1):
            os.makedirs(self.save_path1)
          text = 'adversary' 
          fig.savefig(self.save_path1 + '/plt_{}.png'.format(text), format='png')
          np.save(self.save_path1 + '/win_rates_{}'.format(text), self.win_rates)
          np.save(self.save_path1 + '/episode_rewards_{}'.format(text), self.episode_rewards)
          np.save(self.save_path1 + '/data_set', self.data_set)
          np.save(self.save_path1 + '/adv_data', self.adv_data)
          if self.args.attack_name == 'strategic':
            fig2.savefig(self.save_path1 + '/max_min_diff_plt_{}.png', format='png')
            pyplot.close(fig2)
          pyplot.close(fig)
        else:
          if not os.path.exists(self.save_path2):
            os.makedirs(self.save_path2)
          text = 'normal'
          fig.savefig(self.save_path2 + '/plt_{}.png'.format(text), format='png')
          np.save(self.save_path2 + '/episode_data', self.data_set)
          np.save(self.save_path2 + '/win_rates_{}'.format(text), self.win_rates)
          np.save(self.save_path2 + '/episode_rewards_{}'.format(text), self.episode_rewards)
          pyplot.close(fig)
          
           

