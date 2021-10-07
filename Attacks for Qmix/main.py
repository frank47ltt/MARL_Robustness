import sys
sys.path.insert(0,"/Users/jmccalmon/PycharmProjects/qmix")
from runner import Runner
from env.starcraftenv import StarCraft2Env
from arguments import get_common_args, get_mixer_args 
import numpy as np

if __name__ == '__main__':

    args = get_common_args()
    args = get_mixer_args(args)
    env = StarCraft2Env(map_name=args.map_name,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version,
                        replay_dir=args.replay_dir)

    simu_env = StarCraft2Env(map_name=args.map_name,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version,
                        replay_dir=args.replay_dir)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]                              # number of action is 9
    args.n_agents = env_info["n_agents"]                                 # number of agent is 3
    args.state_shape = env_info["state_shape"]                           # state shape 48
    args.obs_shape = env_info["obs_shape"]                              # obs shape 30
    args.episode_limit = env_info["episode_limit"]                       # episode limit 60

    args.alg = "qmix"
    args.attack_rate = 1
    args.counterfactual_threshold = -1
    args.strategic_threshold = -1

    save = False

    args.attack_name = "inverse_glob"

    runner = Runner(env, args, simu_env)                                 # 调用runner类
    win_rate, avg_reward, transitions = runner.evaluate()       # retrun out the answer
    fname = "WB_"

    # if you want to attack (random, stategic, counter)
    if args.adversary:
        print('Evaluated the {} adversary'.format(args.attack_name))
        """
        if i == 2 or i == 7:
            fname = fname + "SC_{}_{}_test".format(args.attack_name, args.attack_rate)
        else:
            fname = fname + "SC_{}_{}".format(args.attack_name, args.attack_rate)
        """

        attack_rate = np.sum(transitions[:, 5]) / np.sum(transitions[:, 7])     # number of attacks / total timestep

        print('Attack rate was {}'.format(attack_rate))
    # if you want to run benchmark optimal policy
    else:
        print('Evaluated the Optimal Policy')
        fname = fname + "optimal"

    print('The win rate of {} is  {}'.format(args.alg, win_rate))
    print('The average reward is {}'.format(avg_reward))
    print('Transition Shape: {}'.format(transitions[:, :7].shape))

    if save:
        print('Saving Transition...')
        np.save("SC_ZS_WB_100_train", transitions[:, :7])







    env.close()


