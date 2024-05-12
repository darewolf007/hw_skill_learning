import _init_paths
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from dataload.load_imitation_act_data import load_data
from utils.helper import load_config, set_seed, compute_dict_mean, check_and_create_dir, args_overwrite_config
from utils.process_log import plot_history, WandBLogger, AttrDict, setup_logging
from imitation_learning.act_policy import ACTPolicy
from utils.general_utils import detach_dict

def get_args_parser():
    parser = argparse.ArgumentParser('Set base param', add_help=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--yaml_dir', action='store', type=str, help='yaml config', default= "../configs/imitation_act_example.yaml")
    return parser

def main(args):
    is_eval = args.eval
    args_config = load_config(os.path.join(os.path.dirname(__file__), args.yaml_dir))
    args_overwrite_config(args_config)
    set_seed(args_config['seed'])

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(args_config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader = load_data(args_config, os.path.dirname(__file__))

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, args_config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def read_dict_with_twodim_numpy_to_csv(filename):
    import csv
    csv.field_size_limit(1000000)
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        read_data_dict = {key: [] for key in header}
        for row in csv_reader:
            for key, value_str in zip(header, row):
                value = np.array(eval(value_str))
                read_data_dict[key].append(value)
    return read_data_dict

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    state_dim = 9
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    policy_config['num_queries'] = 20
    # policy_config['num_queries'] = 50
    # load policy and stats
    # ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt_path = "/home/haowen/hw_RL_code/act/best.ckpt"
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    # with open(stats_path, 'rb') as f:
    #     stats = pickle.load(f)
    # data = read_dict_with_twodim_numpy_to_csv("/home/haowen/hw_RL_code/imitation/gail-airl-ppo.pytorch/data/begin to m?icrowave.csv")
    # episode_len = 20
    data = read_dict_with_twodim_numpy_to_csv("/home/haowen/hw_RL_code/act/data/microwave_kettle_bottom_burner_3.csv")
    actions = data['actions']
    states = data["states"]
    episode_len = 25
    step_len = 20
    action = [actions[i][-episode_len:] for i in range(len(actions))]
    state = [states[i][-episode_len:] for i in range(len(actions))]
    all_goal = state[0][0,30:]
    qpos = [state[i][:, :9] for i in range(len(actions))]
    action_mean_value = np.mean(np.array(action).reshape(-1, action[0].shape[-1]), axis=0,
                                     dtype=np.float32)
    action_variance_value = np.var(np.array(action).reshape(-1, action[0].shape[-1]), axis=0,
                                        dtype=np.float32)
    qpos_mean_value = np.mean(np.array(qpos).reshape(-1, qpos[0].shape[-1]), axis=0, dtype=np.float32)
    qpos_variance_value = np.var(np.array(qpos).reshape(-1, qpos[0].shape[-1]), axis=0, dtype=np.float32)
    stats = {}
    stats['qpos_mean'] =  np.array([-1.1820256, -1.7640175,    1.8354667, -1.4707927, -0.69022393, 1.3342745, 2.4933178, 0.03350934, 0.02098096])
    stats['qpos_std'] = np.array([2.3016173e-02,3.6310303e-05,1.5356850e-03,1.7665071e-02, 4.1391127e-02, 9.3428660e-03,4.1383069e-02,8.1541679e-05, 2.2902248e-04])
    stats['action_std'] = np.array([0.3474667,0.00537833,0.00510395,0.18782791,0.08568452,0.01770139,0.0668947,0.00873178,0.00487917])
    stats['action_mean'] = np.array([0.19639176, 0.03206327, -0.1724804,0.01981193,0.01140714,0.0282891,0.07306643, -0.14533165, -0.0670352])
    # stats['qpos_mean'] = qpos_mean_value
    # stats['qpos_std'] = qpos_variance_value
    # stats['action_std'] = action_variance_value
    # stats['action_mean'] = action_mean_value

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        # env = make_sim_env(task_name)
        # env_max_reward = env.task.max_reward
        # env = make_kitchen_sim_env(0) # initialize env
        # env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        env = make_kitchen_sim_env(0, begin_step=episode_len) # initialize env
        env_max_reward = env.task.max_reward
        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        print("hereeee")
        with torch.inference_mode():
            for t in range(step_len):
                print(t)
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)
                curr_state = np.concatenate((obs['env_state'], all_goal))
                curr_state = torch.from_numpy(curr_state).float().cuda().unsqueeze(0)
                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_state)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                for j in range(9):
                    # action[j] = obs['env_state'][j] + actions[rollout_id][episode_len + t][j] * 2 * 0.002 * 40
                    action[j] = obs['env_state'][j] + action[j] * 2 * 0.002 * 40
                target_qpos = action

                ### step the environment
                for _ in range(40):
                    ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
            print("done once")
            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return

def forward_pass(data, policy):
    env_data, qpos_data, action_data, is_pad = data
    env_data, qpos_data, action_data, is_pad = env_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, env_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    base_path = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_path, config['ckpt_dir'])
    check_and_create_dir(ckpt_dir)
    seed = config['seed']
    policy_class = config['policy_class']

    set_seed(seed)

    policy = make_policy(policy_class, config)
    if config['load_pretrain']:
        ckpt_path = os.path.join(ckpt_dir, config['pretrain_ckpt'])
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print("success load pretrain: ", loading_status)

    if config['use_wandb']: 
        logger = setup_logging(config)
    
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
                if config['use_wandb']: 
                    logger.log_scalar_dict(forward_dict, prefix='val')
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                best_epoch, min_val_loss, best_state_dict = best_ckpt_info
                ckpt_path = os.path.join(ckpt_dir, f'best_{best_epoch}_seed_{seed}.ckpt')
                torch.save(best_state_dict, ckpt_path)
                print(f'Training save best:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            if config['use_wandb']: 
                    logger.log_scalar_dict(forward_dict, prefix='train')
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            if config["plt"]:
                plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    if config["plt"]:
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robotic dynamic training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
