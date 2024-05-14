import _init_paths
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
# from dataload.load_imitation_act_data import load_data
from utils.helper import load_config, set_seed, compute_dict_mean, check_and_create_dir, args_overwrite_config, save_videos, read_pickle
from utils.process_log import plot_history, WandBLogger, setup_logging
from imitation_learning.act_policy import ACTPolicy
from utils.general_utils import detach_dict
from environment.kitchen_env import Kitchen_Grasp_Task
from utils.general_class import ParamDict
from dataload.load_goal_tqdm_data import load_data

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
        policy_class = args_config['policy_class']
        set_seed(args_config['seed'])
        policy = make_policy(policy_class, args_config)
        base_path = os.path.dirname(__file__)
        ckpt_dir = os.path.join(base_path, args_config['ckpt_dir'])
        ckpt_path = os.path.join(ckpt_dir, args_config['pretrain_ckpt'])
        policy.load_state_dict(torch.load(ckpt_path))
        eval_bc(args_config, policy)
    else:
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

def post_process_action(t, all_actions, stats_info, all_time_actions, num_queries, query_frequency, config):
    action_mean_value = stats_info['action_mean_value']
    action_variance_value = stats_info['action_variance_value']
    post_process = lambda a: a * action_variance_value + action_mean_value
    if config['temporal_agg']:
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
    raw_action = raw_action.squeeze(0).cpu().numpy()
    action = post_process(raw_action)
    return action
      
def pre_process_env(qpos, stats_info, env_dim):
    state_mean_value = stats_info['state_mean_value']
    state_variance_value = stats_info['state_variance_value']
    pre_process = lambda qpos: (qpos - state_mean_value[:env_dim]) / state_variance_value[:env_dim]
    return pre_process(qpos)

def eval_calculation(episode_returns, highest_rewards, env_max_reward, num_rollouts):
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'
    print(summary_str)

def test_eval_bc(config, policy, save_episode=False, data=None):
    seed = config['seed']
    set_seed(seed)
    policy.eval()
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, config[config['task_name']]['dataset_dir'])
    stats_info = read_pickle(data_path + '/dataset_state.pkl')
    if data is not None:
        env_data, qpos_data, action_data, is_pad = data
        state_mean_value = stats_info['state_mean_value']
        state_variance_value = stats_info['state_variance_value']
        post_env_data = env_data * state_variance_value[:30] + state_mean_value[:30]
        for i in range(env_data.shape[0]):
            init_param = ParamDict()
            init_param.update(config)
            init_param.start_arm_pose = post_env_data[i][:30]
            env = Kitchen_Grasp_Task(init_param)
            env.make_env()
            obs, reward, done, info  = env.reset()
            for t in range(config['episode_len']):
                action = action_data[i][t]
                action_mean_value = stats_info['action_mean_value']
                action_variance_value = stats_info['action_variance_value']
                post_action_data = action * action_variance_value + action_mean_value
                obs, reward, done, info  = env.step(post_action_data)
            print(f'Episode reward: {reward}')

def eval_bc(config, policy, save_episode=False):
    seed = config['seed']
    set_seed(seed)
    policy.eval()
    base_path = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_path, config['ckpt_dir'])
    if config['temporal_agg']:
        query_frequency = 1
        num_queries = config['num_queries']
    else:
        query_frequency = config['num_queries']
    data_path = os.path.join(base_path, config[config['task_name']]['dataset_dir'])
    stats = read_pickle(data_path + '/dataset_state.pkl')
    max_timesteps = int(config['max_episode_steps'] * 1) # may increase for real-world tasks
    num_rollouts = config['num_rollouts']
    for rollout_id in range(num_rollouts):
        init_param = ParamDict()
        init_param.update(config)
        ### evaluation loop
        if config['temporal_agg']:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, config['action_dim']]).cuda()
        image_list = [] 
        rewards = []
        with torch.inference_mode():
            env = Kitchen_Grasp_Task(init_param)
            env.make_env()
            env_max_reward = config['max_reward']
            new_obs, reward, done, info  = env.reset()
            for t in range(config['max_episode_steps']):
                image_list.append(info['images'])
                obs = np.copy(new_obs)
                pre_obs = pre_process_env(obs, stats, config['env_dim'])
                qpos_numpy = pre_obs[:9]
                qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
                curr_state = torch.from_numpy(pre_obs).float().cuda().unsqueeze(0)
                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_state)
                else:
                    raise NotImplementedError
                ### post-process actions
                action = post_process_action(t, all_actions, stats, all_time_actions, num_queries, query_frequency, config)
                new_obs, reward, done, info  = env.step(action)
                if done:
                    rewards.append(reward)
                    break
            rewards.append(reward)

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_highest_reward = np.max(rewards)
        avg_reward = np.mean(rewards)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        if save_episode:
            save_videos(image_list, config['DT'], video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
        return avg_reward

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
            avg_reward = eval_bc(config, policy, False)
            avg_reward_dict = {}
            avg_reward_dict['train_episode_reward'] = avg_reward
            if config['use_wandb']:
                logger.log_scalar_dict(avg_reward_dict)
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
            # torch.save(policy.state_dict(), ckpt_path)
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
