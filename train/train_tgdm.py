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
from utils.helper import load_config, set_seed, compute_dict_mean, detach_dict, check_and_create_dir, args_overwrite_config
from utils.process_log import plot_history, WandBLogger, AttrDict, setup_logging
from imitation_learning.act_policy import ACTPolicy
from TGDM.tgdm_policy import TGDMPolicy

def get_args_parser():
    parser = argparse.ArgumentParser('Set base param', add_help=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--yaml_dir', action='store', type=str, help='yaml config', default= "../configs/tgdm_test.yaml")
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
    elif policy_class == 'TGDM':
        policy = TGDMPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'TGDM':
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
   pass

def forward_pass(data, policy):
    if len(data) == 4:
        env_data, qpos_data, action_data, is_pad = data
        env_data, qpos_data, action_data, is_pad = env_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, env_data, action_data, is_pad)
    elif len(data) == 6:
        env_data, qpos_data, action_data, is_pad, skip_state_data, skip_qpos_data = data
        env_data, qpos_data, action_data, is_pad, skip_state_data, skip_qpos_data = env_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), skip_state_data.cuda(), skip_qpos_data.cuda()
        return policy(qpos_data, env_data, action_data, is_pad, skip_state_data, skip_qpos_data)
    else:
        raise ValueError
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
            loss = forward_dict['new_loss']
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
