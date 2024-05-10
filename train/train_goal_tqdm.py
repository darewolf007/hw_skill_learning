import _init_paths
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from torch import nn
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from utils.helper import load_config, set_seed, compute_dict_mean, detach_dict, check_and_create_dir, args_overwrite_config
from utils.process_log import plot_history, WandBLogger, AttrDict, setup_logging
from TGDM.models.goal_tqdm_vae import Goal_TGDM_VAE
from dataload.load_goal_tqdm_data import load_data

def build_tgdm_model_and_optimizer(args):
    model = Goal_TGDM_VAE(args)
    model.cuda()
    if args["use_state"]:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]},]
    elif args["use_image"]:
        raise ValueError("not implemented yet")
    else:
        raise ValueError("Please choose at least one of state or image to use")
    optimizer = torch.optim.AdamW(param_dicts, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    return model, optimizer

class Goal_TGDM_Policy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_tgdm_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args['kl_weight']
        self.mode = "image" if args['use_image'] else "state"
        self.num_queries = args['num_queries']

    def __call__(self, input_dict):
        if self.mode == "state":
            image = None
        else:
            raise ValueError("not implemented yet")

        if input_dict.actions is not None: # training time
            input_dict.actions = input_dict.actions[:, :self.num_queries]
            input_dict.is_pad = input_dict.is_pad[:, :self.num_queries]
            model_output = self.model(input_dict)
            loss_dict = self.model.loss(model_output, input_dict)
            return loss_dict
        else: # inference time
            raise ValueError("not implemented yet")

    def configure_optimizers(self):
        return self.optimizer


def get_args_parser():
    parser = argparse.ArgumentParser('Set base param', add_help=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--yaml_dir', action='store', type=str, help='yaml config', default= "../configs/goal_tgdm.yaml")
    return parser

def main(args):
    is_eval = args.eval
    args_config = load_config(os.path.join(os.path.dirname(__file__), args.yaml_dir))
    args_overwrite_config(args_config)
    set_seed(args_config['seed'])

    if is_eval:
        pass

    train_dataloader, val_dataloader = load_data(args_config, os.path.dirname(__file__))

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, args_config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_policy(policy_class, policy_config):
    if policy_class == 'TGDM':
        policy = Goal_TGDM_Policy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'TGDM':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def eval_bc(config, ckpt_name, save_episode=True):
   pass

def forward_pass(data, policy):
    state_data, qpos_data, action_data, is_pad, env_goal, hl_actions, endeffector_xpose, ll_goal_xpose = data
    state_data, qpos_data, action_data, is_pad, env_goal, hl_actions, endeffector_xpose, ll_goal_xpose = state_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), env_goal.cuda(), hl_actions.cuda(), endeffector_xpose.cuda(), ll_goal_xpose.cuda()
    input_dict = AttrDict()
    input_dict['qpos'] = qpos_data
    input_dict['env_state'] = state_data
    input_dict['env_goal'] = env_goal
    input_dict['actions'] = action_data
    input_dict['is_pad'] = is_pad
    input_dict['goal_xpose'] = ll_goal_xpose
    input_dict['endeffector_xpose'] = endeffector_xpose
    input_dict['hl_actions'] = hl_actions
    loss = policy(input_dict)
    return loss


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
