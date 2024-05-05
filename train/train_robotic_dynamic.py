import _init_paths
import torch.nn as nn
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from robot_dynamic.transformer_robot_dynamic import robot_dynamic, robot_dynamic_with_jointxpose
from utils.helper import load_config, set_seed, compute_dict_mean, detach_dict, euclidean_distance, combined_loss
from utils.process_log import plot_history, WandBLogger, AttrDict, setup_logging
from train.dataload.load_robotic_dynamic_data import load_data, load_alljoint_data
from torch.nn import functional as F

class Train_Policy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = self.build_model_and_optimizer(args)
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def __call__(self, joint_data, is_train, predict_gripper = False):
        joint_qpose, end_effector_xpos, leftfinger_xpos, rightfinger_xpos, joint_base_xpose, joints_xpose= joint_data
        if self.args['device'] == 'cuda':
            joint_qpose = joint_qpose.cuda()
            end_effector_xpos = end_effector_xpos.cuda()
            leftfinger_xpos = leftfinger_xpos.cuda()
            rightfinger_xpos = rightfinger_xpos.cuda()
            joint_base_xpose = joint_base_xpose.cuda()
            joints_xpose = joints_xpose.cuda()
        if is_train:
            end_positions_result = self.model(joint_qpose, joint_base_xpose, joints_xpose[:,1:])
        else:
            end_positions_result = self.model(joint_qpose, joint_base_xpose)
        loss_dict = dict()
        joint_target = joints_xpose
        for joint_id in range(1, self.args['joint_dim'] + 1):
            joint_name = ''
            if joint_id < self.args['joint_dim'] - self.args['gripper_dim']:
                joint_name = 'joint_' + str(joint_id) if joint_id != 0 else 'joint_base'
                pred_quaternion = end_positions_result[:,joint_id,3:]
                traget_quaternion = joint_target[:,joint_id,3:]
                pred_position = end_positions_result[:,joint_id,:3]
                traget_position = joint_target[:,joint_id,:3]
                pred_joint_loss = combined_loss(pred_quaternion, traget_quaternion, pred_position, traget_position)
                loss_dict[joint_name] = pred_joint_loss
            elif joint_id == self.args['joint_dim'] - self.args['gripper_dim']:
                joint_name = 'joint_' + str(joint_id) if joint_id != 0 else 'joint_base'
                pred_position = end_positions_result[:,joint_id,:3]
                predict_end_positions_loss = torch.mean(euclidean_distance(pred_position, end_effector_xpos))
                loss_dict[joint_name] = predict_end_positions_loss
            elif predict_gripper:
                if joint_id == self.args['joint_dim'] - self.args['gripper_dim'] + 1:
                    joint_name = 'grapper_left'
                    gripper_predict_end_positions = end_positions_result[:, joint_id]
                    predict_end_positions_loss = torch.mean(euclidean_distance(gripper_predict_end_positions, leftfinger_xpos))
                else:
                    joint_name = 'grapper_right'
                    gripper_predict_end_positions = end_positions_result[:, joint_id]
                    predict_end_positions_loss = torch.mean(euclidean_distance(gripper_predict_end_positions, rightfinger_xpos))
            else:
                continue
        if predict_gripper:
            loss_dict['loss'] = torch.mean(sum([loss_dict[joint_name] for joint_name in loss_dict if 'joint' in joint_name]))
        else:
            loss_dict['loss'] = torch.mean(sum([loss_dict[joint_name] for joint_name in loss_dict]))
        return loss_dict

    
    def configure_optimizers(self):
        return self.optimizer

    def build_model_and_optimizer(self, args):
        if args['all_joints_predict']:
            model = robot_dynamic_with_jointxpose(args)
        else:
            model = robot_dynamic(args)
        model.cuda()
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]},]
        optimizer = torch.optim.AdamW(param_dicts, lr=args["lr"], weight_decay=args["weight_decay"])
        return model, optimizer

def get_args_parser():
    parser = argparse.ArgumentParser('Set base param', add_help=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--plt', action='store_true')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb log', default=False)
    parser.add_argument('--yaml_dir', action='store', type=str, help='yaml config', default= "../configs/panda_robotic_dynamic.yaml")
    return parser

def get_dataloader(args_config):
    current_directory = os.path.dirname(__file__)
    train_path = os.path.join(current_directory, args_config['data_dict_train_path'])
    val_path = os.path.join(current_directory, args_config['data_dict_val_path'])
    if args_config['all_joints_predict']:
        train_dataloader, val_dataloader = load_alljoint_data(train_path, val_path, args_config['train_batch_size'], args_config['val_batch_size'], args_config['joint_base_xpose'], sample_terminal=args_config['sample_terminal'])
    else:
        train_dataloader, val_dataloader = load_data(train_path, val_path, args_config['train_batch_size'], args_config['val_batch_size'], args_config['joint_base_xpose'], sample_terminal=args_config['sample_terminal'])
    return train_dataloader, val_dataloader

def init_model(args_config):
    set_seed(args_config['seed'])
    policy = Train_Policy(args_config)
    logger = None
    if args_config['load_pretrain']:
        ckpt_path = os.path.join(os.path.dirname(__file__), args_config['pre_train_model_path'])
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        if loading_status:
            print("success load pretrain: ", ckpt_path)
    if args_config['device'] == 'cuda':
        policy.cuda()
    optimizer = policy.configure_optimizers()
    if args.use_wandb: 
        logger = setup_logging(args_config)
    return policy, optimizer, logger

def train_model(args):
    args_config = load_config(os.path.join(os.path.dirname(__file__), args.yaml_dir))
    policy, optimizer, logger = init_model(args_config)
    train_dataloader, val_dataloader = get_dataloader(args_config)
    validation_history = []
    train_history = []
    min_val_loss = np.inf
    for epoch in tqdm(range(args_config['num_epochs'])):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = policy(data, is_train=False)
                epoch_dicts.append(forward_dict)
                if args.use_wandb: 
                    logger.log_scalar_dict(forward_dict, prefix='val')
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                best_epoch, min_val_loss, best_state_dict = best_ckpt_info
                ckpt_dir = os.path.join(os.path.dirname(__file__), args_config['save_model_path'])
                ckpt_path = os.path.join(ckpt_dir, f"best_{best_epoch}_seed_{args_config['seed']}.ckpt")
                torch.save(best_state_dict, ckpt_path)
                print(f"Training save best:\nSeed {args_config['seed']}, val loss {min_val_loss.item():.6f} at epoch {best_epoch}")
        print(f'Val loss:   {epoch_val_loss.item():.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = policy(data,is_train=True)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if args.use_wandb: 
                logger.log_scalar_dict(forward_dict, prefix='train')
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss.item():.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if args_config['save_all_temp'] and epoch % 100 == 0:
            ckpt_dir = os.path.join(os.path.dirname(__file__), args_config['save_temp_model_path'])
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{args_config['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            if args.plt:
                plot_history(train_history, validation_history, epoch, ckpt_dir, args_config['seed'])
        else:
            ckpt_dir = os.path.join(os.path.dirname(__file__), args_config['save_temp_model_path'])
            ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            if args.plt:
                plot_history(train_history, validation_history, epoch, ckpt_dir, args_config['seed'])
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {args_config['seed']}, val loss {min_val_loss.item():.6f} at epoch {best_epoch}")
    print(f'Best ckpt, val loss {min_val_loss.item():.6f} @ epoch{best_epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robotic dynamic training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)

