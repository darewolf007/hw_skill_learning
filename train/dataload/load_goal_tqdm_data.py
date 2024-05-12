import numpy as np
import torch
import os
import h5py
import csv
import random
import pickle
from torch.utils.data import TensorDataset, DataLoader
from utils.helper import read_pickle, list_files_in_directory
from utils.general_class import AttrDict

class KitenchenDataset(torch.utils.data.Dataset):
    def __init__(self, episode_len = 100, data_path = None, skip_data = None):
        super(KitenchenDataset).__init__()
        self.episode_len = episode_len
        self.all_state = []
        self.all_action = []
        self.all_endeffector_pose = []
        path_list = list_files_in_directory(data_path)
        self.extract_filter_data(path_list)
        self.actions = np.concatenate(self.all_action, axis=0)
        self.states = np.concatenate(self.all_state, axis=0)
        self.end_effector_xpos = np.concatenate(self.all_endeffector_pose, axis=0)
        self.action_mean_value = np.mean(self.actions, axis=0, dtype=np.float32)
        self.action_variance_value = np.var(self.actions, axis=0, dtype=np.float32)
        self.state_mean_value = np.mean(self.states, axis=0, dtype=np.float32)
        self.state_variance_value = np.var(self.states, axis=0, dtype=np.float32)
        self.end_effector_xpos_mean_value = np.mean(self.end_effector_xpos, axis=0, dtype=np.float32)
        self.end_effector_xpos_variance_value = np.var(self.end_effector_xpos, axis=0, dtype=np.float32)
        dataset_dict = {}
        dataset_dict['action_mean_value'] = self.action_mean_value
        dataset_dict['action_variance_value'] = self.action_variance_value
        dataset_dict['state_mean_value'] = self.state_mean_value
        dataset_dict['state_variance_value'] = self.state_variance_value
        dataset_dict['end_effector_xpos_mean_value'] = self.end_effector_xpos_mean_value
        dataset_dict['end_effector_xpos_variance_value'] = self.end_effector_xpos_variance_value
        file_name = "dataset.pkl"
        with open(file_name, 'wb') as f:
                pickle.dump(dataset_dict, f)
        self.sample_full_episode = False
        self.skip_data = skip_data
    def __len__(self):
        return len(self.all_action)

    def __getitem__(self, index):
        original_action_shape = self.all_action[index].shape
        episode_len = original_action_shape[0]
        max_episode_len = self.episode_len
        if self.sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        action = self.all_action[index][start_ts:]
        action_len = episode_len - start_ts
        
        padded_action = np.zeros((max_episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(max_episode_len)
        is_pad[action_len:] = 1

        state_data = self.all_state[index][start_ts][:30]
        env_goal = self.all_state[index][0][30:]
        endeffector_xpose = self.all_endeffector_pose[index][start_ts]
        hl_actions = self.all_endeffector_pose[index][start_ts + 1] if start_ts + 1 < episode_len else self.all_endeffector_pose[index][-1]
        ll_goal_xpose = hl_actions
        qpos = self.all_state[index][start_ts][:9]
        # construct observations
        state_data = torch.from_numpy(state_data).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        env_goal = torch.from_numpy(env_goal).float()
        hl_actions = torch.from_numpy(hl_actions).float()
        endeffector_xpose = torch.from_numpy(endeffector_xpose).float()
        ll_goal_xpose = torch.from_numpy(ll_goal_xpose).float()
        # normalize image and change dtype to float
        action_data = (action_data - self.action_mean_value) / self.action_variance_value
        qpos_data = (qpos_data - self.state_mean_value[:9]) / self.state_variance_value[:9]
        state_data = (state_data - self.state_mean_value[:30]) / self.state_variance_value[:30]
        hl_actions = (hl_actions - self.end_effector_xpos_mean_value) / self.end_effector_xpos_variance_value
        endeffector_xpose = (endeffector_xpose - self.end_effector_xpos_mean_value) / self.end_effector_xpos_variance_value
        ll_goal_xpose = (ll_goal_xpose - self.end_effector_xpos_mean_value) / self.end_effector_xpos_variance_value
        if self.skip_data is not None and self.skip_data > 0:
            skip_state_data = self.state[index][start_ts + self.skip_data] if start_ts + self.skip_data < episode_len else self.state[index][-1]
            skip_qpos_data = self.qpos[index][start_ts + self.skip_data] if start_ts + self.skip_data < episode_len else self.qpos[index][-1]
            skip_qpos_data = (skip_qpos_data - self.qpos_mean_value) / self.qpos_variance_value
            skip_state_data = torch.from_numpy(skip_state_data).float()
            skip_qpos_data = torch.from_numpy(skip_qpos_data).float()
            return state_data, qpos_data, action_data, is_pad, skip_state_data, skip_qpos_data
        return state_data, qpos_data, action_data, is_pad, env_goal, hl_actions, endeffector_xpose, ll_goal_xpose

    def get_useful_data_index(self, seq_value):
        load_in_data = AttrDict()
        seq_action = seq_value['actions']
        seq_states = seq_value['states']
        seq_end_effector_xpos = seq_value['end_effector_xpos']
        seq_subtask_info = seq_value['subtask_info']
        useful_data_id = 0
        for _, value in seq_subtask_info.items():
            useful_data_id = value['done_task_idx'] if value['done_task_idx'] > useful_data_id else useful_data_id
        load_in_data.useful_data_id = useful_data_id
        load_in_data.action = np.array(seq_action[1:useful_data_id]).astype(np.float32)
        load_in_data.state = np.array(seq_states[1:useful_data_id]).astype(np.float32)
        load_in_data.end_effector_xpos = np.array(seq_end_effector_xpos[1:useful_data_id]).astype(np.float32)
        return load_in_data
    
    def extract_filter_data(self, path_list):
        self.data_collect = []
        for path in path_list:
            seq_data = read_pickle(path)
            for _, seq_value in seq_data.items():
                useful_data = self.get_useful_data_index(seq_value)
                if useful_data.useful_data_id > 10:
                    self.data_collect.append(useful_data)
                    self.all_state.append(useful_data.state)
                    self.all_action.append(useful_data.action)
                    self.all_endeffector_pose.append(useful_data.end_effector_xpos)

def load_data(args_config, base_dir):
    data_path = os.path.join(base_dir, args_config[args_config['task_name']]['dataset_dir'])
    episode_len = args_config[args_config['task_name']]['episode_len']
    if args_config['task_name'] == "goal_tgdm_kitchen_panda":
        train_dataset = KitenchenDataset(episode_len = episode_len, data_path = data_path+ '/train')
        val_dataset = KitenchenDataset(episode_len = episode_len, data_path = data_path+ '/val')
    else:
        raise ValueError("Please choose a valid task name")
    train_dataloader = DataLoader(train_dataset, batch_size=args_config['train_batch_size'], shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args_config['val_batch_size'], shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1, drop_last=True)
    return train_dataloader, val_dataloader