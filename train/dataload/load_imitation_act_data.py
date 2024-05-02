import numpy as np
import torch
import os
import h5py
import csv
import random
from torch.utils.data import TensorDataset, DataLoader
from utils.helper import read_pickle, read_dict_with_twodim_numpy_to_csv

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

class KitenchenDataset(torch.utils.data.Dataset):
    def __init__(self,path_num = 40,episode_len=35, data_path = None,load_all_data = False, load_state_action = False):
        super(KitenchenDataset).__init__()
        self.episode_len = episode_len
        self.episode_num = path_num
        if load_all_data:
            self.data = read_pickle(data_path)
            self.all_state = self.data['all_states']
            if not load_state_action:
                self.all_action = self.data['all_actions']
            else:
                self.action_shape = self.data['all_actions'][0].shape[-1]
                self.all_action = []
                for id in range(len(self.all_state)):
                    step_action = np.zeros((self.all_state[id].shape[0], self.action_shape))
                    for i in range(self.all_state[id].shape[0] - 1):
                        step_action[i] = self.all_state[id][i + 1, :9]
                    step_action[-1] = self.all_state[id][-1, :9]
                    self.all_action.append(step_action)
            self.all_goal = self.data['all_goal']
            episode_sample_ids = random.sample(range(len(self.all_action)), path_num)
            episode_len_min = np.min(np.array([self.all_action[i].shape[0] for i in range(len(self.all_action))]))
            print("kitchen_episode_len_min", episode_len_min)
            assert episode_len_min > episode_len
            self.action = [self.all_action[i][-episode_len:] for i in episode_sample_ids]
            self.state = [self.all_state[i][-episode_len:] for i in episode_sample_ids]
        else:
            self.data = read_dict_with_twodim_numpy_to_csv(data_path)
            self.all_action = self.data['actions']
            self.all_state = self.data['states']
            episode_sample_ids = random.sample(range(len(self.all_action)), path_num)
            episode_len_min = np.min(np.array([self.data['actions'][i].shape[0] for i in range(len(self.data['actions']))]))
            print("kitchen_episode_len_min", episode_len_min)
            assert episode_len_min > episode_len
            self.action = [self.all_action[i][-episode_len:] for i in episode_sample_ids]
            self.state = [self.all_state[i][-episode_len:] for i in episode_sample_ids]
        self.qpos = [self.state[i][:,:9] for i in range(self.episode_num)]
        self.action_mean_value = np.mean(np.array(self.action).reshape(-1, self.action[0].shape[-1]), axis=0, dtype=np.float32)
        self.action_variance_value = np.var(np.array(self.action).reshape(-1, self.action[0].shape[-1]), axis=0, dtype=np.float32)
        self.qpos_mean_value = np.mean(np.array(self.qpos).reshape(-1, self.qpos[0].shape[-1]), axis=0, dtype=np.float32)
        self.qpos_variance_value = np.var(np.array(self.qpos).reshape(-1, self.qpos[0].shape[-1]), axis=0, dtype=np.float32)
        self.sample_full_episode = False
        self.is_sim = True
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return self.episode_num

    def __getitem__(self, index):
        original_action_shape = self.action[0].shape
        episode_len = original_action_shape[0]
        if self.sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.qpos[index][start_ts]

        if self.is_sim:
            action = self.action[index][start_ts:]
            action_len = episode_len - start_ts
        else:
            action = self.action[index][max(0, start_ts - 1):] # hack, to make timesteps more aligned
            action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned


        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1


        # construct observations
        state_data = torch.from_numpy(self.state[index][start_ts]).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize image and change dtype to float
        action_data = (action_data - self.action_mean_value) / self.action_variance_value
        qpos_data = (qpos_data - self.qpos_mean_value) / self.qpos_variance_value

        return state_data, qpos_data, action_data, is_pad


def load_example_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    def example_get_norm_stats(dataset_dir, num_episodes):
        all_qpos_data = []
        all_action_data = []
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
        all_qpos_data = torch.stack(all_qpos_data)
        all_action_data = torch.stack(all_action_data)
        all_action_data = all_action_data

        # normalize action data
        action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
        action_std = all_action_data.std(dim=[0, 1], keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

        stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
                "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
                "example_qpos": qpos}

        return stats

    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = example_get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader

def load_data(args_config, base_dir):
    data_path = os.path.join(base_dir, args_config[args_config['task_name']]['dataset_dir']+ args_config[args_config['task_name']]['dataset_name'])
    path_num = args_config[args_config['task_name']]['num_episodes']
    episode_len = args_config[args_config['task_name']]['episode_len']
    if args_config['task_name'] == "kitchen_one_task":
        train_dataset = KitenchenDataset(path_num = path_num,episode_len=episode_len, data_path = data_path)
        val_dataset = KitenchenDataset(path_num = path_num,episode_len=episode_len, data_path = data_path)
    elif args_config['task_name'] == "kitchen_all_task":
        train_dataset = KitenchenDataset(path_num = path_num, episode_len = episode_len, data_path = data_path)
        val_dataset = KitenchenDataset(path_num = path_num, episode_len = episode_len, data_path = data_path)
    elif args_config['task_name'] == "kitchen_one_task_allstep":
        train_dataset = KitenchenDataset(path_num = path_num,episode_len=episode_len, data_path = data_path,load_all_data=True, load_state_action=True)
        val_dataset = KitenchenDataset(path_num = path_num,episode_len=episode_len, data_path = data_path,load_all_data=True, load_state_action=True)
    elif args_config['task_name'] == "act_example":
        from simulation_mujoco.assets.act_example.constants import SIM_TASK_CONFIGS
        example_data_path = SIM_TASK_CONFIGS[args_config[args_config['task_name']]['dataset_name']]['dataset_dir']
        data_path = os.path.join(base_dir, example_data_path)
        path_num = SIM_TASK_CONFIGS[args_config[args_config['task_name']]['dataset_name']]['num_episodes']
        camera_names = SIM_TASK_CONFIGS[args_config[args_config['task_name']]['dataset_name']]['camera_names']
        return load_example_data(data_path, path_num, camera_names, args_config['train_batch_size'], args_config['val_batch_size'])
    else:
        raise ValueError("Please choose a valid task name")
    train_dataloader = DataLoader(train_dataset, batch_size=args_config['train_batch_size'], shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=args_config['val_batch_size'], shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    return train_dataloader, val_dataloader