import torch
import numpy as np
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader

class Robotic_FK_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict_path, joint_base_xpose, sample_terminal = 10, use_angle = True):
        super(Robotic_FK_Dataset).__init__()
        data_files = os.listdir(data_dict_path)
        self.use_angle = use_angle
        self.joint_qpose_data = []
        self.end_effector_xpos_data = []
        self.leftfinger_xpos_data = []
        self.rightfinger_xpos_data = []
        self.joint_base_xpose = np.array(joint_base_xpose)
        for file in data_files:
            file_path = os.path.join(data_dict_path, file)
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                for seq_id, all_values in loaded_data.items():
                    for key, value in all_values.items():
                        if key == 'joint_qpose':
                            self.joint_qpose_data.extend(value[::sample_terminal])
                        if key == 'end_effector_xpos':
                            self.end_effector_xpos_data.extend(value[::sample_terminal])
                        if key == 'leftfinger_xpos':
                            self.leftfinger_xpos_data.extend(value[::sample_terminal])
                        if key == 'rightfinger_xpos':
                            self.rightfinger_xpos_data.extend(value[::sample_terminal])
        assert len(self.joint_qpose_data) == len(self.end_effector_xpos_data) == len(self.leftfinger_xpos_data) == len(self.rightfinger_xpos_data)


    def __len__(self):
        return len(self.joint_qpose_data)
    
    def __getitem__(self, index):
        joint_qpose = self.joint_qpose_data[index]
        if self.use_angle:
            joint_qpose = np.degrees(joint_qpose)
        end_effector_xpos = self.end_effector_xpos_data[index]
        leftfinger_xpos = self.leftfinger_xpos_data[index]
        rightfinger_xpos = self.rightfinger_xpos_data[index]
        joint_qpose = torch.from_numpy(joint_qpose).float()
        end_effector_xpos = torch.from_numpy(end_effector_xpos).float()
        leftfinger_xpos = torch.from_numpy(leftfinger_xpos).float()
        rightfinger_xpos = torch.from_numpy(rightfinger_xpos).float()
        joint_base_xpose = torch.from_numpy(self.joint_base_xpose).float()
        return joint_qpose, end_effector_xpos, leftfinger_xpos, rightfinger_xpos, joint_base_xpose

def load_data(data_dict_train_path, data_dict_val_path, batch_size_train, batch_size_val, joint_base_xpose = np.array([0,0,0]), sample_terminal = 10):
    train_dataset = Robotic_FK_Dataset(data_dict_train_path, joint_base_xpose, sample_terminal= sample_terminal)
    val_dataset = Robotic_FK_Dataset(data_dict_val_path, joint_base_xpose,  sample_terminal= 2 * sample_terminal)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=True, num_workers=1, prefetch_factor=1)
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    load_data("/home/haowen/corl2024/data/robotic_model/train", "/home/haowen/corl2024/data/robotic_model/test", 2, 1)