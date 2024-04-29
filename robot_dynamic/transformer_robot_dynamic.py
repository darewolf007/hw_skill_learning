import _init_paths
import torch
from torch import nn
import numpy as np
from robot_dynamic.utils import build_transformer

class robot_dynamic_with_jointxpose(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gripper_dim = args["gripper_dim"]
        self.joint_dim = args["joint_dim"]
        self.register_buffer('pos_table', build_transformer.get_sinusoid_encoding_table(1+args["joint_dim"], args["hidden_dim"])) 
        self.encoder_joint_qpos_proj = nn.Linear(1, args["hidden_dim"])  # project qpos to embedding
        self.encoder_joint_xpos_proj = nn.Linear(3, args["hidden_dim"])  # project xpos to embedding
        self.decoder_joint_xpos_proj = nn.Linear(3, args["hidden_dim"])  # project xpos to embedding
        self.encoder = build_transformer.build_encoder(args)
        self.xpos_decoders = nn.ModuleList([nn.Linear(args["hidden_dim"]*2, 3) for _ in range(self.joint_dim)])


    def forward(self, joint_qpos, joint0_xpos, joint_xpos = None):
        is_training = joint_xpos is not None # train or val
        bs, _ = joint_qpos.shape
        joint_qpos_embed = torch.unsqueeze(joint_qpos, axis=2)
        joint_qpos_embed = self.encoder_joint_qpos_proj(joint_qpos_embed)  # (bs, joint_dim, hidden_dim)
        joint_xqpos_embed = torch.unsqueeze(joint0_xpos, axis=1)  # (bs, 1, hidden_dim)
        joint_xqpos_embed = self.encoder_joint_xpos_proj(joint_xqpos_embed)
        encoder_input = torch.cat([joint_xqpos_embed, joint_qpos_embed], axis=1) # (bs, seq+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        encoder_output = self.encoder(encoder_input, pos=pos_embed)
        joint_embedding = encoder_output.permute(1, 0, 2)
        decoded_positions = [joint0_xpos]
        for joint_id, decoder in enumerate(self.xpos_decoders):
            if is_training:
                assert joint_xpos.shape[1] == self.joint_dim + 1
                if joint_id < self.joint_dim - self.gripper_dim:
                    joint_id_embedding = joint_embedding[:, joint_id, :]
                    joint_id_xpos = joint_xpos[:, joint_id, :]
                    joint_xpos_embedding = self.decoder_joint_xpos_proj(joint_id_xpos)
                    joint_input = torch.cat([joint_id_embedding, joint_xpos_embedding], axis=1)
                    decoded_position = decoder(joint_input.reshape(bs, -1))
                    decoded_positions.append(decoded_position)
                else:
                    joint_id_xpos = joint_xpos[:, self.joint_dim - self.gripper_dim, :]
                    joint_xpos_embedding = self.decoder_joint_xpos_proj(joint_id_xpos)
                    joint_gripper_embedding = joint_embedding[:, joint_id, :]
                    joint_gripper_input = torch.cat([joint_xpos_embedding, joint_gripper_embedding], axis=1)
                    decoded_position = decoder(joint_gripper_input.reshape(bs, -1))
                    decoded_positions.append(decoded_position)
            else:
                if joint_id < self.joint_dim - self.gripper_dim:
                    joint_id_embedding = joint_embedding[:, joint_id, :]
                    joint_id_xpos = decoded_positions[-1]
                    joint_xpos_embedding = self.decoder_joint_xpos_proj(joint_id_xpos)
                    joint_input = torch.cat([joint_id_embedding, joint_xpos_embedding], axis=1)
                    decoded_position = decoder(joint_input.reshape(bs, -1))
                    decoded_positions.append(decoded_position)
                else:
                    joint_id_xpos = decoded_positions[self.joint_dim - self.gripper_dim]
                    joint_xpos_embedding = self.decoder_joint_xpos_proj(joint_id_xpos)
                    joint_gripper_embedding = joint_embedding[:, joint_id, :]
                    joint_gripper_input = torch.cat([joint_xpos_embedding, joint_gripper_embedding], axis=1)
                    decoded_position = decoder(joint_gripper_input.reshape(bs, -1))
                    decoded_positions.append(decoded_position)
        decoded_positions = torch.stack(decoded_positions, dim=1)
        return decoded_positions

class robot_dynamic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gripper_dim = args["gripper_dim"]
        self.joint_dim = args["joint_dim"]
        self.register_buffer('pos_table', build_transformer.get_sinusoid_encoding_table(1+args["joint_dim"], args["hidden_dim"])) 
        self.encoder_joint_qpos_proj = nn.Linear(1, args["hidden_dim"])  # project qpos to embedding
        self.encoder_joint_xpos_proj = nn.Linear(3, args["hidden_dim"])  # project xpos to embedding
        self.encoder = build_transformer.build_encoder(args)
        self.xpos_decoders = nn.ModuleList([nn.Linear(args["hidden_dim"], 3) for _ in range(self.joint_dim + 1)])


    def forward(self, joint_qpos, joint0_xpos):
        bs, _ = joint_qpos.shape
        joint_qpos_embed = torch.unsqueeze(joint_qpos, axis=2)
        joint_qpos_embed = self.encoder_joint_qpos_proj(joint_qpos_embed)  # (bs, joint_dim, hidden_dim)
        joint_xqpos_embed = torch.unsqueeze(joint0_xpos, axis=1)  # (bs, 1, hidden_dim)
        joint_xqpos_embed = self.encoder_joint_xpos_proj(joint_xqpos_embed)
        encoder_input = torch.cat([joint_xqpos_embed, joint_qpos_embed], axis=1) # (bs, seq+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        encoder_output = self.encoder(encoder_input, pos=pos_embed)
        joint_embedding = encoder_output.permute(1, 0, 2)
        decoded_end_positions = []
        for joint_id, decoder in enumerate(self.xpos_decoders):
            if joint_id <= self.joint_dim - self.gripper_dim:
                joint_id_embedding = joint_embedding[:, joint_id, :]
                decoded_position = decoder(joint_id_embedding)
                decoded_end_positions.append(decoded_position)
            else:
                joint_gripper_embedding = joint_embedding[:, joint_id, :]
                decoded_position = decoder(joint_gripper_embedding)
                decoded_end_positions.append(decoded_position)
        pred_end_positions = torch.stack(decoded_end_positions, dim=1)
        return pred_end_positions

if __name__ == '__main__':
    import yaml
    config_file = "/home/haowen/corl2024/configs/panda_robotic_dynamic.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    model = robot_dynamic(config)
    model.train()
    model.to("cuda")
    joint_qpos = torch.rand(2, 9, device='cuda')
    joint0_xpos = torch.rand(2, 3, device='cuda')
    joint_xpos = torch.rand(2, 10, 3, device='cuda')
    # model(joint_qpos, joint0_xpos, joint_xpos)
    joint_xpos = None
    model(joint_qpos, joint0_xpos)
    