import torch
from torch import nn
from torch.autograd import Variable
from robot_dynamic.utils.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np


def build_encoder(args):
    d_model = args["hidden_dim"]
    dropout = args["dropout"]
    nhead = args["nheads"]
    dim_feedforward = args["dim_feedforward"]
    num_encoder_layers = args["enc_layers"]
    normalize_before = args["pre_norm"]
    activation = args["activation"]

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)