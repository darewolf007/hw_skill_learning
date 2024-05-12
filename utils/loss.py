import torch
import numpy as np
def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数表示的旋转角度转换为旋转矩阵
    """
    w, x, y, z = quaternion
    rotation_matrix = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], device=quaternion.device)  # 将旋转矩阵分配到与四元数相同的设备上
    return rotation_matrix

def angle_loss(quaternion1, quaternion2):
    """
    计算两个四元数之间的角度损失
    """
    rotation_matrix1 = quaternion_to_rotation_matrix(quaternion1)
    rotation_matrix2 = quaternion_to_rotation_matrix(quaternion2)
    dot_product = torch.trace(torch.matmul(rotation_matrix1, torch.transpose(rotation_matrix2, 0, 1)))
    cosine_similarity = dot_product / 3.0
    angle = torch.acos(torch.clamp(cosine_similarity, -1.0 + 1e-7, 1.0 - 1e-7))
    return angle

def position_loss(position1, position2):
    """
    计算两个位置向量之间的位置损失
    """
    return torch.norm(position1 - position2)

def combined_loss(quaternion1, quaternion2, position1, position2, angle_weight=0.5):
    """
    计算结合了角度和位置的损失函数
    """
    combined_loss = 0.0
    for bs in range(quaternion1.shape[0]):
        angle_loss_val = angle_loss(quaternion1[bs], quaternion2[bs])
        position_loss_val = position_loss(position1[bs], position2[bs])
        combined_loss += (1.0 - angle_weight) * angle_loss_val + angle_weight * position_loss_val
    combined_loss /= quaternion1.shape[0]
    return combined_loss