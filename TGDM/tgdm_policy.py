import _init_paths
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import argparse
from torch.nn import functional as F
from imitation_learning.act_util.helper import kl_divergence
from TGDM.models.test_tgdm_vae import build_tgdm_vae

def build_tgdm_model_and_optimizer(args):
    model = build_tgdm_vae(args)
    model.cuda()
    if args["use_state"]:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]},]
    elif args["use_image"]:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args["lr_backbone"],
            },
        ]
    else:
        raise ValueError("Please choose at least one of state or image to use")
    optimizer = torch.optim.AdamW(param_dicts, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    return model, optimizer

class TGDMPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_tgdm_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args['kl_weight']
        self.mode = "image" if args['use_image'] else "state"
        self.num_queries = args['num_queries']
        self.skip_step = args['skip_data']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, env_data, actions=None, is_pad=None, skip_state_data=None, skip_qpos_data=None):
        if self.mode == "state":
            image = None
            env_state = env_data
        else:
            env_state = None
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(env_data)

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            if self.skip_step is not None:
                now_traj, _, (_, _) = self.model(skip_qpos_data, image, skip_state_data, None, None)
                last_traj_queries = a_hat[:,self.skip_step:self.num_queries//2+self.skip_step]
                now_traj_queries = now_traj[:,:self.num_queries//2]
                self_traj_loss = F.l1_loss(last_traj_queries, now_traj_queries, reduction='mean')
            else:
                self_traj_loss = 0
            self.last_traj = a_hat
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            loss_dict['self_traj_loss'] = self_traj_loss
            loss_dict['new_loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['self_traj_loss'] * self.kl_weight * 2
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

if __name__ == "__main__":
    import yaml
    config_file = "/home/haowen/corl2024/configs/imitation_act_example.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    build_tgdm_model_and_optimizer(config)