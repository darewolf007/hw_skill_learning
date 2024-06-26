import _init_paths
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import argparse
from torch.nn import functional as F
from imitation_learning.act_util.detr_vae import build_detr_vae
from imitation_learning.act_util.helper import kl_divergence

def build_ACT_model_and_optimizer(args):
    model = build_detr_vae(args)
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

class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args['kl_weight']
        self.mode = "image" if args['use_image'] else "state"
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, env_data, actions=None, is_pad=None):
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
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
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
    build_detr_vae(config)