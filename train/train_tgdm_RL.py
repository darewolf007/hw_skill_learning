import _init_paths
import torch
import numpy as np
import os
import argparse
from utils.helper import load_config, set_seed, compute_dict_mean, check_and_create_dir, args_overwrite_config
from utils.general_utils import detach_dict
from utils.process_log import setup_logging
def get_args_parser():
    parser = argparse.ArgumentParser('Set base param', add_help=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--yaml_dir', action='store', type=str, help='yaml config', default= "../configs/tgdm_test.yaml")
    parser.add_argument('--gpu', default=0, type=int, help='will set CUDA_VISIBLE_DEVICES to selected value')
    parser.add_argument('--debug', default=False, type=int, help='if True, runs in debug mode')
    return parser


class RLTrainer:
    def __init__(self, args):
        self.args = args
        self.args_config = load_config(os.path.join(os.path.dirname(__file__), args.yaml_dir))
        args_overwrite_config(self.args_config)
        set_seed(self.args_config['seed'])
        self.setup_device()
        if self.args_config['use_wandb']: 
            logger = setup_logging(self.args_config)
        else:
            self.logger = None
       
        # build env
        

        # build agent (that holds actor, critic, exposes update method)
        self.conf.agent.num_workers = self.conf.mpi.num_workers
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        if args.resume or self.conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, self.conf.ckpt_path)
            self._hp.n_warmup_steps = 0     # no warmup if we reload from checkpoint!

        # start training/evaluation
        if args.mode == 'train':
            self.train(start_epoch)
        elif args.mode == 'val':
            self.val()
        else:
            self.generate_rollouts()

    def train(self, start_epoch):
        """Run outer training loop."""
        if self._hp.n_warmup_steps > 0:
            self.warmup()

        for epoch in range(start_epoch, self._hp.num_epochs):
            print("Epoch {}".format(epoch))
            self.train_epoch(epoch)

            if not self.args.dont_save and self.is_chef:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                }, os.path.join(self._hp.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))
                self.agent.save_state(self._hp.exp_path)
                self.val()

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Task Generalization from Demonstration Manipulation', parents=[get_args_parser()])
    args = parser.parse_args()