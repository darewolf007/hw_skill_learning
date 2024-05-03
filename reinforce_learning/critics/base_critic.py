import torch
import torch.nn as nn
import copy
from reinforce_learning.util.helper import ParamDict, AttrDict

class Critic(nn.Module):
    """Base critic class."""
    def __init__(self):
        super().__init__()
        self._net = self._build_network()

    def _default_hparams(self):
        default_dict = ParamDict({
            'action_dim': 1,    # dimensionality of the action space
            'normalization': 'none',        # normalization used in policy network ['none', 'batch']
            'action_input': True,       # forward takes actions as second argument if set to True
        })
        return default_dict

    def forward(self, obs, actions=None):
        raise NotImplementedError("Needs to be implemented by child class.")

    @staticmethod
    def dummy_output():
        return AttrDict(q=None)

    def _build_network(self):
        """Constructs the policy network."""
        raise NotImplementedError("Needs to be implemented by child class.")