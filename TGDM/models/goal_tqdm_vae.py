import os
import sys
sys.path.insert(0, os.getcwd())
import torch
from torch import nn
from torch.autograd import Variable
from utils.transformer.build_transformer import build_encoder, get_sinusoid_encoding_table
from utils.transformer.transformer import build_transformer
from TGDM.models.base_model import BaseModel
from TGDM.util.help import ParamDict, AttrDict

class Goal_TGDM_VAE(BaseModel):
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        self._hp = self.update_hparams(params)
        self.build_network()
        self.load_weights_and_freeze()

    def _default_hparams(self):
        default_dict = ParamDict({
            'use_images': False,
            'use_state': True,
            'use_robotic_dynamic': True,
            'device': None,
            'embedding_checkpoint': None,
        })
        # Network size
        default_dict.update({
            'num_queries': None, 
            'action_dim': None,
            'joint_dim': None,        
            'env_dim': None,         
            'hidden_dim': None,
        })

        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def update_hparams(self, hparams):
        high_param = self._default_hparams()
        high_param.overwrite(hparams)
        return high_param

    def build_network(self):
        transformer = build_transformer(self._hp)

        encoder = build_encoder(self._hp)

        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        # backbone parameters
        self.action_head = nn.Linear(hidden_dim, self._hp["action_dim"])
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(self._hp["num_queries"], hidden_dim)
        if self._hp["use_image"]:
            raise ValueError("not implemented yet")
        elif self._hp["use_state"]:
            self.input_proj_robot_state = nn.Linear(self._hp["joint_dim"], hidden_dim)
            self.input_proj_env_state = nn.Linear(self._hp["env_dim"], hidden_dim)
            self.pos = torch.nn.Embedding(1, hidden_dim)
            self.backbones = None
        else:
            raise ValueError("Please choose at least one of state or image to use")

        # encoder extra parameters
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self._hp["action_dim"], hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(self._hp["joint_dim"], hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self._hp["latent_dim"]*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+self._hp["num_queries"], hidden_dim)) # [CLS], qpos, a_seq
        self.register_buffer('state_pos_table', get_sinusoid_encoding_table(1, hidden_dim)) # [CLS], qpos, a_seq
        self.register_buffer('robotic_pos_table', get_sinusoid_encoding_table(1+self._hp["joint_dim"], hidden_dim)) 
        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self._hp["latent_dim"], hidden_dim) # project latent sample to embedding
        #robotic_dynamic model
        if self._hp["use_robotic_dynamic"]:
            self.robotic_transformer_args = self._hp[self._hp['robotic_name']]
            self.base_xpos = torch.tensor(self._hp[self._hp['robotic_name']]["joint_base_xpose"]).float().cuda()
            self.encoder_joint_qpos_proj = nn.Linear(1, hidden_dim)  # project qpos to embedding
            self.encoder_joint_xpos_proj = nn.Linear(3, hidden_dim)  # project xpos to embedding
            self.robotic_dynamic_encoder = build_encoder(self.robotic_transformer_args)
            self.additional_pos_embed = nn.Embedding(1 + 1 + self._hp["joint_dim"], hidden_dim)
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        # if self._hp.embedding_checkpoint is not None:
        #     print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
        #     self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
        #     self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
        #     freeze_modules([self.decoder, self.decoder_input_initalizer, self.decoder_hidden_initalizer, self.q])
        pass

    def forward(self, inputs):
        qpos = inputs['qpos']
        env_state = inputs['env_state']
        actions = inputs['actions']
        is_pad = inputs['is_pad']
        bs, _ = qpos.shape

        # project action sequence to embedding dim, and concat with a CLS token
        action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
        qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
        qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
        cls_embed = self.cls_embed.weight # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
        # do not mask cls token
        cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
        is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        # query model
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
        encoder_output = encoder_output[0] # take cls output only
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, :self._hp["latent_dim"]]
        logvar = latent_info[:, self._hp["latent_dim"]:]
        latent_sample = self.reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)
        
        hs = self.decode(qpos, env_state, latent_input)
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
    
    def val_mode(self, inputs):
        qpos = inputs['qpos']
        env_state = inputs['env_state']
        bs, _ = qpos.shape
        mu = logvar = None
        latent_sample = torch.zeros([bs, self._hp["latent_dim"]], dtype=torch.float32).to(qpos.device)
        latent_input = self.latent_out_proj(latent_sample)
        self.decode(qpos, env_state, latent_input)
    
    def decode(self, qpos, env_state, latent_input):
        bs, _ = qpos.shape
        if self._hp.use_robotic_dynamic:
                base_xpos = self.base_xpos.repeat(bs, 1)
                joint_qpos_embed = torch.unsqueeze(qpos, axis=2)
                joint_qpos_embed = self.encoder_joint_qpos_proj(joint_qpos_embed)  # (bs, joint_dim, hidden_dim)
                joint_xqpos_embed = torch.unsqueeze(base_xpos, axis=1)  # (bs, 1, hidden_dim)
                joint_xqpos_embed = self.encoder_joint_xpos_proj(joint_xqpos_embed)
                encoder_input = torch.cat([joint_xqpos_embed, joint_qpos_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                pos_embed = self.robotic_pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                encoder_output = self.encoder(encoder_input, pos=pos_embed)
                joint_embedding = encoder_output.permute(1, 0, 2)
                env_state = self.input_proj_env_state(env_state)
                hs = self.transformer(joint_embedding, None, self.query_embed.weight, self.pos.weight, latent_input, env_state, self.additional_pos_embed.weight)[6]
        else:
            joint_embedding = self.robotic_dynamic_encoder(qpos)
            env_state = self.input_proj_env_state(env_state)
            env_state = torch.unsqueeze(env_state, axis=1)
            hs = self.transformer(env_state, None, self.query_embed.weight, self.pos.weight, latent_input, joint_embedding, self.additional_pos_embed.weight)[6]
        return hs
    
    def loss(self, model_output, inputs):
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
    
    def compute_learned_prior(self, inputs, first_only=False):
        
        return self._compute_learned_prior(self.p[0], inputs)


if __name__ == "__main__":
    import yaml
    config_file = "/home/haowen/corl2024/configs/tgdm_test.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['env_dim'] = 30
    param_dict = ParamDict()
    param_dict.update(config)
    model = Goal_TGDM_VAE(param_dict)
    model.to(torch.device('cuda'))
    input_dict = AttrDict()
    input_dict['qpos'] = torch.randn(8, 9).to(torch.device('cuda'))
    input_dict['env_state'] = torch.randn(8, 30).to(torch.device('cuda'))
    input_dict['actions'] = torch.randn(8, param_dict.num_queries, 9).to(torch.device('cuda'))
    input_dict['is_pad'] = torch.randn(8, param_dict.num_queries).to(torch.device('cuda'))
    input_dict['goal_xpose'] = torch.randn(8, 3).to(torch.device('cuda'))
    model(input_dict)
    model.val_mode(input_dict)

