import os
import sys
sys.path.insert(0, os.getcwd())
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils.transformer.build_transformer import build_encoder, get_sinusoid_encoding_table
from utils.transformer.transformer import build_transformer
from TGDM.models.base_model import BaseModel
from TGDM.util.helper import ParamDict, AttrDict
from TGDM.util.helper import kl_divergence, KLDivLoss, NLL
from TGDM.util.gaussian import MultivariateGaussian
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

    def _build_encoder_network(self):
        self.encoder = build_encoder(self._hp)
        self.action_head = nn.Linear(self.hidden_dim, self._hp["action_dim"])
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(self._hp["num_queries"], self.hidden_dim)
        if self._hp["use_image"]:
            raise ValueError("not implemented yet")
        elif self._hp["use_state"]:
            self.input_proj_robot_state = nn.Linear(self._hp["joint_dim"], self.hidden_dim)
            self.input_proj_env_state = nn.Linear(self._hp["env_dim"], self.hidden_dim)
            self.pos = torch.nn.Embedding(1, self.hidden_dim)
            self.backbones = None
        else:
            raise ValueError("Please choose at least one of state or image to use")
        self.cls_embed = nn.Linear(self._hp["hl_action_dim"], self.hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self._hp["action_dim"], self.hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(self._hp["joint_dim"], self.hidden_dim)  # project qpos to embedding
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+self._hp["num_queries"], self.hidden_dim)) # [CLS], qpos, a_seq
        self.register_buffer('state_pos_table', get_sinusoid_encoding_table(1, self.hidden_dim)) # [CLS], qpos, a_seq
        self.register_buffer('robotic_pos_table', get_sinusoid_encoding_table(1+self._hp["joint_dim"], self.hidden_dim)) 
        self.register_buffer('hl_pos_table', get_sinusoid_encoding_table(self._hp["hl_env_pos_dim"], self.hidden_dim)) 

    def _build_inference_network(self):
        self.latent_proj = nn.Linear(self.hidden_dim, self._hp["latent_dim"]*2) # project hidden state to latent std, var
        self.latent_out_proj = nn.Linear(self._hp["latent_dim"], self.hidden_dim) # project latent sample to embedding

    def _build_decoder_network(self):
        self.transformer = build_transformer(self._hp)
        self.hidden_dim = self.transformer.d_model
        if self._hp["use_robotic_dynamic"]:
            self.robotic_transformer_args = self._hp[self._hp['robotic_name']]
            self.base_xpos = torch.tensor(self._hp[self._hp['robotic_name']]["joint_base_xpose"]).float().cuda()
            self.encoder_joint_qpos_proj = nn.Linear(1, self.hidden_dim)  # project qpos to embedding
            self.encoder_joint_xpos_proj = nn.Linear(3, self.hidden_dim)  # project xpos to embedding
            self.robotic_dynamic_encoder = build_encoder(self.robotic_transformer_args)
            self.additional_pos_embed = nn.Embedding(1 + 1 + self._hp["joint_dim"], self.hidden_dim)
        else:
            self.additional_pos_embed = nn.Embedding(2, self.hidden_dim) # learned position embedding for proprio and latent

    def _build_prior_network(self):
        self.prior_proj_robot_state = nn.Linear(self._hp["joint_dim"], self.hidden_dim)
        self.prior_proj_env_state = nn.Linear(self._hp["env_dim"], self.hidden_dim)
        self.prior_proj_goal_state = nn.Linear(self._hp["goal_dim"], self.hidden_dim)
        self.prior_proj_action = nn.Linear(self._hp["hl_action_dim"], self.hidden_dim)
        self.prior_proj_endeffector = nn.Linear(self._hp["endeffector_dim"], self.hidden_dim)
        self.prior_encoder = build_encoder(self._hp.prior_inference)
        self.prior_latent_proj = nn.Linear(self.hidden_dim, self._hp["latent_dim"]*2)
        self.prior_latent_out_proj = nn.Linear(self._hp["latent_dim"], self.hidden_dim)

    def build_network(self):
        self._build_decoder_network()
        self._build_encoder_network()
        self._build_inference_network()
        self._build_prior_network()

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
        model_output = AttrDict()
        encoder_output = self.encode(inputs)
        latent_info = self.latent_proj(encoder_output)
        prior_latent_info = self.prior_inference(inputs)
        model_output.infer_latent_info = latent_info
        model_output.prior_latent_info = prior_latent_info
        prior_mu = prior_latent_info[:, :self._hp["latent_dim"]]
        prior_logvar = prior_latent_info[:, self._hp["latent_dim"]:]

        # inference actions z
        mu = latent_info[:, :self._hp["latent_dim"]]
        logvar = latent_info[:, self._hp["latent_dim"]:]
        latent_sample = self.reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)

        hs = self.decode(inputs, latent_input)
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        model_output.a_hat = a_hat
        model_output.is_pad_hat = is_pad_hat
        model_output.mu = mu
        model_output.logvar = logvar
        model_output.prior_mu = prior_mu
        model_output.prior_logvar = prior_logvar

        return model_output
    
    def val_mode(self, inputs):
        qpos = inputs['qpos']
        env_state = inputs['env_state']
        bs, _ = qpos.shape
        mu = logvar = None
        latent_sample = torch.zeros([bs, self._hp["latent_dim"]], dtype=torch.float32).to(qpos.device)
        latent_input = self.latent_out_proj(latent_sample)
        self.decode(qpos, env_state, latent_input)
    
    def encode(self, inputs):
        qpos = inputs['qpos']
        actions = inputs['actions']
        is_pad = inputs['is_pad']
        gole_state = inputs['goal_xpose']
        bs, _ = qpos.shape
        # project action sequence to embedding dim, and concat with a CLS token
        action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
        qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
        qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
        cls_embed = self.cls_embed(gole_state).detach()
        cls_embed = torch.unsqueeze(cls_embed, axis=1)  # (bs, 1, hidden_dim)
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
        return encoder_output

    def decode(self, env_inputs, latent_input):
        qpos = env_inputs['qpos']
        env_state = env_inputs['env_state']
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

    def prior_inference(self, inputs):
        now_state_robot = inputs['qpos']
        now_state_object = inputs['env_state']
        now_state_goal = inputs['env_goal']
        now_state_endeffector = inputs['endeffector_xpose']
        now_state_action = inputs['hl_actions']

        prior_qpos_embed = self.prior_proj_robot_state(now_state_robot)
        prior_env_state = self.prior_proj_env_state(now_state_object)
        prior_goal_state = self.prior_proj_env_state(now_state_goal).detach()
        prior_action_embed = self.prior_proj_action(now_state_action)
        prior_endeffector_embed = self.prior_proj_endeffector(now_state_endeffector)
        prior_qpos_embed = torch.unsqueeze(prior_qpos_embed, axis=1)
        prior_env_state = torch.unsqueeze(prior_env_state, axis=1)
        prior_goal_state = torch.unsqueeze(prior_goal_state, axis=1)
        prior_action_embed = torch.unsqueeze(prior_action_embed, axis=1)
        prior_endeffector_embed = torch.unsqueeze(prior_endeffector_embed, axis=1)
        prior_encoder_input = torch.cat([prior_goal_state, prior_qpos_embed, prior_env_state, prior_action_embed, prior_endeffector_embed], axis=1)
        prior_encoder_input = prior_encoder_input.permute(1, 0, 2)
        pos_embed = self.hl_pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)
        prior_infer_encoder_output = self.prior_encoder(prior_encoder_input, pos=pos_embed)[0]
        prior_infer_latent_info = self.prior_latent_proj(prior_infer_encoder_output)
        return prior_infer_latent_info
    
    def prior_loss(self, model_output):
        infer_multiGaussian = MultivariateGaussian(model_output.infer_latent_info)
        prior_multiGaussian = MultivariateGaussian(model_output.prior_latent_info)
        if self._hp.nll_prior_train:
            loss = NLL(breakdown=0)(prior_multiGaussian, infer_multiGaussian.detach())
        else:
            loss = KLDivLoss(breakdown=0)(infer_multiGaussian.detach(), prior_multiGaussian)
        return loss

    def loss(self, model_output, inputs):
        a_hat = model_output.a_hat
        is_pad_hat = model_output.is_pad_hat
        mu, logvar = model_output.mu, model_output.logvar
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        actions = inputs.actions
        is_pad = inputs.is_pad
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        if self._hp.skip_data is not None:
            skip_qpos_data = inputs.skip_qpos_data
            skip_state_data = inputs.skip_state_data
            now_traj, _, (_, _) = self.model(skip_qpos_data, None, skip_state_data, None, None)
            last_traj_queries = a_hat[:,self.skip_step:self.num_queries//2+self.skip_step]
            now_traj_queries = now_traj[:,:self.num_queries//2]
            self_traj_loss = F.l1_loss(last_traj_queries, now_traj_queries, reduction='mean')
        else:
            self_traj_loss = 0
        self.last_traj = a_hat
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['prior_loss'] = self.prior_loss(model_output)['value']
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self._hp.kl_weight + loss_dict['prior_loss']
        return loss_dict
    
    def _compute_learned_prior(self, inputs, first_only=False):
        pass
    
    def compute_learned_prior(self, inputs, first_only=False):
        pass
        # return MultivariateGaussian(prior_mdl(inputs))
    

if __name__ == "__main__":
    import yaml
    config_file = "/home/haowen/corl2024/configs/goal_tgdm.yaml"
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
    input_dict['env_goal'] = torch.randn(8, 30).to(torch.device('cuda'))
    input_dict['actions'] = torch.randn(8, param_dict.num_queries, 9).to(torch.device('cuda'))
    input_dict['is_pad'] = torch.randn(8, param_dict.num_queries).bool().to(torch.device('cuda'))
    input_dict['goal_xpose'] = torch.randn(8, 3).to(torch.device('cuda'))
    input_dict['endeffector_xpose'] = torch.randn(8, 3).to(torch.device('cuda'))
    input_dict['hl_actions'] = torch.randn(8, 3).to(torch.device('cuda'))
    model_output = model(input_dict)
    model.loss(model_output, input_dict)
    # model.val_mode(input_dict)

