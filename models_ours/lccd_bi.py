import torch
import torch.nn as nn
from models_ours.linear import Linear
from models_ours.cross_att import Basic_block
import models_ours
from models_ours import register
@register('lccd')
        
class LCCD(nn.Module):

    def __init__(self, encoder_spec_src, encoder_spec_0, no_imnet):
        super().__init__()
        self.encoder_0 = models_ours.make(encoder_spec_src)
        self.encoder_1 = models_ours.make(encoder_spec_0)
        self.f_dim = self.encoder_0.out_dim
        self.fusion = Basic_block(dim=self.f_dim, num_heads=8)
        self.linear_0 = Linear(in_dim=1536, out_dim=self.f_dim*2, hidden_list=[self.f_dim, self.f_dim, self.f_dim])
        self.linear_1 = Linear(in_dim=1536, out_dim=self.f_dim*2, hidden_list=[self.f_dim, self.f_dim, self.f_dim])
        self.sigmoid = nn.Sigmoid()
        if no_imnet:
            self.imnet = None
        else:
            self.imnet = Linear(in_dim=self.f_dim, out_dim=4, hidden_list=[self.f_dim*2, self.f_dim*2, self.f_dim, self.f_dim, 512, 256, 128, 64])

    def forward(self, src, tgt, prompt_src, prompt_tgt):
        #train together
        
        param_0_src = self.linear_0(prompt_src)
        param_0_src = self.sigmoid(param_0_src)
        alpha_0_src, beta_0_src = param_0_src[:, :, :self.f_dim], param_0_src[:, :, self.f_dim:]
        param_1_tgt = self.linear_1(prompt_tgt)
        param_1_tgt = self.sigmoid(param_1_tgt)
        alpha_1_tgt, beta_1_tgt = param_1_tgt[:, :, :self.f_dim], param_1_tgt[:, :, self.f_dim:]
        
        #src_tgt
        feat_0_src_tgt = self.encoder_0(src)
        content_src_tgt = (feat_0_src_tgt - beta_0_src.squeeze(1).unsqueeze(-1).unsqueeze(-1)) / alpha_0_src.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        content_src_tgt_1 = content_src_tgt * alpha_1_tgt.squeeze(1).unsqueeze(-1).unsqueeze(-1) + beta_1_tgt.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        pred_src_tgt = self.imnet(content_src_tgt_1.permute(0,2,3,1)).permute(0,3,1,2)

        #tgt_src
        feat_0_tgt_src = self.encoder_0(tgt)
        content_tgt_src = (feat_0_tgt_src - beta_1_tgt.squeeze(1).unsqueeze(-1).unsqueeze(-1)) / alpha_1_tgt.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        content_tgt_src_1 = content_tgt_src * alpha_0_src.squeeze(1).unsqueeze(-1).unsqueeze(-1) + beta_0_src.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        pred_tgt_src = self.imnet(content_tgt_src_1.permute(0,2,3,1)).permute(0,3,1,2)

        return pred_src_tgt, pred_tgt_src, content_src_tgt, content_tgt_src, feat_0_src_tgt, content_tgt_src_1, feat_0_tgt_src, content_src_tgt_1

       
