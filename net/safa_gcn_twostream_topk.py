import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .sa_gcn_topk import Model as SA_GCN
from .sa_gcn_topk import Attention, STAttention
from .fa_gcn_topk import Model as FA_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = SA_GCN(*args, **kwargs)
        self.motion_stream = FA_GCN(*args, **kwargs)
        self.attn = Attention(in_dim=128, hid_dim=64)
        self.st_attn = STAttention(in_dim=128, hid_dim=64)
        # fcn for prediction
        self.fcn = nn.Conv2d(128, kwargs['num_class'], kernel_size=1)
        self.kappa = kwargs['kappa']

    def topK(self, scores):
        # scores.shape = N*T*num_class
        T = scores.shape[1]
        k = np.ceil(T/self.kappa).astype('int32')
        topk,_ = scores.topk(k, dim=1, largest=True, sorted = True)
        # video_level_score.shape = N*num_class
        video_level_score = topk.mean(axis=1, keepdim=False)
        return video_level_score
    
    def forward(self, x):
        # x = x.squeeze(dim=0)
        # N, C, T, V, M = x.size()
        # m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
        #                 x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
        #                 torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)
        # x = x.unsqueeze(dim=0)
        # m = m.unsqueeze(dim=0)
        # ori_res, ori_video_score = self.origin_stream(x)
        # motion_res, motion_video_score = self.motion_stream(m)
        st_feat = self.origin_stream(x)
        ff_feat = self.motion_stream(x)
        fuse_feat = torch.cat((st_feat, ff_feat), dim=2)
        # weak sup
        clip_x = fuse_feat.detach()
        clip_x = self.st_attn(clip_x)
        clip_x = self.fcn(clip_x)
        clip_x = clip_x.view(clip_x.size(0), clip_x.size(1))
        clip_x = clip_x.unsqueeze(dim = 0)
        video_level_score = self.topK(clip_x) 
        # N, dim
        return clip_x.squeeze(dim=0), video_level_score