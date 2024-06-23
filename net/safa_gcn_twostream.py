import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .sa_gcn_FA import Model as SA_GCN
from .sa_gcn_FA import Attention, STAttention
from .fa_gcn_FA import Model as FA_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = SA_GCN(*args, **kwargs)
        self.motion_stream = FA_GCN(*args, **kwargs)
        self.attn = Attention(in_dim=128, hid_dim=64)
        self.st_attn = STAttention(in_dim=128, hid_dim=64)
        # fcn for prediction
        self.fcn = nn.Conv2d(128, kwargs['num_class'], kernel_size=1)

    def forward(self, x):
        # x = x.squeeze(dim=0)
        # N, C, T, V, M = x.size()
        # m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
        #                 x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
        #                 torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)
        # x = x.unsqueeze(dim=0)
        # m = m.unsqueeze(dim=0)
        st_feat = self.origin_stream(x)
        ff_feat = self.motion_stream(x)
        fuse_feat = torch.cat((st_feat, ff_feat), dim=2)
        # clip Attn
        x = self.attn(fuse_feat)
        # st Attn
        x = self.st_attn(x)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x