import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph_new import Graph

class FAGCNModel(nn.Module):
    r"""Frequency Attention graph convolutional networks

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.fa_gcn_networks = nn.ModuleList((
            fa_gcn(in_channels, 32, kernel_size, 1, residual=False, **kwargs0),
            fa_gcn(32, 64, kernel_size, 1, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.fa_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.fa_gcn_networks)
            
        self.st_attn = Attention(in_dim=64, hid_dim=64)
        # fcn for prediction
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        
        # data normalization
        N, C, T, V, M = x.size()
        T = T // 2  # Frequency 
        x = torch.fft.fft(x, dim=2).real[:, :, :T, :, :]  # FFT
        
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.fa_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        x = x.view(N, c, t, v).permute(0, 1, 2, 3)#
        x = F.avg_pool2d(x, (x.size()[2], 1))
        x = x.squeeze(dim=2).permute(0, 2, 1)

        # Mask Attention
        x = self.st_attn(x)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.fa_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
    
class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim, mask_proportion=0.5):
        super(Attention, self).__init__()
        self.mask_proportion = mask_proportion  # Hyperparameter to control mask
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        e = self.attention(x)
        # Sort 
        sorted_e, _ = torch.sort(e, dim=1)
        T = e.size(1)
        # the threshold index mask proportion
        threshold_index = int(T * self.mask_proportion)
        threshold_value = sorted_e[:, threshold_index]

        # Masking high-frequency weights
        mask = torch.ones_like(e)
        mask[e >= threshold_value.unsqueeze(1)] = -99999999
        e = e + mask
        beta = torch.softmax(e, dim=1)
        x = torch.bmm(torch.transpose(beta, 1, 2), x).permute(0, 2, 1)
        x = x.unsqueeze(dim=3)
        return x

class fa_gcn(nn.Module):
    r"""Applies a frequency graph convolution over an input graph sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.25,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        # x = self.tcn(x) + res
        x = res + x
        return self.relu(x), A