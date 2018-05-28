"""
Capsule layer
The layer will encapsulate the primary and secondary layers
of a capsule layer, the input to the primry layer will usually
be an ouput from some convolutional layer
Author: Uddeshya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import caps_utils
from caps_utils import squash

class _CapsNd_primary(nn.Module):
    def __init__(self,
            primary_in_channels,
            num_primary_caps,
            primary_caps_dim,
            #params
            cuda_enabled = False,
            primary_kernel_size=3,
            primary_stride=1,
            primary_padding=1,
        ):
        super(_CapsNd_primary, self).__init__()
        self.primary_in_channels = primary_in_channels
        self.num_primary_caps = num_primary_caps
        self.primary_caps_dim = primary_caps_dim
        self.primary_kernel_size = primary_kernel_size
        self.primary_stride = primary_stride
        self.primary_padding = primary_padding
        self.cuda_enabled = cuda_enabled

class _CapsNd_secondary(nn.Module):
    def __init__(self,
            secondary_in_caps,
            secondary_in_dim,
            num_secondary_caps,
            secondary_caps_dim,
            #params
            cuda_enabled = False,
            num_routing = 3,
        ):
        super(_CapsNd_secondary, self).__init__()
        self.secondary_in_caps = secondary_in_caps
        self.secondary_in_dim = secondary_in_dim
        self.num_secondary_caps = num_secondary_caps
        self.secondary_caps_dim = secondary_caps_dim
        self.cuda_enabled = cuda_enabled
        self.num_routing = num_routing

################################################################
####################### 1 D Capsules API #######################
################################################################
class Caps1d_primary(_CapsNd_primary):
    """
        expected input tensor of shape:
        [batch_size, in_channels, height] (height is 1d data width)
        output is of the shape:
        [batch_size, 
        -1, 
        primary_caps_dim]

        When cuda_enabled=True then
        the expected input is cuda tensor wraped in Variable
    """
    def __init__(self,
            primary_in_channels,
            num_primary_caps,
            primary_caps_dim,
            #params
            cuda_enabled=False,
            primary_kernel_size=3,
            primary_stride=1,
            primary_padding=1,
        ):
        super(Caps1d_primary, self).__init__(
            primary_in_channels,
            num_primary_caps,
            primary_caps_dim,
            #params
            cuda_enabled,
            primary_kernel_size,
            primary_stride,
            primary_padding,
        )
        print('dbg primary_in_channels :', self.primary_in_channels, self.num_primary_caps[0], self.primary_caps_dim)
        self.primary_conv=nn.Conv1d(
            self.primary_in_channels,
            self.num_primary_caps[0]*self.primary_caps_dim,
            kernel_size=self.primary_kernel_size,
            stride=self.primary_stride,
            padding=self.primary_padding,
        )
        if self.cuda_enabled:
            self.primary_conv = self.primary_conv.cuda()
    
    def forward(self, x):
        batch_size = x.size(0)
        primary_out = self.primary_conv(x)
        # print('dbg interim size ', primary_out.shape)
        primary_out = primary_out.view(
            batch_size,
            -1,
            self.primary_caps_dim
        )
        return primary_out

class Caps1d_secondary(_CapsNd_secondary):
    """
        expected input tensor of shape:
        [batch_size, num_in_cpas, in_dim]
        output is of the shape:
        [batch_size, 
        num_secondary_caps, 
        secondary_caps_dim]

        When cuda_enabled=True then
        the expected input is cuda tensor wraped in Variable
    """
    def __init__(self,
            secondary_in_caps,
            secondary_in_dim,
            num_secondary_caps,
            secondary_caps_dim,
            #params
            cuda_enabled=False,
            num_routing=3
        ):
        super(Caps1d_secondary, self).__init__(
            secondary_in_caps,
            secondary_in_dim,
            num_secondary_caps,
            secondary_caps_dim,
            #params
            cuda_enabled,
            num_routing
        )
        self.weight = nn.Parameter(
            torch.randn(
                1,
                self.num_secondary_caps,
                self.secondary_in_caps,
                self.secondary_caps_dim,
                self.secondary_in_dim
            )
        )
        if self.cuda_enabled:
            self.weight = nn.Parameter(
                torch.randn(
                    1,
                    self.num_secondary_caps,
                    self.secondary_in_caps,
                    self.secondary_caps_dim,
                    self.secondary_in_dim
                ).cuda()
            )
    
    def forward(self, x):
        batch_size = x.size(0)
        _x = x.unsqueeze(1).unsqueeze(4) #_x_shape : [128,1,1152,8,1]
        # print('dbg weight_shape, _x_shape, x_shape : ', self.weight.shape, _x.shape, x.shape)
        u_hat = torch.matmul(self.weight, _x)
        # print('dbg u_hat shape ', u_hat.shape)
        u_hat = u_hat.squeeze(-1)
        temp_u_hat = u_hat
        # reference https://github.com/danielhavir/capsule-network/blob/master/capsules.py
        b = Variable(torch.zeros(batch_size, self.num_secondary_caps, self.secondary_in_caps, 1))
        if self.cuda_enabled:
            b = b.cuda()
        for _ in range(self.num_routing):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            # c = F.softmax(b, dim=1)
            c = F.softmax(b)
            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s) #will squash along dim_caps (i.e the last dimension)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b = b + uv	# this is not in-place, b+=uv creates problem	
		# # last iteration is done on the original u_hat, without the routing weights update
		# c = F.softmax(b, dim=1)
		# s = (c * u_hat).sum(dim=2)
		# apply "squashing" non-linearity along dim_caps
        # v = squash(s)
        return v

################################################################
####################### 2 D Capsules API #######################
################################################################
class Caps2d_primary(_CapsNd_primary):
    """
        expected input tensor of shape:
        [batch_size, in_channels, height, width] (height is 1d data width)
        output is of the shape:
        [batch_size, 
        -1, 
        primary_caps_dim]

        When cuda_enabled=True then
        the expected input is cuda tensor wraped in Variable
    """
    def __init__(self,
            primary_in_channels,
            num_primary_caps,
            primary_caps_dim,
            #params
            cuda_enabled=False,
            primary_kernel_size=3,
            primary_stride=1,
            primary_padding=1,
        ):
        super(Caps2d_primary, self).__init__(
            primary_in_channels,
            num_primary_caps,
            primary_caps_dim,
            #params
            cuda_enabled,
            primary_kernel_size,
            primary_stride,
            primary_padding,
        )
        # print('dbg primary_in_channels :', self.primary_in_channels)
        self.primary_conv=nn.Conv2d(
            self.primary_in_channels,
            self.num_primary_caps*self.primary_caps_dim,
            kernel_size=self.primary_kernel_size,
            stride=self.primary_stride,
            padding=self.primary_padding,
        )
        if self.cuda_enabled:
            self.primary_conv = self.primary_conv.cuda()
    
    def forward(self, x):
        batch_size = x.size(0)
        primary_out = self.primary_conv(x)
        # print('dbg interim size ', primary_out.shape)
        primary_out = primary_out.view(
            batch_size,
            -1,
            self.primary_caps_dim
        )
        return primary_out

class Caps2d_secondary(_CapsNd_secondary):
    """
        expected input tensor of shape:
        [batch_size, num_in_cpas, in_dim]
        output is of the shape:
        [batch_size, 
        num_secondary_caps, 
        secondary_caps_dim]

        When cuda_enabled=True then
        the expected input is cuda tensor wraped in Variable
    """
    def __init__(self,
            secondary_in_caps,
            secondary_in_dim,
            num_secondary_caps,
            secondary_caps_dim,
            #params
            cuda_enabled=False,
            num_routing=3
        ):
        super(Caps2d_secondary, self).__init__(
            secondary_in_caps,
            secondary_in_dim,
            num_secondary_caps,
            secondary_caps_dim,
            #params
            cuda_enabled,
            num_routing
        )
        self.weight = nn.Parameter(
            torch.randn(
                1,
                self.num_secondary_caps,
                self.secondary_in_caps,
                self.secondary_caps_dim,
                self.secondary_in_dim
            )
        )
        if self.cuda_enabled:
            self.weight = nn.Parameter(
                torch.randn(
                    1,
                    self.num_secondary_caps,
                    self.secondary_in_caps,
                    self.secondary_caps_dim,
                    self.secondary_in_dim
                ).cuda()
            )
    
    def forward(self, x):
        batch_size = x.size(0)
        _x = x.unsqueeze(1).unsqueeze(4) #_x_shape : [128,1,1152,8,1]
        # print('dbg weight_shape, _x_shape, x_shape : ', self.weight.shape, _x.shape, x.shape)
        u_hat = torch.matmul(self.weight, _x)
        # print('dbg u_hat shape ', u_hat.shape)
        u_hat = u_hat.squeeze(-1)
        temp_u_hat = u_hat
        # reference https://github.com/danielhavir/capsule-network/blob/master/capsules.py
        b = Variable(torch.zeros(batch_size, self.num_secondary_caps, self.secondary_in_caps, 1))
        if self.cuda_enabled:
            b = b.cuda()
        for _ in range(self.num_routing):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            # c = F.softmax(b, dim=1)
            c = F.softmax(b)
            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s) #will squash along dim_caps (i.e the last dimension)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b = b + uv	# this is not in-place, b+=uv creates problem	
		# # last iteration is done on the original u_hat, without the routing weights update
		# c = F.softmax(b, dim=1)
		# s = (c * u_hat).sum(dim=2)
		# apply "squashing" non-linearity along dim_caps
        # v = squash(s)
        return v
