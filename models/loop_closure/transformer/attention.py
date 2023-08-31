"""
Spatial_Attn is modified based on 
https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
"""

import torch
import torch.nn as nn


class Spatial_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Spatial_Attn, self).__init__()
        self.chanel_in = in_dim
        if in_dim == 16:
            out_dim = in_dim // 2
        elif in_dim == 64:
            out_dim = in_dim * 2
        else:
            out_dim = in_dim // 8
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x N
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X N X N
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma*out + x
        return out, attention


class Channel_Attn(nn.Module):
    """ Self channel attention Layer"""
    def __init__(self, in_dim, type):
        super(Channel_Attn, self).__init__()
        if type == "add":
            self.softmax = nn.Softmax(dim=-1)
            self.gamma = nn.Parameter(torch.zeros(1))
        elif type == "reweight":
            self.sigmoid = nn.Sigmoid()
            self.pool = nn.AdaptiveAvgPool2d((in_dim, 1))
            self.fc = nn.Linear(in_dim, in_dim)
        self.type = type
    
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : attention weight + input feature(element-wise)
                attention: B X C X C (C is channel number)
        """
        m_batchsize,C,N = x.size()
        energy = torch.bmm(x, x.permute(0,2,1)) # transpose check
        
        if self.type == "add":
            attention = self.softmax(energy) # B X C X C
            value = torch.bmm(attention.permute(0,2,1), x) # B*C*N
            out = self.gamma*value + x
        elif self.type == "reweight":
            weight = self.pool(energy)
            weight = torch.bmm(energy, weight)
            weight = self.fc(x.squeeze(-1))
            weight = self.sigmoid(weight)
            out = x.view(m_batchsize, C)*weight
        out = out.view(m_batchsize, -1)
        return out