import torch
import torch.nn as nn
from .layers import *
import torch.nn.functional as F
class Pyramid_Scene_Transformer(nn.Module):
    def __init__(self, in_channels, embedding_dim=160,num_heads=4, norm='linear',num_layers=4,sizes=[[8,10],[4,5],[2,3]],args=None):
        super(Pyramid_Scene_Transformer, self).__init__()
        self.norm = norm
        self.in_channels=in_channels
        self.sizes=sizes
        self.embedding_dim = embedding_dim
        self.paths=nn.ModuleList([self._make_stage(in_channels, self.embedding_dim, size, num_heads, num_layers,args) for size in self.sizes])
        self.bottleneck=nn.Sequential(
            nn.Conv2d(self.embedding_dim*len(self.sizes), in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU()
        )
        self.regressor=nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
        )

    def _make_stage(self, in_channels,embedding_dim, patch_size, num_heads,num_layers,args):
        tranformer=PST_Path(in_channels, patch_size, embedding_dim, num_heads,num_layers,args)
        return nn.Sequential(tranformer)

    def forward(self, x):
        b, c, h, w=x.shape
        priors=[path(x) for path in self.paths]
        bin=priors[0][0,...]
        # Change from S, N, E to N, S, E
        for i in range(len(priors)):
            priors[i]=priors[i][1:,...].permute(1, 2, 0).reshape(b,self.embedding_dim,*self.sizes[i])
            #priors[i]=F.interpolate(priors[i],size=(h,w),mode='bilinear',align_corners=True)
            priors[i]=F.interpolate(priors[i], size=(h, w), mode='nearest')

        bottle = self.bottleneck(torch.cat(priors, 1))
        y=self.regressor(bin)
        eps=0.1
        y=y+eps
        y=y/y.sum(dim=1, keepdim=True)

        return y,bottle




