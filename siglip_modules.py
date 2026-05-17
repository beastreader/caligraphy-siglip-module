from TransformerModels import OptimizedDepthwiseConv,OptimizedPointwiseConv,FusedDepthwisePointwise
from FastTransformer import FastEncoder
import torch
import torch.nn as nn

##____ Img  Encoder ____
class Image_encoder(nn.Module):
    def __init__(self,inputdim=1,dropout=0.1,res_blocks=3):
        super().__init__()
        base_dim = 32
        N = 3
        self.layers = [OptimizedPointwiseConv(inputdim,base_dim,False,activation='silu'),nn.GroupNorm(16,base_dim)]
        #______ Downsample Blocks _______
        for _ in range(N):
            self.layers.append(OptimizedDepthwiseConv(base_dim*(_+1),3,2,1,False,dropout,activation='silu'))
            self.layers.append(nn.GroupNorm(16,base_dim*(_+1)))
            self.layers.append(OptimizedPointwiseConv(base_dim*(_+1),base_dim*(_+2),False,activation='silu'))
            self.layers.append(nn.GroupNorm(16,base_dim*(_+2)))
        #______ Res Blocks _______
        for _ in range(res_blocks):
            self.layers.append(FusedDepthwisePointwise(base_dim*(N+1),dropout=dropout,alpha_init=1))
            self.layers.append(nn.GroupNorm(16,base_dim*(N+1)))
        self.layers = nn.Sequential(*self.layers)
        #_____ Projection Layer ____
        self.proj = nn.Sequential(nn.Linear(base_dim*2*(N+1),base_dim*4*(N+1)),nn.BatchNorm1d(base_dim*4*(N+1)),nn.SiLU(),nn.Linear(base_dim*4*(N+1),base_dim*(N+1)),nn.BatchNorm1d(base_dim*(N+1)))
        #_____ config _____
        self.config = {'inputdim':inputdim,'dropout':dropout,'res_blocks':res_blocks}
    def forward(self,x,proj_pool=True):
        embeded = self.layers(x)
        if proj_pool:
            falttend = embeded.flatten(start_dim=2)
            return self.proj(torch.cat((falttend.mean(-1),falttend.amax(-1)),-1))
        else:
            return embeded
##____ Txt  Encoder ____
class Text_encoder(nn.Module):
    def __init__(self,N=100,dropout=0.1,no_blocks=3):
        super().__init__()
        self.config = {'N':N,'dropout':dropout,'no_blocks':no_blocks}
        dim = 128
        #____ Embed Layer ____
        self.embed = nn.Embedding(N,dim)
        #_____ Transformer Layers _____
        self.layers = FastEncoder(dim,no_blocks,4,dropout=dropout,ff_dropout=dropout,rope_type='1d')
        #__ Cls init __
        self.cls = nn.Parameter(torch.randn(1,1,dim) * 0.02)
        #Proj
        self.proj = nn.Sequential(nn.Linear(dim,dim*2),nn.BatchNorm1d(dim*2),nn.SiLU(),nn.Linear(dim*2,dim),nn.BatchNorm1d(dim))

    def forward(self,x,proj_pool=True):
        B,_ = x.shape
        mask = (x != 1)
        embeded = self.embed(x) * mask.unsqueeze(-1)
        if proj_pool:
            return self.proj(self.layers(torch.cat([self.cls.repeat(B,1,1),embeded],1),text_mask=torch.cat([torch.ones_like(mask[:,:1]),mask],1))[:,0])
        else:
            return self.layers(torch.cat([self.cls.repeat(B,1,1),embeded],1),text_mask=torch.cat([torch.ones_like(mask[:,:1]),mask],1))