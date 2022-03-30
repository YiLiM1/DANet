import math
import torch
from torch import nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,kernel_size=3,stride=1,padding=0):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size,stride=stride ,padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class PST_Path(nn.Module):
    def __init__(self, in_channels, patch_size=[], embedding_dim=128, num_heads=4,num_layers=3,args=None):
        super(PST_Path, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)  # takes shape S,N,E

        h,w=args.height,args.width
        sx,sy=math.floor(w/patch_size[1]),math.floor(h/patch_size[0])
        kx,ky=(w-sx*(patch_size[1]-1)), (h-sy*(patch_size[0]-1))

        self.embedding_convPxP=depthwise_separable_conv(in_channels, embedding_dim, (ky,kx), (sy,sx))
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        embeddings = nn.functional.pad(embeddings, (1,0,0,0,0,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x
