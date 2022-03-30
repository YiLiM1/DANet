import torch
import torch.nn as nn

class Efficientnet(nn.Module):
    def __init__(self):
        super(Efficientnet, self).__init__()

        basemodel_name='tf_efficientnet_b0_ap'
        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel=torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')
        print('Removing last two layers (global_pool & classifier).')
        basemodel.conv_head=nn.Identity()
        basemodel.bn2=nn.Identity()
        basemodel.act2=nn.Identity()
        basemodel.global_pool=nn.Identity()
        basemodel.classifier=nn.Identity()
        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')

        self.encoder= Encoder(basemodel)

    def forward(self, x):
        features=self.encoder(x)
        return features

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self,x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        features=[features[4], features[5], features[6], features[8], features[11]]
        return features
