import torch.nn.functional as F
import torch.nn as nn
class E_mobilenet(nn.Module):
    def __init__(self, original_model):
        super(E_mobilenet, self).__init__()
        self.feature=original_model.features
        self.conv=original_model.conv
    def forward(self, x):
        for i in range(18):
            x=self.feature[i](x)
            if i==1:
                x_block0=x
            elif i==3:
                x_block1=x
            elif i==6:
                x_block2=x
            elif i==13:
                x_block3=x
            elif i==17:
                x_block4=x
        feature_pyramid=[x_block0, x_block1, x_block2, x_block3, x_block4]
        return feature_pyramid
class E_resnet(nn.Module):
    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_block0 = x

        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]

        return feature_pyramid


class Refineblock(nn.Module):
    def __init__(self, num_features, kernel_size,in_channels=1,out_channels=1,is_res=True):
        super(Refineblock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.is_res=is_res
        padding=(kernel_size-1)//2

        self.conv1 = nn.Conv2d(in_channels, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features//2)

        self.conv2=nn.Conv2d(num_features//2, num_features//2,kernel_size=kernel_size, stride=1,padding=padding,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features//2)

        self.conv3 = nn.Conv2d(num_features//2, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=True)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.bn1(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv2(x_res)
        x_res = self.bn2(x_res)
        inter = F.relu(x_res)

        x_res = self.conv3(inter)

        if self.is_res==True:
            x2 = x  + x_res
            return inter, x2
        else:
            return inter, x_res

class Decoder(nn.Module):
    def __init__(self, num_features=[],decoder_feature=16):
        super(Decoder, self).__init__()
        self.bin_nums=256
        self.conv=nn.Sequential(
            nn.Conv2d(num_features[-1], decoder_feature, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_feature),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_feature, decoder_feature, kernel_size=1, stride=1, bias=False)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(num_features[-2], decoder_feature, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_feature),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_feature, decoder_feature, kernel_size=1, stride=1, bias=False))

        self.scale4=Refineblock(num_features=decoder_feature*4, kernel_size=3,in_channels=decoder_feature,out_channels=decoder_feature)

        self.conv3=nn.Sequential(
            nn.Conv2d(num_features[-3], decoder_feature, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_feature),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_feature, decoder_feature, kernel_size=1, stride=1, bias=False))

        self.scale3=Refineblock(num_features=decoder_feature*4, kernel_size=3,in_channels=decoder_feature,out_channels=decoder_feature)

        self.conv2=nn.Sequential(
            nn.Conv2d(num_features[-4], decoder_feature, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_feature),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_feature, decoder_feature, kernel_size=1, stride=1, bias=False))
        self.scale2=Refineblock(num_features=decoder_feature*4, kernel_size=3,in_channels=decoder_feature,out_channels=decoder_feature)

        self.conv1=nn.Sequential(
            nn.Conv2d(num_features[-5], decoder_feature, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_feature),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_feature, decoder_feature, kernel_size=1, stride=1, bias=False))

        self.scale1=Refineblock(num_features=decoder_feature*4, kernel_size=3,in_channels=decoder_feature,out_channels=decoder_feature)

    def forward(self, feature_pyramid):
        scale1_size=[feature_pyramid[0].size(2), feature_pyramid[0].size(3)]
        scale2_size=[feature_pyramid[1].size(2), feature_pyramid[1].size(3)]
        scale3_size=[feature_pyramid[2].size(2), feature_pyramid[2].size(3)]
        scale4_size=[feature_pyramid[3].size(2), feature_pyramid[3].size(3)]
        scale5_size=[feature_pyramid[4].size(2), feature_pyramid[4].size(3)]

        # scale5
        scale5_depth=self.conv(feature_pyramid[4])
        # scale4
        scale4_res=self.conv4(feature_pyramid[3])
        scale5_upx2=F.interpolate(scale5_depth, size=scale4_size,mode='bilinear', align_corners=True)
        #scale5_upx2=F.interpolate(scale5_depth, size=scale4_size, mode='nearest')
        ignore,scale4_depth=self.scale4(scale4_res+scale5_upx2)
        # scale3
        scale3_res=self.conv3(feature_pyramid[2])
        scale4_upx2=F.interpolate(scale4_depth, size=scale3_size,mode='bilinear', align_corners=True)
        #scale4_upx2=F.interpolate(scale4_depth, size=scale3_size, mode='nearest')
        ignore,scale3_depth=self.scale3(scale3_res+scale4_upx2)

        # scale2
        scale2_res=self.conv2(feature_pyramid[1])
        scale3_upx2=F.interpolate(scale3_depth, size=scale2_size,mode='bilinear', align_corners=True)
        #scale3_upx2=F.interpolate(scale3_depth, size=scale2_size, mode='nearest')
        ignore,scale2_depth=self.scale2(scale2_res+scale3_upx2)

        # scale1
        scale1_res=self.conv1(feature_pyramid[0])
        scale2_upx2=F.interpolate(scale2_depth, size=scale1_size,mode='bilinear', align_corners=True)
        #scale2_upx2=F.interpolate(scale2_depth, size=scale1_size, mode='nearest')
        ignore,scale1_depth=self.scale1(scale1_res+scale2_upx2)

        scale_res=[scale1_res, scale2_res, scale3_res, scale4_res]
        scale_depth=[scale5_depth,scale4_depth,scale3_depth,scale2_depth,scale1_depth]

        # return scale_res+scale_depth
        return scale_depth