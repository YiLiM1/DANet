from models import modules
from models import get_models
from models.miniViT import *
import torch.nn.functional as F

class DANet(nn.Module):
	def __init__(self, args):
		super(DANet, self).__init__()
		print("backbone:", args.backbone)
		self.feature_extraction = get_models(args)
		self.max_depth=args.max_depth
		self.min_depth=args.min_depth
		if args.backbone in ["MobileNetV2"]:
			block_channel = [16, 24, 32, 96, 320]

		if args.backbone in ["EfficientNet"]:
			block_channel = [16, 24, 40, 112, 320]

		if args.backbone in ["ResNet18", "ResNet34"]:
			block_channel = [64, 64, 128, 256, 512]

		if args.backbone in ["ResNet50", "ResNet101", "ResNet152"]:
			block_channel = [64, 256, 512, 1024, 2048]

		self.adaptive_bins_layer=Pyramid_Scene_Transformer(in_channels=block_channel[-1], num_layers=3,norm='linear',sizes=[[8,10],[4,5],[2,3]],args=args)
		self.decoder=modules.Decoder(block_channel, decoder_feature=16)
		self.conv_out=nn.Sequential(nn.Conv2d(16, 256, kernel_size=1, stride=1, padding=0))

	def forward(self, x):
		feature_pyramid=self.feature_extraction(x)
		bin_widths_normed, feature_PST=self.adaptive_bins_layer(feature_pyramid[-1])

		feature_pyramid[-1]=feature_PST
		multiscale_depth=self.decoder(feature_pyramid)
		range_attention_maps=F.interpolate(multiscale_depth[-1], scale_factor=2, mode='bilinear',align_corners=True)
		out=self.conv_out(range_attention_maps)
		out=F.softmax(out, dim=1)  # softmax
		bin_widths=(self.max_depth-self.min_depth)*bin_widths_normed  # .shape = N, dim_out
		bin_widths=nn.functional.pad(bin_widths, (1, 0), mode='constant', value=0.01)
		bin_edges=torch.cumsum(bin_widths, dim=1)
		centers=0.5*(bin_edges[:, :-1]+bin_edges[:, 1:])
		n, dout=centers.size()
		centers=centers.view(n, dout, 1, 1)
		pred=torch.sum(out*centers, dim=1, keepdim=True)
		return bin_edges, [pred]