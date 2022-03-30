import os
from models.modules import E_resnet,E_mobilenet
from models.Efficientnet import Efficientnet
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenet import mobilenetv2
__models__ = {
    'ResNet18': lambda :E_resnet(resnet18(pretrained = True)),
	'ResNet34': lambda :E_resnet(resnet34(pretrained = True)),
	'ResNet50': lambda :E_resnet(resnet50(pretrained = True)),
	'ResNet101': lambda :E_resnet(resnet101(pretrained = True)),
	'ResNet152': lambda :E_resnet(resnet152(pretrained = True)),
    'MobileNetV2': lambda: E_mobilenet(mobilenetv2(pretrained="imagenet")),
    'EfficientNet':lambda :Efficientnet(),
}
def get_models(args):
    backbone = args.backbone
    if os.getenv('TORCH_HOME') != args.pretrained_dir:
        os.environ['TORCH_HOME'] = args.pretrained_dir
    else:
        pass
    return __models__[backbone]()

