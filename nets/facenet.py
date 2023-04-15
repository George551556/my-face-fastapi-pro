import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1

# 导入注意力模块
from .attention import se_block, cbam_block, CA_Block
attention_blocks = [se_block, cbam_block, CA_Block]

class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x) # 160,160,3 -> 20,20,256
        x = self.model.stage2(x) # 20,20,256 -> 10,10,512
        x = self.model.stage3(x) # 10,10,512 -> 5,5,1024
        return x

class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x
        
class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False, phi=0):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            # 这个全连接层，即是分类器的构建。作用：用于辅助Triplet Loss的收敛
            self.classifier = nn.Linear(embedding_size, num_classes)# num_classes:对输出的图片作num_classes个的分类
        #attention 模块----------------------------
        self.phi        = phi
        if phi>1 and phi<3:
            self.x_attention = attention_blocks[phi-1](1024)
        #attention 模块----------------------------


    def forward(self, x, mode = "predict"):
        if mode == 'predict':
            x = self.backbone(x)
            #attention 模块----------------------------
            if self.phi>1 and self.phi<3:
                x_copy_2 = x
                x = self.x_attention(x)
                x = x+x_copy_2
            #attention 模块----------------------------
            x = self.avg(x)
            if self.phi>1 and self.phi<3:
                x_copy_3 = x
                x = self.x_attention(x)#attattattatt
                x = x+x_copy_3
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)# 全连接，输出长度为128
            x = self.last_bn(x)# batch_normal
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.backbone(x)
        #attention 模块----------------------------
        if self.phi>1 and self.phi<3:
            x_copy = x
            x = self.x_attention(x)
            x = x+x_copy
        #attention 模块----------------------------
        x = self.avg(x)
        if self.phi>1 and self.phi<3:
            x_copy_1 = x
            x = self.x_attention(x)#attattattatt
            x = x+x_copy_1
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)# 全连接，输出长度为128
        before_normalize = self.last_bn(x)
        
        # 这个L2标准化是为了使得不同人脸的特征向量可以属于同一数量级，方便比较
        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x
