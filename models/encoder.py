import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common import Conv2dBNAct, C3,ResidualConv2dBNAct
import torch.nn as nn
import numpy as np
import math
from safetensors.torch import load_file
from models.common import Upsample_ConvBlock


class Backbone(nn.Module):
    def __init__(self, 
                 imgs_color,
                 num_features,
                 d_model,
                 downsampling,
                 num_backbone_layes,
                 cnn_act='gelu',
                 cnn_norm: str="instancenorm",
                 cnn_conv2d_has_bias=False,
                 cnn_bn_has_bias=True,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        assert imgs_color=="gray" or imgs_color =="rgb" ,"imgs_color should be gray or rgb"
        self.in_channels = 1 if imgs_color == "gray" else 3
        
        # 使用第一个网络的结构定义
        self.block1 = nn.Sequential(
            Conv2dBNAct(self.in_channels, num_features // 2, 5, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
            Conv2dBNAct(num_features//2, num_features, 3, 2, act=cnn_act,norm=cnn_norm),
        )
        num_layers = int(np.log2(downsampling//4))

        self.block2 = nn.ModuleList([Conv2dBNAct(num_features,num_features, 3, 1,act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_backbone_layes)])
        
        self.block3 = nn.ModuleList([Conv2dBNAct(num_features,num_features, 3, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_layers)])
       
        self.block4 = C3(num_features, num_features, act=cnn_act, norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)
        
        self.proj_layer = Conv2dBNAct(num_features,d_model, 1, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)

    def forward(self,x):
        
        res = []
        # 分别处理block1中的两个层，以便收集中间特征
        x = self.block1[0](x)  # 第一个Conv2dBNAct层
        res.append(x)
        x = self.block1[1](x)  # 第二个Conv2dBNAct层
        res.append(x)
        
        for layer in self.block2:
            x = layer(x)
            res.append(x)

        for layer in self.block3:
            x = layer(x)
            res.append(x)

        x = self.block4(x)
        res[-1] = x
        x = self.proj_layer(x)
        
        return x, res[::-1]

class Backbone_16842_2222(nn.Module):
    def __init__(self, 
                 imgs_color,
                 num_features,
                 d_model,
                 downsampling,
                 num_backbone_layes,
                 cnn_act='gelu',
                 cnn_norm: str="instancenorm",
                 cnn_conv2d_has_bias=False,
                 cnn_bn_has_bias=True,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        assert imgs_color=="gray" or imgs_color =="rgb" ,"imgs_color should be gray or rgb"
        self.in_channels = 1 if imgs_color == "gray" else 3
        
        # 使用第一个网络的结构定义
        self.block1 = nn.Sequential(
            Conv2dBNAct(self.in_channels, num_features, 5, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
            Conv2dBNAct(num_features, num_features, 3, 2, act=cnn_act,norm=cnn_norm),
        )
        num_layers = int(np.log2(downsampling//4))

        self.block2 = nn.ModuleList([Conv2dBNAct(num_features,num_features, 3, 1,act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_backbone_layes)])
        
        self.block3 = nn.ModuleList([Conv2dBNAct(num_features,num_features, 3, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_layers)])
       
        self.block4 = C3(num_features, num_features, act=cnn_act, norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)
        
        self.proj_layer = Conv2dBNAct(num_features,d_model, 1, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)

    def forward(self,x):
        
        res = []
        # 分别处理block1中的两个层，以便收集中间特征
        x = self.block1[0](x)  # 第一个Conv2dBNAct层
        res.append(x)
        x = self.block1[1](x)  # 第二个Conv2dBNAct层
        res.append(x)
        
        for layer in self.block2:
            x = layer(x)
            res.append(x)

        for layer in self.block3:
            x = layer(x)
            res.append(x)

        x = self.block4(x)
        res[-1] = x
        x = self.proj_layer(x)
        
        return x, res[::-1]

class Backbone_421(nn.Module):
    def __init__(self, 
                 imgs_color,
                 num_features,
                 d_model,
                 downsampling,
                 num_backbone_layes,
                 cnn_act='gelu',
                 cnn_norm: str="batchnorm",
                 cnn_conv2d_has_bias=False,
                 cnn_bn_has_bias=True,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        assert imgs_color in ["gray", "rgb"], "imgs_color should be gray or rgb"
        assert downsampling >= 4 and math.log2(downsampling).is_integer(), "downsampling should be a power of 2 and >= 4"
        
        self.in_channels = 1 if imgs_color == "gray" else 3
        self.downsampling = downsampling

        self.block0 = nn.Sequential(
            Conv2dBNAct(self.in_channels, num_features, 5, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
        )
        
        self.block1 = nn.Sequential(
            ResidualConv2dBNAct(num_features, num_features, 5, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
        )

        num_additional_layers = int(math.log2(downsampling//2))

        self.block2 = nn.ModuleList([ResidualConv2dBNAct(num_features, num_features, 3, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_additional_layers)])
       
        self.block3 = C3(num_features, num_features, act=cnn_act)

        self.proj_layer = Conv2dBNAct(num_features,d_model, 1, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)
        
    def forward(self, x):
        features = []
        
        x = self.block0(x)
        features.append(x)  # 1x

        x = self.block1(x)
        features.append(x)  # 2x

        for layer in self.block2:
            x = layer(x)
            features.append(x)

        x = self.block3(x)
        features[-1] = x  
        x = self.proj_layer(x)
        
        return x, features[-3:][::-1]

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, cnn_norm='group', stride=1):
        super(ResidualBlock, self).__init__()
        """
        contain 2 conv layer, and conv1 invole with stride more than 1
        """

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if cnn_norm == 'groupnorm':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif cnn_norm == 'batchnorm':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif cnn_norm == 'instancenorm':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes)

        elif cnn_norm == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), 
            self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
    
class McNet_Encoder(nn.Module):
    def __init__(self, num_features,d_model, cnn_norm='instancenorm', dropout=0.0,**kwagrs):
        super(McNet_Encoder, self).__init__()
        self.cnn_norm = cnn_norm

        c1 = 16
        
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(c1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = c1
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(48, stride=2) #down sample 2 times
        self.layer3 = self._make_layer(64, stride=2) #down sample 2 times
        # self.layer4 = self._make_layer(80, stride=2)
        
        
        self.out_conv1 = nn.Conv2d(32, num_features, kernel_size=1)
        self.out_conv2 = nn.Conv2d(48, num_features, kernel_size=1)
        self.out_conv3 = nn.Conv2d(64, num_features, kernel_size=1)  

        self.out_conv_last_scale = nn.Conv2d(num_features,d_model, kernel_size=1)     
        
        # self.out_conv4 = nn.Conv2d(80, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.cnn_norm, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.cnn_norm, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x_128 = self.out_conv1(x)
        
        x = self.layer2(x)
        x_64 = self.out_conv2(x)
        
        x = self.layer3(x)
        x_32 = self.out_conv3(x)
        x_32_d_model = self.out_conv_last_scale(x_32)
        
        return x_32_d_model,[x_32,x_64,x_128]


class Dinov3_multiscale_encoder(nn.Module):
    # 修正点1：参数名改为 d_model，与内部保持一致
    def __init__(self, num_features, d_model, dinov3_version='s', cnn_norm='instancenorm', cnn_act='gelu', cnn_conv2d_has_bias=False, cnn_bn_has_bias=True):
        super(Dinov3_multiscale_encoder, self).__init__()
        
        # 根据模型大小配置不同的参数
        model_configs = {
            's': {
                'path': "./ckpt/dinov3-vits16-pretrain-lvd1689m",
                'feature_dim': 384
            },
            'l': {
                'path': "./ckpt/dinov3-vitl16-pretrain-lvd1689m",
                'feature_dim': 1024
            }
        }
        
        if dinov3_version not in model_configs:
            raise ValueError(f"dinov3_version must be 's' or 'l', got '{dinov3_version}'")
        
        config_info = model_configs[dinov3_version]
        model_path = config_info['path']
        dinov3_feature_dim = config_info['feature_dim']
        
        # 加载对应的模型
        config = DINOv3ViTConfig.from_pretrained(model_path)
        dinov3_model = DINOv3ViTModel(config)
        
        state_dict = load_file(f"{model_path}/model.safetensors")
        dinov3_model.load_state_dict(state_dict)
      
        self.encoder = dinov3_model
        # 这里的 mask_token 也是视具体实现而定，通常没问题
        self.encoder.embeddings.mask_token.requires_grad = False 
        
        for name, param in self.encoder.named_parameters():
            if any(x in name for x in ['layer.9.', 'layer.10.', 'layer.11.']):
                param.requires_grad = True
            elif 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 使用动态的特征维度
        self.dinov3_channels_to_dmodel_proj_layer = Conv2dBNAct(
            dinov3_feature_dim, d_model, 
            kernel_size=1, stride=1, 
            act=cnn_act, norm=cnn_norm, 
            conv_has_bias=cnn_conv2d_has_bias, bn_has_bias=cnn_bn_has_bias
        )

        self.d_model_to_num_features_proj_layer = Conv2dBNAct(
            d_model, num_features, 
            kernel_size=1, stride=1, 
            act=cnn_act, norm=cnn_norm, 
            conv_has_bias=cnn_conv2d_has_bias, bn_has_bias=cnn_bn_has_bias
        )

        self.upsample_16_to_8 = Upsample_ConvBlock(num_features, num_features, norm=cnn_norm, act=cnn_act)
        self.upsample_8_to_4 = Upsample_ConvBlock(num_features, num_features, norm=cnn_norm, act=cnn_act)
    
    def reshape_patch_features_to_spatial(self, patch_features, img_height, img_width, patch_size=16):
        batch_size, num_patches, feature_dim = patch_features.shape
        
        patches_h = int(img_height) // patch_size
        patches_w = int(img_width) // patch_size
        
        spatial_features = patch_features.transpose(1, 2).contiguous()
        spatial_features = spatial_features.reshape(batch_size, feature_dim, patches_h, patches_w)
        
        return spatial_features

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x_16 = self.encoder(x)
        x_16 = x_16.last_hidden_state[:, 5:, :]
        x_16 = self.reshape_patch_features_to_spatial(x_16, h, w)

        x_16_d_model = self.dinov3_channels_to_dmodel_proj_layer(x_16)
        
        x_16 = self.d_model_to_num_features_proj_layer(x_16_d_model)
        x_8 = self.upsample_16_to_8(x_16)
        x_4 = self.upsample_8_to_4(x_8)
    

        return x_16_d_model, [x_16, x_8, x_4]
        
        

if __name__ == "__main__":
   
    test_cas_vit_backbone()