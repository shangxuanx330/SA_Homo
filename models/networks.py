import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common import Conv2dBNAct, Bottleneck,ResidualConv2dBNAct
from models.utils.position_encoding import PositionEncodingSine
from models.attention_module.transformer import  Multiscale_Linear_attention
from models.homography_transformed import disp_to_coords_p2w
import torch.nn.functional as F
from models.decoder import *

from models.encoder import *

import kornia
import models.corr_implement as corr_implement    


class SCEM(nn.Module):
    def __init__(
        self, 
        num_features=96, 
        downsampling = 16,
        color_format_search='rgb',
        template_size=(768,768),
        search_size=(768,768),
        max_shape=(256,256),
        d_model = 256,
        SCEM_num_features_predition_head=256,
        n_heads = 4,
        layer_names=['self', 'cross'] * 2,
        head_kernel_size=1,
        num_backbone_layes=1,
        attention_drop_out_rate=0.0,
        num_of_predited_pts = 144,
        cnn_act = 'gelu',
        att_act = 'gelu',
        cnn_norm: str="instancenorm",
        cnn_conv2d_has_bias=False,
        cnn_bn_has_bias=True,
        att_scales=(3,5),
        att_conv2d_has_bias=False,
        att_bn_has_bias=True,
        attn_type='MLA',
        use_share_encoder=False,
        cal_gradient = False,
        kernel_fn='elu',
        SCEM_feature_encoder_name = "Backbone",
        sim_conv_kernel_sizes=[3,5],
        dual_softmax_method=None,
        **kwargs
    ) -> None:
        
           
        super().__init__()
        self.d_model = d_model
        self.num_of_predited_pts = num_of_predited_pts
        self.sim_conv_kernel_sizes = sim_conv_kernel_sizes
        self.dual_softmax_method = dual_softmax_method
        self.pos_encoding = PositionEncodingSine(
                             d_model,
                             max_shape,
                             )
        
        if use_share_encoder:
            self.search_module = globals()[SCEM_feature_encoder_name](color_format_search,num_features,d_model,downsampling,num_backbone_layes,cnn_act,cnn_norm,cnn_conv2d_has_bias=cnn_conv2d_has_bias,cnn_bn_has_bias=cnn_bn_has_bias,cal_gradient=cal_gradient)
            self.template_module = self.search_module
        else:
            self.search_module =  globals()[SCEM_feature_encoder_name](color_format_search,num_features,d_model,downsampling,num_backbone_layes,cnn_act,cnn_norm,cnn_conv2d_has_bias=cnn_conv2d_has_bias,cnn_bn_has_bias=cnn_bn_has_bias,cal_gradient=cal_gradient)
            self.template_module =  globals()[SCEM_feature_encoder_name](color_format_search,num_features,d_model,downsampling,num_backbone_layes,cnn_act,cnn_norm,cnn_conv2d_has_bias=cnn_conv2d_has_bias,cnn_bn_has_bias=cnn_bn_has_bias,cal_gradient=cal_gradient)
        
        self.attention_module = Multiscale_Linear_attention(d_model=d_model,
                                                        nhead=n_heads,
                                                        layer_names=layer_names,
                                                        attention_drop_out_rate=attention_drop_out_rate,
                                                        scales=att_scales,
                                                        att_act=att_act,
                                                        att_conv2d_has_bias=att_conv2d_has_bias,
                                                        att_bn_has_bias=att_bn_has_bias,
                                                        attn_type = attn_type,
                                                        kernel_fn = kernel_fn,
                                                        )
   
        in_channels_head = template_size[0]*template_size[1]//(downsampling**2)
        
        # 动态创建多尺度卷积层
        self.template_conv_layers = nn.ModuleDict()
        for idx in range(len(self.sim_conv_kernel_sizes)):
            self.template_conv_layers[f'{idx}'] = Conv2dBNAct(
                d_model, d_model, 3, 1,
                act=cnn_act, norm=cnn_norm,
                conv_has_bias=cnn_conv2d_has_bias,
                bn_has_bias=cnn_bn_has_bias
            )

        out_conv_layers_score = [
            ResidualConv2dBNAct(in_channels_head, SCEM_num_features_predition_head, 1, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias),
            ResidualConv2dBNAct(SCEM_num_features_predition_head , SCEM_num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias),
            ResidualConv2dBNAct(SCEM_num_features_predition_head , SCEM_num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias),
        ]
        self.out_conv = nn.Sequential(*out_conv_layers_score)
        
        self.head_cls = nn.ModuleList([ResidualConv2dBNAct(SCEM_num_features_predition_head , SCEM_num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)]+[Conv2dBNAct(SCEM_num_features_predition_head, num_of_predited_pts, head_kernel_size, 1, has_activation=False, act=cnn_act,norm=None,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)])                  
        self.head_reg = nn.ModuleList([ResidualConv2dBNAct(SCEM_num_features_predition_head , SCEM_num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)]+[Conv2dBNAct(SCEM_num_features_predition_head, num_of_predited_pts*2, head_kernel_size, 1, has_activation=False, act=cnn_act,norm=None,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)])


    def dual_softmax(self, similarity_matrix, tau=1.0, method=None):
        B, M, N = similarity_matrix.shape
    
        if method == 'separate':
            tau_sqrt = tau ** 0.5
            row_softmax = F.softmax(similarity_matrix / tau_sqrt, dim=2)
            col_softmax = F.softmax(similarity_matrix / tau_sqrt, dim=1)
            combined = row_softmax * col_softmax
            normalized_matrix = combined / (combined.sum(dim=(1,2), keepdim=True) + 1e-8)
            
        elif method == 'sequential':
            temp = F.softmax(similarity_matrix / tau, dim=2)
            normalized_matrix = F.softmax(temp, dim=1)  # 修正：不重复除tau
            
        elif method == 'sinkhorn':
            matrix = similarity_matrix / tau
            num_iters = 5
            
            for _ in range(num_iters):
                matrix = matrix - torch.logsumexp(matrix, dim=2, keepdim=True)
                matrix = matrix - torch.logsumexp(matrix, dim=1, keepdim=True)
            matrix = matrix - torch.logsumexp(matrix, dim=2, keepdim=True)  # 额外归一化
            normalized_matrix = torch.exp(matrix)

        elif method is None:  
            normalized_matrix = similarity_matrix
        else:
            raise ValueError("method must be 'separate', 'sequential', 'sinkhorn', or None")
        
        return normalized_matrix


    def forward(self, imgs_search, imgs_template):
        """
        imgs_search (b,1,search_size, search_size)
        imgs_template (b,1,search_size, search_size)
        output (b,2,search_size)
        """

        x_search, x_search_rs = self.search_module(imgs_search)
        x_template, x_template_rs = self.template_module(imgs_template)

        b,c,h_search,w_search = x_search.shape
        _,_,h_template,w_template = x_template.shape
        
        x_search = self.pos_encoding(x_search)
        x_template = self.pos_encoding(x_template)
        
        search_out, template_out = self.attention_module(x_search,x_template)

        # 动态应用多尺度卷积层
        template_out_multiscale = []
        template_out_clone = template_out.clone()
        for idx in range(len(self.sim_conv_kernel_sizes)):
            conv_layer = self.template_conv_layers[f'{idx}']
            template_out_clone = conv_layer(template_out_clone)
            template_out_multiscale.append(template_out_clone)

        search_out = search_out.reshape(b,-1,h_search*w_search).transpose(-1,-2).contiguous()
        template_out = template_out.reshape(b,-1,h_template * w_template ).transpose(-1,-2).contiguous()
        
        # 将多尺度卷积结果重塑
        template_out_multiscale_reshaped = []
        for template_out_conv in template_out_multiscale:
            template_out_multiscale_reshaped.append(
                template_out_conv.reshape(b,-1,h_template * w_template ).transpose(-1,-2).contiguous()
            )

        # 标准化特征
        features_to_normalize = [search_out, template_out] + template_out_multiscale_reshaped
        normalized_features = list(map(lambda feat: feat / feat.shape[-1]**.5, features_to_normalize))
        
        search_out = normalized_features[0]
        template_out = normalized_features[1]
        template_out_multiscale_normalized = normalized_features[2:]

        # 计算相似度矩阵
        sim_matrices = []
        # 添加原始模板特征的相似度矩阵
        sim_matrices.append(torch.einsum("nlc,nsc->nls", search_out, template_out))
        # 添加多尺度卷积特征的相似度矩阵
        for template_out_conv in template_out_multiscale_normalized:
            sim_matrices.append(torch.einsum("nlc,nsc->nls", search_out, template_out_conv))

        sim_matrix_sum = sum(sim_matrices)

        sim_matrix_sum = self.dual_softmax(similarity_matrix=sim_matrix_sum, 
                                    tau=1.0,  # 
                                    method=self.dual_softmax_method
                                    ) 
       
        sim_matrix_out = sim_matrix_sum.clone().permute(0,-1,-2).reshape(b, h_template*w_template, h_search,w_search).contiguous()

        relation_map = self.out_conv(sim_matrix_out)        
  
        score_map = relation_map
        offset_map = relation_map
        for layer in  self.head_cls:
            score_map = layer(score_map )
        for layer in  self.head_reg:
            offset_map = layer(offset_map )

        return sim_matrix_sum,score_map,offset_map, x_search_rs, x_template_rs

class IHERM(nn.Module):
    def __init__(
        self, 
        downsampling: int = 16,
        template_size: tuple = (512, 512),
        search_size: tuple = (800, 800),
        d_model: int = 64,
        cnn_act: str = 'gelu',
        cnn_norm: str = "batchnorm",
        cnn_conv2d_has_bias: bool = False,
        cnn_bn_has_bias: bool = True,
        inner_dis: int = 0,
        scales: list = [4],
        n_warp_each_scale: list = [3],
        use_share_encoder=False,
        **kwargs
    ) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.search_size = search_size[0]
        self.inner_dis = inner_dis
        self.template_size = template_size
        self.downsampling = downsampling
        self.scales = scales
        self.n_scale = len(scales)
        self.n_warp_each_scale = n_warp_each_scale
        self.use_share_encoder = use_share_encoder
        # 搜索分支和模板分支的特征提取模块

        self.search_proj_layers = nn.ModuleList([Bottleneck(in_channels=d_model, 
                                                            expansion_factor=2, 
                                                            has_conv3=True, 
                                                            act=cnn_act, 
                                                            norm=cnn_norm,
                                                            conv_has_bias=cnn_conv2d_has_bias,
                                                            bn_has_bias=cnn_bn_has_bias) for i in range(self.n_scale)]
                                                )
        
        if self.use_share_encoder:
            self.template_proj_layers = self.search_proj_layers
        else:
            self.template_proj_layers = nn.ModuleList([Bottleneck(in_channels=d_model, 
                                                                  expansion_factor=2, 
                                                                  has_conv3=True, 
                                                                  act=cnn_act, 
                                                                  norm=cnn_norm,
                                                                  conv_has_bias=cnn_conv2d_has_bias,
                                                                  bn_has_bias=cnn_bn_has_bias) for i in range(self.n_scale)]
                                                    )
        
        prediction_channels_corr_out = 81 

        self.conv_corr_out = nn.Sequential(
                        ResidualConv2dBNAct(prediction_channels_corr_out, d_model, 3, 1, 
                        act=cnn_act, norm=cnn_norm,
                        conv_has_bias=cnn_conv2d_has_bias, 
                        bn_has_bias=cnn_bn_has_bias)
                        )
    
        self.pts_predictors = nn.ModuleList([
            Decoder_adapative(input_dim=d_model,
                              input_size=(self.template_size[0]//downsampling*(2**i),self.template_size[1]//downsampling*(2**i)),
                              target_size=(2,2)) for i in range(self.n_scale)
        ])

    def get_template_4_corners(self, bs):
        h, w = self.template_size
        inner_dis = self.inner_dis
        corners = torch.tensor([
            [inner_dis, inner_dis],
            [inner_dis, w - inner_dis - 1],
            [h - inner_dis - 1, inner_dis],
            [h - inner_dis - 1, w - inner_dis - 1]
        ], dtype=torch.float32)
        
        return corners.repeat(bs, 1, 1)
    
    def warp(self, coords, image, h, w):
        #对 coords 的第 0 和第 1 维进行操作，将其值标准化到 [-1, 1] 的范围
        coords[: ,0 ,: ,:] = 2.0 * coords[: ,0 ,: ,:].clone() / max(w -1 ,1 ) -1.0
        coords[: ,1 ,: ,:] = 2.0 * coords[: ,1 ,: ,:].clone() / max(h -1 ,1 ) -1.0

        valid_mask = (coords[:, 0, :, :] >= 0) & (coords[:, 0, :, :] < w) & \
                (coords[:, 1, :, :] >= 0) & (coords[:, 1, :, :] < h)
        valid_mask = valid_mask.float() # [B, H, W]

        #将维度调整为[bs,h,w,2]
        coords = coords.permute(0 ,2 ,3 ,1)
        output = F.grid_sample(image, coords, align_corners=True, padding_mode="zeros")

        return output , valid_mask

    def forward(self, x_search, x_template,init_four_disp):
        """
        Args:
            imgs_search (tensor): 搜索图像，形状为 (b, 1, Hs, Ws)
            imgs_template (tensor): 模板图像，形状为 (b, 1, Ht, Wt)
        Returns:
            list of tensor: 每个尺度下的角点偏移量
        """
        
        x_search = [self.search_proj_layers[i](x_search[i]) for i in range(self.n_scale)]
        x_template = [self.template_proj_layers[i](x_template[i]) for i in range(self.n_scale)]

        dev = x_search[0].device
        b = x_search[0].shape[0]
        
        delta = []
        four_point_disp = torch.zeros((b, 4, 2)).to(dev) + init_four_disp
        org_pst = self.get_template_4_corners(bs=b).to(dev)
        
        for scale_idx, scale in enumerate(self.scales):
            h_search, w_search = x_search[scale_idx].shape[-2:]
            for iter in range(self.n_warp_each_scale[scale_idx]):
                pred_pts = org_pst + four_point_disp
                H = kornia.geometry.transform.get_perspective_transform(
                            org_pst[:, :, [1, 0]]/scale, pred_pts[:, :, [1, 0]]/scale,
                        )
                h_template, w_template = x_template[scale_idx].shape[-2:]
                coords1 = disp_to_coords_p2w(h_template, w_template, H, dev)
                x_search_wapped, _ = self.warp(coords1, x_search[scale_idx], h_search, w_search)

               
                corr_out = corr_implement.FunctionCorrelation(x_template[scale_idx].float(),x_search_wapped.float())
                corr_out = self.conv_corr_out(corr_out)

                pts_pred = self.pts_predictors[scale_idx](corr_out).transpose(-1, -2).reshape(-1, 4, 2)
                four_point_disp = four_point_disp + pts_pred
                delta.append(four_point_disp.clone())

        return delta

