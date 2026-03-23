import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention, LinearAttention_formal
from .multiscale_linear_attention import *

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention_drop_out_rate=0.0,
                 attention='linear',
                 act ='gelu',
                 ):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        if attention == 'linear':
            self.attention = LinearAttention() 
        elif attention == 'linear_formal':
            self.attention = LinearAttention_formal()
        else:
            self.attention = FullAttention()

        self.merge = nn.Linear(d_model, d_model, bias=False)

        if act =='relu':
            act = nn.ReLU()
        elif act =='gelu':
            act = nn.GELU()
        elif act =='silu':
            act = nn.SiLU()
        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            act,
            nn.Dropout(p=attention_drop_out_rate),  # Adding Dropout layer with a dropout probability of 0.1
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=attention_drop_out_rate)
        self.dropout2 = nn.Dropout(p=attention_drop_out_rate)

        self.dropout_q = nn.Dropout(p=attention_drop_out_rate)
        self.dropout_k = nn.Dropout(p=attention_drop_out_rate)
        self.dropout_v = nn.Dropout(p=attention_drop_out_rate)

    def forward(self, x, source, q_mask=None, kv_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            q_mask (torch.Tensor): [N, L] (optional)
            kv_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.dropout_q(self.q_proj(query)).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.dropout_k(self.k_proj(key)).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.dropout_v(self.v_proj(value)).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=q_mask, kv_mask=kv_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.dropout1(message)
        message = self.norm1(message)
        
        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.dropout2(message)
        message = self.norm2(message)
        
        return x + message
    
class LocalFeatureTransformer(nn.Module):

    def __init__(self, 
                 d_model,
                 nhead,
                 layer_names=['self', 'cross'] * 4,
                 attention_drop_out_rate=0.0,
                 attention='linear',
                 act = 'gelu',
                 ):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        
        encoder_layer = EncoderLayer(d_model=d_model, nhead=nhead, attention=attention, attention_drop_out_rate=attention_drop_out_rate, act=act)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1

class Multiscale_Linear_attention(nn.Module):

    def __init__(self, 
                 d_model,
                 nhead,
                 layer_names=['self', 'cross'] * 4,
                 attention_drop_out_rate=0.0,
                 scales=(5,),
                 att_act = 'gelu',
                 att_conv2d_has_bias=True,
                 att_bn_has_bias=False,
                 att_norm = 'instancenorm',
                 attn_type='v0',
                 kernel_fn='relu',
                 ):
        super(Multiscale_Linear_attention, self).__init__()

        self.attn_type = attn_type
        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        
        attn_params = {
        'd_model': d_model,
        'nhead': nhead,
        'attention_drop_out_rate': attention_drop_out_rate,
        'act': att_act,
        'scales': scales,
        'conv_has_bias': att_conv2d_has_bias,
        'bn_has_bias': att_bn_has_bias,
        'norm': att_norm,
        'kernel_fn': kernel_fn
        }

        orgv_params = {
            'd_model':d_model, 
            'nhead':nhead, 
            'attention':'linear', 
            'attention_drop_out_rate':0, 
            'act': att_act
        }

        if attn_type=='MLA':
            encoder_layer_class = EncoderLayer_Multiscale_linear
        elif attn_type=='linearFFN':
            encoder_layer_class =  EncoderLayer_Multiscale_linear_with_linearFFN 
        elif attn_type=='orgv':
            encoder_layer_class =  EncoderLayer
        else:
            raise Exception("attn type incorrect")
        

        if attn_type=='orgv':
            encoder_layer = encoder_layer_class(**orgv_params)
        else:
            encoder_layer = encoder_layer_class(**attn_params)
            
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])#不共享参数
                    
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, feat0, feat1,  q_mask=None, kv_mask=None):
        """
        Args:
            feat0 (torch.Tensor): [B, D, H1, W1]
            feat1 (torch.Tensor): [B, D, H2, W2]
           
        """
        b,c,h,w = feat0.size()
        _,c1,h1,w1 = feat1.size()
        if self.attn_type == 'orgv':
            feat0 = feat0.view(b, c, h*w).transpose(-1, -2)
            feat1 = feat1.view(b, c1, h1*w1).transpose(-1, -2)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0)
                feat1 = layer(feat1, feat1,q_mask,kv_mask) 
            elif name == 'cross':
                feat0 = layer(feat0, feat1,q_mask=None,kv_mask=kv_mask) 
                feat1 = layer(feat1, feat0,q_mask,kv_mask=None)
            else:
                raise KeyError

        if self.attn_type == 'orgv':
            feat0 = feat0.transpose(-1, -2).view(b, c, h,w)
            feat1 = feat1.transpose(-1, -2).view(b, c1, h1,w1)

        return feat0, feat1


