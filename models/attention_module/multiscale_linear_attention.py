import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class Conv2dBNAct(nn.Module):
    # Standard Conv2d+BN+Act.
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int,
        dilation=1,
        groups: int = 1,
        has_activation: bool=True,
        act: str="gelu",
        norm: str="batchnorm",
        conv_has_bias: bool = False,
        bn_has_bias: bool = True,
        gourpnorm_size = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=kernel_size // 2,
            bias=conv_has_bias,
        )

        if norm=="instancenorm":
            self.bn = nn.InstanceNorm2d(out_channels, affine=bn_has_bias, eps=1e-5)
        elif norm=="batchnorm":
            self.bn = nn.BatchNorm2d(out_channels, affine=bn_has_bias,eps=1e-5)
        elif norm=='groupnorm':
            self.bn = nn.GroupNorm(num_groups=out_channels // gourpnorm_size, num_channels=out_channels, affine=bn_has_bias)
        else:
            self.bn = nn.Identity()

        if act =='relu':
            act = nn.ReLU()
        elif act =='gelu':
            act = nn.GELU()
        elif act =='silu':
            act = nn.SiLU()
        self.act = act if has_activation is True else nn.Identity()
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ELUPlusOne(nn.Module):
    def __init__(self, alpha=1.0):
        super(ELUPlusOne, self).__init__()
        # 定义 ELU 激活函数，alpha 是 ELU 的超参数，默认为 1.0
        self.elu = nn.ELU(alpha=alpha)
    
    def forward(self, x):
        # 先通过 ELU，然后加 1
        return self.elu(x) + 1
    
# class MultiscaleLinearAttention(nn.Module):
#     def __init__(self, dim, nhead, scales=(5,), eps=1e-2, kernel_fn='relu'):
#         super().__init__()
    
#         self.dim = dim
#         self.heads = nhead
#         self.eps = eps
#         if kernel_fn =='relu':
#             self.kernel_fn = nn.ReLU()
#         elif kernel_fn =='gelu':
#             self.kernel_fn = nn.GELU()
#         elif kernel_fn =='elu':
#             self.kernel_fn = ELUPlusOne()
#         elif kernel_fn =='silu':
#             self.kernel_fn = nn.SiLU()
        
#         self.scales = scales
#         self.aggreg_q = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(
#                         nhead * dim,
#                         nhead * dim,
#                         scale,
#                         padding=scale // 2,
#                         groups=nhead * dim,
#                         bias=False,
#                     ),
#                     nn.Conv2d(nhead * dim, nhead * dim, 1, groups=1, bias=False),
#                 )
#                 for scale in scales
#             ]
#         )

#         self.aggreg_k = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(
#                         nhead * dim,
#                         nhead * dim,
#                         scale,
#                         padding=scale // 2,
#                         groups=nhead * dim,
#                         bias=False,
#                     ),
#                     nn.Conv2d(nhead * dim, nhead * dim, 1, groups=1, bias=False),
#                 )
#                 for scale in scales
#             ]
#         )

#         self.aggreg_v = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(
#                         nhead * dim,
#                         nhead * dim,
#                         scale,
#                         padding=scale // 2,
#                         groups=nhead * dim,
#                         bias=False,
#                     ),
#                     nn.Conv2d(nhead * dim, nhead * dim, 1, groups=1, bias=False),
#                 )
#                 for scale in scales
#             ]
#         )

#         self.proj = nn.Conv2d(
#                         in_channels=nhead * dim * (len(scales)+1),
#                         out_channels=nhead * dim,
#                         kernel_size=1,
#                         padding=0,
#                         groups=1,
#                         bias=False,
#                     )
#     def forward(self, queries, keys, values,  q_mask=None, kv_mask=None):
#         """
#         queries: torch.tensor [bs,dim,h1,w1]
#         keys: torch.tensor [bs,dim,h2,w2]
#         values: torch.tensor [bs,dim,h2,w2]
#         """
#         b,d,h1,w1 = queries.shape
#         _,_,h2,w2 = keys.shape

#         multi_scale_q = [queries]
#         multi_scale_k = [keys]
#         multi_scale_v = [values]
        
#         for op in self.aggreg_q:
#             multi_scale_q.append(op(queries))
            
#         for op in self.aggreg_k:
#             multi_scale_k.append(op(keys))

#         for op in self.aggreg_v:
#             multi_scale_v.append(op(values))

#         q = torch.cat(multi_scale_q, dim=1)
#         k = torch.cat(multi_scale_k, dim=1)
#         v = torch.cat(multi_scale_v, dim=1)
        

#         q = self.kernel_fn(q)
#         k = self.kernel_fn(k)

#         q = q.reshape(b,-1,self.dim,h1*w1)
#         k = k.reshape(b,-1,self.dim,h2*w2)
#         v = v.reshape(b,-1,self.dim,h2*w2)
       
        
#         # 应用mask
#         if q_mask is not None:
#             q_mask = q_mask.view(b, 1, 1, h1*w1)
#             q = q * q_mask

#         if kv_mask is not None:
#             kv_mask = kv_mask.view(b, 1, 1, h2*w2)
#             k = k * kv_mask
#             v = v * kv_mask

#         trans_k = k.transpose(-1, -2) #[bs,nhead*(len(scale)+1),h2*w2,dim]

#         v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)#[bs,nhead*(len(scale)+1),dim,h2*w2+1]

#         v = v/(v.shape[-1])
       
#         vk = torch.matmul(v, trans_k) # 变小了 v.shape
    
#         out = torch.matmul(vk, q) # 变小了 v.shape

#         denom = torch.clamp(out[:, :, -1:].abs(), min=self.eps) * torch.sign(out[:, :, -1:])
#         out = out[:, :, :-1] / denom
      
#         out =  out * (v.shape[-1])

#         out = torch.reshape(out, (b, -1, h1, w1))
      
#         out = self.proj(out)


#         return out

class MultiscaleLinearAttention(nn.Module):
    def __init__(self, dim, nhead, scales=(5,), eps=1e-6, kernel_fn='relu'):
        super().__init__()
    
        self.dim = dim
        self.heads = nhead
        self.eps = eps
        if kernel_fn =='relu':
            self.kernel_fn = nn.ReLU()
        elif kernel_fn =='gelu':
            self.kernel_fn = nn.GELU()
        elif kernel_fn =='elu':
            self.kernel_fn = ELUPlusOne()
        elif kernel_fn =='silu':
            self.kernel_fn = nn.SiLU()
        
        self.scales = scales
        self.aggreg_q = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        nhead * dim,
                        nhead * dim,
                        scale,
                        padding=scale // 2,
                        groups=nhead * dim,
                        bias=False,
                    ),
                    nn.Conv2d(nhead * dim, nhead * dim, 1, groups=1, bias=False),
                )
                for scale in scales
            ]
        )

        self.aggreg_k = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        nhead * dim,
                        nhead * dim,
                        scale,
                        padding=scale // 2,
                        groups=nhead * dim,
                        bias=False,
                    ),
                    nn.Conv2d(nhead * dim, nhead * dim, 1, groups=1, bias=False),
                )
                for scale in scales
            ]
        )

        self.aggreg_v = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        nhead * dim,
                        nhead * dim,
                        scale,
                        padding=scale // 2,
                        groups=nhead * dim,
                        bias=False,
                    ),
                    nn.Conv2d(nhead * dim, nhead * dim, 1, groups=1, bias=False),
                )
                for scale in scales
            ]
        )

        self.proj = nn.Conv2d(
                        in_channels=nhead * dim * (len(scales)+1),
                        out_channels=nhead * dim,
                        kernel_size=1,
                        padding=0,
                        groups=1,
                        bias=False,
                    )
    def forward(self, queries, keys, values,  q_mask=None, kv_mask=None):
        """
        queries: torch.tensor [bs,dim,h1,w1]
        keys: torch.tensor [bs,dim,h2,w2]
        values: torch.tensor [bs,dim,h2,w2]
        """
        b,d,h1,w1 = queries.shape
        _,_,h2,w2 = keys.shape

        multi_scale_q = [queries]
        multi_scale_k = [keys]
        multi_scale_v = [values]
        
        for op in self.aggreg_q:
            multi_scale_q.append(op(queries))
            
        for op in self.aggreg_k:
            multi_scale_k.append(op(keys))

        for op in self.aggreg_v:
            multi_scale_v.append(op(values))

        q = torch.cat(multi_scale_q, dim=1)
        k = torch.cat(multi_scale_k, dim=1)
        v = torch.cat(multi_scale_v, dim=1)
     
        
        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        q = q.reshape(b,-1,h1*w1).permute(0,2,1).reshape(b,h1*w1,-1,self.dim)
        k = k.reshape(b,-1,h2*w2).permute(0,2,1).reshape(b,h2*w2,-1,self.dim)
        v = v.reshape(b,-1,h2*w2).permute(0,2,1).reshape(b,h2*w2,-1,self.dim)
       
        # 应用mask
        if q_mask is not None:
            q = q * q_mask[:, :, None, None]
        if kv_mask is not None:
            k = k * kv_mask[:, :, None, None]
            v = v * kv_mask[:, :, None, None]

        v_length = v.shape[1]
        v = v / (v_length)  # prevent fp16 overflow
        kv = torch.einsum("nshd,nshv->nhdv", k, v)  # (S,D)' @ S,V
        z = 1 / torch.clamp(torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)/ (v_length)), min=self.eps) # prevent fp16 overflow
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", q, kv, z) 
        out = queried_values.permute(0,2,3,1).reshape(b,-1, h1, w1)

        out = self.proj(out)
        

        return out



class EncoderLayer_Multiscale_linear(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention_drop_out_rate=0.0,
                 act='gelu',
                 scales=(5,),
                 kernel_fn='relu',
                 conv_has_bias=True,
                 bn_has_bias=False,
                 norm='batchnorm',
                 ):
        super(EncoderLayer_Multiscale_linear, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_conv = Conv2dBNAct(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            groups=1,
            stride=1,
            norm=norm,
            conv_has_bias=conv_has_bias,
            bn_has_bias=bn_has_bias,
            act=act,
        )

        self.k_conv = Conv2dBNAct(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            groups=1,
            stride=1,
            norm=norm,
            conv_has_bias=conv_has_bias,
            bn_has_bias=bn_has_bias,
            act=act,
        )

        self.v_conv = Conv2dBNAct(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            groups=1,
            stride=1,
            norm=norm,
            conv_has_bias=conv_has_bias,
            bn_has_bias=bn_has_bias,
            act=act,
        )

        self.attention = MultiscaleLinearAttention(self.dim, nhead, scales=scales,kernel_fn=kernel_fn)

        self.merge = Conv2dBNAct(
            in_channels=d_model,
            out_channels= d_model,
            kernel_size=1,
            groups=1,
            stride=1,
            norm=norm,
            conv_has_bias=conv_has_bias,
            bn_has_bias=bn_has_bias,
            act=act,
        )

        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        elif act == 'silu':
            act = nn.SiLU()
        
        # feed-forward network
        self.conv_ffn = nn.Sequential(
                Conv2dBNAct(
                    in_channels=d_model,
                    out_channels= 2*d_model,
                    kernel_size=1,
                    groups=1,
                    stride=1,
                    norm=norm,
                    conv_has_bias=conv_has_bias,
                    bn_has_bias=bn_has_bias,
                    act=act,
                ),

                Conv2dBNAct(
                    in_channels=2*d_model,
                    out_channels= d_model,
                    kernel_size=1,
                    groups=1,
                    stride=1,
                    norm=norm,
                    conv_has_bias=conv_has_bias,
                    bn_has_bias=bn_has_bias,
                    act=act,
                ),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source,  q_mask=None, kv_mask=None):
        """
        x: torch.tensor [bs,dmodel,h1,w1]
        y:torch.tensor [bs,dmodel,h2,w2]
        return [bs,dmodel,h1,w1]
        """
       
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_conv(query)
        key = self.k_conv(key)
        value = self.v_conv(value)
        
        message = self.attention(query, key, value,q_mask,kv_mask) #(bs, ,dmodel, h1, w1)
        message = self.merge(message)

        message = x + message
        message = self.norm1(message.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        
        # feed-forward network
        ff_output = self.conv_ffn(message)

        output = ff_output + message  # Changed to use residual connection
        output = self.norm2(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


        return output

# 运行测试
if __name__ == "__main__":
   print('hello world')