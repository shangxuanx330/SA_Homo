import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

class Conv2dBNAct(nn.Module):
    # Standard Conv2d+BN+Act.
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int,
        groups: int = 1,
        has_activation: bool=True,
        act: str="gelu",
        norm: str="batchnorm",
        conv_has_bias: bool = False,
        bn_has_bias: bool = True,
        gourpnorm_size = 4,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups,
            padding=kernel_size // 2,
            bias=conv_has_bias,
        )

        if norm=="instancenorm":
            self.bn = nn.InstanceNorm2d(out_channels, affine=bn_has_bias, eps=1e-4)
        elif norm=="batchnorm":
            self.bn = nn.BatchNorm2d(out_channels, affine=bn_has_bias,eps=1e-4)
        elif norm=='groupnorm':
            self.bn = nn.GroupNorm(num_groups=out_channels // gourpnorm_size, num_channels=out_channels, affine=bn_has_bias)
        else:
            [print(f"Warning: No normalization used in layer with {out_channels} channels") for _ in range(1)]
            self.bn = nn.Identity()

        if act =='relu':
            act = nn.ReLU()
        elif act =='gelu':
            act = nn.GELU()
        elif act =='silu':
            act = nn.SiLU()
        self.act = act if has_activation is True else nn.Identity()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, x: Tensor) -> Tensor: 
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvTranspose2dBNAct(nn.Module):
    # Transposed Conv2d + BN + Act for upsampling
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 2,  # Default stride of 2 for 2x upsampling
        groups: int = 1,
        has_activation: bool = True,
        act: str = "gelu",
        norm: str = "batchnorm",
        conv_has_bias: bool = False,
        bn_has_bias: bool = True,
        groupnorm_size: int = 4,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups,
            padding=kernel_size // 2,
            output_padding=stride - 1,  # Ensure output size is exactly 2x input size
            bias=conv_has_bias,
        )

        if norm == "instancenorm":
            self.bn = nn.InstanceNorm2d(out_channels, affine=bn_has_bias, eps=1e-5)
        elif norm == "batchnorm":
            self.bn = nn.BatchNorm2d(out_channels, affine=bn_has_bias, eps=1e-5)
        elif norm == 'groupnorm':
            self.bn = nn.GroupNorm(num_groups=out_channels // groupnorm_size, num_channels=out_channels, affine=bn_has_bias)
        else:
            self.bn = nn.Identity()

        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        elif act == 'silu':
            act = nn.SiLU()
        self.act = act if has_activation else nn.Identity()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class ResidualConv2dBNAct(nn.Module):
    # Standard Conv2d+BN+Act.
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int,
        groups: int = 1,
        has_activation: bool=True,
        act: str="gelu",
        norm: str="batchnorm",
        conv_has_bias: bool = False,
        bn_has_bias: bool = True,
        gourpnorm_size = 4,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups,
            padding=kernel_size // 2,
            bias=conv_has_bias,
        )

        if in_channels == out_channels and stride==1:
            self._res = nn.Identity()
        else:
            self._res = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=stride,
            groups=1,
            padding=0,
            bias=False,
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
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, x: Tensor) -> Tensor:
        res_x = self._res(x)

        x = self.conv(x)
        x = self.bn(x)
        
        return  self.act(x + res_x)

class Bottleneck(nn.Module):
    # Stride 1 bottleneck.
    def __init__(self, in_channels: int, expansion_factor: float, has_conv3: bool,act='gelu', norm: str="batchnorm",conv_has_bias=False,bn_has_bias=True):
        super().__init__()
        hidden_channels = int(expansion_factor * in_channels)
        self.conv1 = Conv2dBNAct(in_channels, hidden_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv2 = Conv2dBNAct(hidden_channels, hidden_channels, 3, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv3 = Conv2dBNAct(hidden_channels, in_channels, 1, 1, has_activation=False, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias) if has_conv3 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions.
    def __init__(self, in_channels: int, out_channels: int,act='gelu', norm: str="batchnorm",conv_has_bias=False,bn_has_bias=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv2dBNAct(in_channels, hidden_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv2 = Conv2dBNAct(in_channels, hidden_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv3 = Conv2dBNAct(2 * hidden_channels, out_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.bottleneck = Bottleneck(hidden_channels, 1, has_conv3=False, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.conv2(x)
        x = self.bottleneck(self.conv1(x))
        x = torch.cat((x, skip), dim=1)
        x = self.conv3(x)
        return x

def reconstruct_from_sim_matrix(sim_matrix, search_img_size, template_img_size):
    """
    从相似度矩阵重建图像
    
    参数:
    sim_matrix [bs, template_h*template_w, search_h*search_w] - 相似度矩阵
    search_img_size [2] - 搜索图像的高度和宽度 (smaller)
    template_img_size [2] - 模板图像的高度和宽度 (larger)
    
    返回:
    out_img [bs, 1, search_h, search_w] - 重建的二进制图像
    """
    batch_size = sim_matrix.shape[0]
    template_size = template_img_size[0] * template_img_size[1]
    
    # 创建输出图像
    out_img = torch.zeros(size=(batch_size, 1, search_img_size[0], search_img_size[1]),
                          device=sim_matrix.device)
    
    # 对于模板中的每个点，找到搜索图像中最相似的点
    # dim=2是因为我们在search_h*search_w维度上找最大值
    indices = sim_matrix.argmax(dim=1)  # [bs, template_h*template_w]
    
    for i in range(batch_size):
        for j in range(template_size):
            # 获取在搜索图像中的索引
            idx = indices[i, j]
            
            # 计算在搜索图像中的2D坐标
            h = idx // search_img_size[1]
            w = idx % search_img_size[1]
            
            # 在输出图像中标记该位置
            out_img[i, 0, h, w] = 1
    
    return out_img

class get_gt_sim_matrix_v1(nn.Module):
    def __init__(
        self,
        size_optical, 
        size_sar,
        stride: int,
        batch_size: int,
        scale_factor_4_sub_img: int=2,
    ) -> None:   
        super().__init__()
        self.h = size_optical[0]*size_optical[1]//(stride**2)
        self.w = size_sar[0]*size_sar[1]//(stride**2)
        self.batch_size = batch_size
        self.sar_h = size_sar[0]//stride
        self.sar_w = size_sar[1]//stride
        self.optical_w = size_optical[1]//stride
        self.stride = stride
        
        self.register_buffer(
            "gt_sim_matrix",
            torch.zeros(size=(batch_size,self.h,self.w))
        )

        meshgrid = torch.meshgrid(torch.arange(self.sar_h), torch.arange(self.sar_w), indexing='ij')
        grid_x, grid_y = meshgrid[0], meshgrid[1]
        grid_x = grid_x *stride// scale_factor_4_sub_img
        grid_y = grid_y *stride// scale_factor_4_sub_img 
        self.register_buffer(
            "grid_x",
            grid_x
        )        
        
        self.register_buffer(
            "grid_y",
            grid_y
        )  
        
        self.register_buffer(
            "batch_indexes",
            torch.arange(batch_size).view(-1, 1).expand(-1, self.w).reshape(-1)
        ) 
        
        
    def forward(self,tl_croped,tl, H):
        """
        tl_croped [b,x,y] 剪裁出来的部分在进行单应性矩阵变换后的左上角坐标
        tl [b,x,y] sar图在光学图片中的左上角坐标
        H  [b,3,3]
        """
        self.gt_sim_matrix[:,:,:] = 0 
        # Compute grid for indexing    
        grid_x = self.grid_x.repeat(self.batch_size, 1, 1)
        grid_y = self.grid_y.repeat(self.batch_size, 1, 1)

        # Adjust grid by gt offsets
        grid_x = grid_x + tl_croped[:, 0].unsqueeze(1).unsqueeze(2)
        grid_y = grid_y + tl_croped[:, 1].unsqueeze(1).unsqueeze(2)

        # Calculate inverse homography matrix for each batch
        H_inv = torch.inverse(H)

        # Construct homogeneous coordinates for batch processing
        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_y, grid_x, ones], dim=-1).to(torch.float32)
        
        # Apply the inverse homographic transformation
        coords_transformed = torch.matmul(coords.view(self.batch_size, -1, 3), H_inv.transpose(-1, -2))
        coords_transformed = coords_transformed.reshape(self.batch_size, self.sar_h,self.sar_w,3)
        coords_transformed[..., :2] /= coords_transformed[..., 2:3]
        tl = tl.unsqueeze(dim=0).unsqueeze(dim=0)
        
        coords_transformed = coords_transformed[..., :2] + tl

        # Convert coordinates to indices in sim_matrix
        indices_h = coords_transformed[..., 1] // self.stride
        indices_w = coords_transformed[..., 0] // self.stride

        indices = indices_h * self.optical_w + indices_w
        indices = indices.long()
        flattened_indexes = indices.reshape(-1)

        self.gt_sim_matrix[self.batch_indexes, flattened_indexes, torch.arange(self.w).repeat(self.batch_size)] = 1

        return self.gt_sim_matrix
    
class FinePreprocess(nn.Module):
    def __init__(self, window_size,d_model_c, d_model_f, cat_c_feat=True):
        super().__init__()

        self.W = window_size
        self.d_model_f = d_model_f
        self.cat_c_feat = cat_c_feat
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1,selected_points,stride):
        """
        feat_f0,feat_f1 : 由backbone 提取出来比较大的特征  [B,c,h//stride, w//stride]
        feat_c0, feat_c1 : 经过corase matching的输出
        selected_points :  被选择的点 [bs,k,2]
        stride : 放缩倍数 [1]
        """
        W = self.W
    

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2) # B,W*w*C,L
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)# B,L,w*w*C
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)# B,W*w*C,L
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)# B,L,w*w*C

        # 2. select only the predicted matches
        selected_points_up *=stride
        # 3. 将selected_points转换为特征图中的索引
        b_ids_up = torch.arange(selected_points_up.shape[0]).unsqueeze(-1).repeat(1, selected_points_up.shape[1]).to(torch.long)
        i_ids_up = selected_points_up[..., 0].to(torch.long)
        j_ids_up = selected_points_up[..., 1].to(torch.long)

        b_ids = torch.arange(selected_points.shape[0]).unsqueeze(-1).repeat(1, selected_points.shape[1]).to(torch.long)
        i_ids = selected_points[..., 0].to(torch.long)
        j_ids = selected_points[..., 1].to(torch.long)


        feat_f0_unfold = feat_f0_unfold[b_ids_up, i_ids_up]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[b_ids_up, j_ids_up]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[b_ids, i_ids],
                                                   feat_c1[b_ids, j_ids]], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
    
def grouped_matrix_multiplication(A: torch.Tensor, B: torch.Tensor, K: int) -> torch.Tensor:
    """
    执行分组矩阵乘法。

    参数:
    A (torch.Tensor): 形状为 [B, N, dim] 的输入张量
    B (torch.Tensor): 形状为 [B, M, dim] 的输入张量
    K (int): 分组数量

    返回:
    torch.Tensor: 形状为 [B, N, M, K] 的输出张量
    """
    b, n, dmodel = A.shape
    _, m, _ = B.shape
    
    # 确保dim能被K整除
    assert dmodel % K == 0, "dim must be divisible by K"
    
    group_size = dmodel // K
    
    # 重塑A和B以引入分组
    A_grouped = A.view(b, n, K, group_size)
    B_grouped = B.view(b, m, K, group_size)
    
    # 执行分组矩阵乘法
    result = torch.einsum('bnkg,bmkg->bnmk', A_grouped, B_grouped)
    
    return result

class Upsample_ConvBlock(nn.Module):
    """
    改进版上采样模块：Upsample(Bilinear) -> Conv -> Norm -> Act
    避免转置卷积的棋盘格效应
    """
    def __init__(self, in_channels, out_channels, norm='instancenorm', act='gelu'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if norm == 'instancenorm':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
       
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x