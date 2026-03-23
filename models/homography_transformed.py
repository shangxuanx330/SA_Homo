"""
该文件包含 单应性变换的各种转换函数
"""
import torch
import torch.nn.functional as F
import kornia

def restore_single_point(coords, H):
    """
    将多个点反映射到原来的位置
    Args:
        coords: Tensor [bs, 2], [:,0]代表纵坐标 [:,1]代表横坐标
        H: Tensor [bs, 3, 3]，每一个矩阵对应于一个坐标点的反映射矩阵

    Returns:
        Tensor: 反映射后的坐标点 [bs, 2]  [:,0]代表纵坐标 [:,1]代表横坐标
    """
    bs = coords.shape[0]
    # 添加一个维度以适应齐次坐标（最后一列为1）
    ones = torch.ones(bs, 1, dtype=coords.dtype, device=coords.device)
    # homog_coords = torch.cat([coords, ones], dim=1).to(H.device)
    homog_coords = torch.cat([coords[:,1].unsqueeze(1),coords[:,0].unsqueeze(1), ones], dim=1).to(H.device)  # [bs, 3]

    # 计算逆矩阵
    H_inv = torch.inverse(H)  # [bs, 3, 3]

    # 应用变换矩阵
    mapped_coords = torch.bmm(H_inv, homog_coords.unsqueeze(-1))  # [bs, 3, 1]


    # 将齐次坐标转换回二维坐标
    mapped_coords = mapped_coords.squeeze(-1)  # [bs, 3]
    final_coords = mapped_coords[:, :2] / mapped_coords[:, 2].unsqueeze(1)  # [bs, 2]

    return final_coords[:,[1,0]] #变换回(h,w)的形式

def restore_multiple_points(coords, H):
    """
    批量反映射多个点到原来的位置（高度并行化版本）
    Args:
        coords: Tensor [bs, k, 2], [:,:,0]代表纵坐标 [:,:,1]代表横坐标
        H: Tensor [bs, 3, 3]，变换矩阵

    Returns:
        Tensor: 反映射后的坐标点 [bs, k, 2]  [:,:,0]代表纵坐标 [:,:,1]代表横坐标
    """
    bs, k, _ = coords.shape
    
    # 转换坐标格式：(y,x) -> (x,y) 并添加齐次坐标
    ones = torch.ones(bs, k, 1, dtype=coords.dtype, device=coords.device)
    # 重新排列为 (x, y, 1) 格式进行齐次坐标变换
    homog_coords = torch.cat([coords[:,:,1:2], coords[:,:,0:1], ones], dim=2)  # [bs, k, 3]
    
    # 计算逆矩阵（一次性计算所有batch的逆矩阵）
    H_inv = torch.inverse(H)  # [bs, 3, 3]
    
    # 重塑坐标以进行高效的批量矩阵乘法
    # 将 [bs, k, 3] -> [bs, 3, k] 以便与 [bs, 3, 3] 进行bmm
    homog_coords_transposed = homog_coords.transpose(1, 2)  # [bs, 3, k]
    
    # 一次性批量矩阵乘法，处理所有点
    mapped_coords = torch.bmm(H_inv, homog_coords_transposed)  # [bs, 3, k]
    mapped_coords = mapped_coords.transpose(1, 2)  # [bs, k, 3]
    
    # 齐次坐标归一化
    final_coords = mapped_coords[:, :, :2] / mapped_coords[:, :, 2:3]  # [bs, k, 2]
    
    # 转换回 (y, x) 格式
    return final_coords[:, :, [1, 0]]
   
def restore_img(image, H, device):
    """
    将一张经过单应性变换过的图像复原， H 为变换所用的矩阵
    Args:
        image Tensor [bs,1,h,w]
        H Tensor [bs,3,3]
    Returns:
        Tensor [bs,1,h,w] 反单应性变换后的图像
    """

    b, c, h, w = image.shape
    # Create normalized meshgrid
    meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid_x, grid_y = meshgrid[0].to(device, non_blocking=True), meshgrid[1].to(device, non_blocking=True)
    ones = torch.ones_like(grid_x).to(device, non_blocking=True)
    grid = torch.stack([grid_y,grid_x, ones], dim=-1).unsqueeze(dim=0)
    
    grid = grid.repeat(b, 1, 1, 1).to(torch.float32)
    
    # Transform the grid
    transformed_grid = torch.matmul(grid.view(b, -1, 3), H.transpose(-1, -2))
    divisor = transformed_grid[:, :, 2].unsqueeze(dim=-1)
    transformed_grid = transformed_grid[:, :, :2] / divisor
    transformed_grid = transformed_grid.view(b, h, w, 2)
    # transformed_grid = torch.floor(transformed_grid[...,[1, 0 ]])
 
    # normalize and reshape
    transformed_grid[..., 0] = (transformed_grid[..., 0] / (w - 1)) * 2 - 1
    transformed_grid[..., 1] = (transformed_grid[..., 1] / (h - 1)) * 2 - 1 
      
    # Sample grid
    transformed_image = F.grid_sample(image, transformed_grid,align_corners=False,mode='nearest')
    
    return transformed_image
    
def get_gt_mask(template_img_size, search_img_size, gt_tl,H_matrix, scale_factor_4_sub_img, stride,device):
    """
    Args:
        template_img_size :(h,w) 模板图的大小
        search_img_size :(H,W) 原图的大小
        gt_tl Tensor :[b,2] 记录着若干组左上角真值
        H_matrix Tensor :[b,3,3] 单应性变换矩阵
        scale_factor_4_sub_img :[1] 对sub img 的缩放倍数
        stride :[1] 模型对输入图像的放缩倍数
        device   
    Returns:
        Tensor [b,1,H,W] 一张真值掩码
    """
    bs = gt_tl.shape[0]
    h_o, w_o = search_img_size
    h_sub, w_sub = template_img_size

    h_sub = h_sub //scale_factor_4_sub_img
    w_sub = w_sub //scale_factor_4_sub_img

    mask = torch.zeros(size=(bs,1,h_o*3,w_o*3)).to(device, non_blocking=True)
    ones = torch.ones(size=(bs,1,h_sub,w_sub)).to(device, non_blocking=True)

    for i in range(bs):
        mask[i,0,int(gt_tl[i,0]):int(gt_tl[i,0])+h_sub, int(gt_tl[i,1]):int(gt_tl[i,1])+w_sub] = ones[i,:,:,:]
        
    mask = restore_img(mask,H_matrix,device)
    #只返回与larger图片同样大小的gt_mask
    mask = mask[:,:,:h_o,:w_o]
    mask = F.interpolate(mask, scale_factor = 1/stride , mode='nearest')

    return mask
    
def disp_to_coords(four_point, coords, downsample=4):
    """
    计算图像四个角的位移，生成透视变换矩阵，并对图像的坐标网格进行透视变换，从而生成新的坐标网格,与原版不同
    """
    four_point = four_point / downsample  # 形状保持不变 (N, 2, 2, 2)
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)  # 形状 (2, 2, 2)

    four_point_org[:, 0, 0] = torch.Tensor([0, 0])  # 左上角
    four_point_org[:, 0, 1] = torch.Tensor([coords.shape[3] - 1, 0])  # 右上角
    four_point_org[:, 1, 0] = torch.Tensor([0, coords.shape[2] - 1])  # 左下角
    four_point_org[:, 1, 1] = torch.Tensor([coords.shape[3] - 1, coords.shape[2] - 1])  # 右下角

    # 增加一个批次维度并重复以匹配批量大小
    four_point_org = four_point_org.unsqueeze(0)  # 形状变为 (1, 2, 2, 2)
    four_point_org = four_point_org.repeat(coords.shape[0], 1, 1, 1)  # 形状变为 (N, 2, 2, 2)
    # 计算新的四个点坐标
    four_point_new = four_point_org + four_point
    # 将四个点的维度展开并调整顺序以计算透视变换矩阵
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()  # 形状变为 (N, 4, 2)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous() # 形状变为 (N, 4, 2)
    # 计算透视变换矩阵
    H = kornia.geometry.transform.get_perspective_transform(four_point_new,four_point_org)  # 形状 (N, 3, 3)
    # 生成网格坐标
    gridy, gridx = torch.meshgrid(torch.linspace(0, coords.shape[3] - 1, steps=coords.shape[3]), torch.linspace(0, coords.shape[2] - 1, steps=coords.shape[2]))  # gridx和gridy的形状为 (H, W)
    # 为了并行化计算单应性映射
    points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, coords.shape[3] * coords.shape[2]))),
                       dim=0).unsqueeze(0).repeat(coords.shape[0], 1, 1).to(four_point.device)  # points形状为 (N, 3, H*W)
   
    points_new = H.bmm(points)
   
    denominator = points_new[:, 2, :].unsqueeze(1)
    epsilon = 1e-10
    denominator = torch.clamp(denominator, min=epsilon)
    
    points_new = points_new / denominator  # 形状 (N, 3, H*W)
    
    points_new = points_new[:, 0:2, :]  # 形状 (N, 2, H*W)
    
    # 生成新的坐标网格
    coords = torch.cat((points_new[:, 0, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1),
                       points_new[:, 1, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1)), dim=1)
    
    return coords   

def disp_to_coords_p2w(H_out, W_out, H, device=None):
    if device is None:
        device = H.device
        
    N = H.shape[0]  # 从变换矩阵H获取batch size
    
    # 生成完整范围的坐标网格
    gridy, gridx = torch.meshgrid(
        torch.linspace(0, H_out - 1, steps=H_out),
        torch.linspace(0, W_out - 1, steps=W_out)
    )
    gridx = gridx.to(device)
    gridy = gridy.to(device)
    
    # 构建齐次坐标
    points = torch.cat((
        gridx.flatten().unsqueeze(0),
        gridy.flatten().unsqueeze(0),
        torch.ones((1, H_out * W_out), device=device)
    ), dim=0).unsqueeze(0).repeat(N, 1, 1)
    
    # 应用逆变换（从大图坐标映射回小图坐标）
    points_new = H.bmm(points)
    
    # 处理透视除法，避免除零
    denominator = torch.clamp(points_new[:, 2, :], min=1e-10).unsqueeze(1)
    points_new = points_new / denominator
    
    # 只保留x,y坐标并重塑为所需形状
    points_new = points_new[:, :2, :]
    coords_new = points_new.view(N, 2, H_out, W_out)
    
    return coords_new

def warp(coords, image, h, w):
        #对 coords 的第 0 和第 1 维进行操作，将其值标准化到 [-1, 1] 的范围
        coords[: ,0 ,: ,:] = 2.0 * coords[: ,0 ,: ,:].clone() / max(w -1 ,1 ) -1.0
        coords[: ,1 ,: ,:] = 2.0 * coords[: ,1 ,: ,:].clone() / max(h -1 ,1 ) -1.0
        #将维度调整为[bs,h,w,2]
        coords = coords.permute(0 ,2 ,3 ,1)
        output = F.grid_sample(image, coords, align_corners=True, padding_mode="border")
        return output
            
def coords_grid(batch, ht, wd):
    """
    
    """
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    #使第零维度相反
    coords = torch.stack(coords[::-1], dim=0).float() # coords[0][x][y]代表纵坐标
    return coords[None].expand(batch, -1, -1, -1)# coords[None] = coords.unsqueeze(dim=0)
      
def initialize_flow(img, downsample=4):
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H//downsample, W//downsample).to(img.device)
    coords1 = coords_grid(N, H//downsample, W//downsample).to(img.device)

    return coords0, coords1
