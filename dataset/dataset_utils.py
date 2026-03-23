from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import torch.nn.functional as F
import numpy as np
import math
import torch
from models.homography_transformed import restore_single_point
import torch.nn as nn
import kornia


def is_valid_crop(original_img_size, tl, H, crop_size, min_overlap_ratio=0.75, total_samples=200):
    """
    通过计算重叠率来检查裁剪区域的有效性（使用向量化计算）。

    Args:
        original_img_size: tuple(width, height)
            原始图像的尺寸 (宽度, 高度)。
        tl: tuple(x, y)
            裁剪区域左上角在变换后图像中的坐标 (宽度方向x, 高度方向y)。
        H: ndarray, shape=(3,3)
            将原始图像坐标映射到变换后图像坐标的单应性变换矩阵。
        crop_size: tuple(width, height)
            需要裁剪的区域大小 (宽度, 高度)。
        min_overlap_ratio: float, default=0.75
            最小重叠率阈值，范围 [0, 1]。
        total_samples: int, default=100
            期望的总采样点数量（实际数量可能略有差异）。

    Returns:
        tuple(bool, float):
            (是否有效, 实际重叠率)
    """
    width, height = original_img_size
    crop_width, crop_height = crop_size

    try:
        # 计算逆矩阵，将变换后的图像坐标映射回原始图像坐标
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("警告: 单应性矩阵 H 不可逆。")
        return False, 0.0

    # 计算合适的网格密度，使总点数接近target_samples
    aspect_ratio = crop_width / crop_height
    # 解方程: grid_x * grid_y = total_samples, grid_x / grid_y = aspect_ratio
    grid_y = int(np.sqrt(total_samples / aspect_ratio))
    grid_x = int(total_samples / grid_y)
    
    # 确保至少有1个点
    grid_x = max(1, grid_x)
    grid_y = max(1, grid_y)
    
    actual_total_samples = grid_x * grid_y
    
    # 向量化生成所有采样点
    x_samples = np.linspace(tl[0], tl[0] + crop_width, grid_x)
    y_samples = np.linspace(tl[1], tl[1] + crop_height, grid_y)
    
    # 生成网格坐标
    X, Y = np.meshgrid(x_samples, y_samples)
    
    # 将所有点转换为齐次坐标 (N, 3)
    points_homo = np.stack([X.ravel(), Y.ravel(), np.ones(actual_total_samples)], axis=1)
    
    # 向量化变换所有点: (3, 3) @ (N, 3).T = (3, N)
    transformed_points = H_inv @ points_homo.T  # shape: (3, N)
    
    # 向量化归一化齐次坐标
    z_coords = transformed_points[2, :]  # 第三行，所有z坐标
    
    # 找出有效的点（z坐标不接近0）
    valid_mask = np.abs(z_coords) > 1e-8
    
    if not np.any(valid_mask):
        return False, 0.0
    
    # 归一化有效点的坐标
    orig_x = transformed_points[0, valid_mask] / z_coords[valid_mask]
    orig_y = transformed_points[1, valid_mask] / z_coords[valid_mask]
    
    # 向量化边界检查
    in_bounds_mask = (orig_x >= 0) & (orig_x < width) & (orig_y >= 0) & (orig_y < height)
    
    # 计算在边界内的点数
    valid_points = np.sum(in_bounds_mask)
    
    # 计算重叠率（基于实际处理的点数）
    overlap_ratio = valid_points / actual_total_samples
    is_valid = overlap_ratio >= min_overlap_ratio
    
    return is_valid, overlap_ratio

def crop_valid_region(transformed_img, H, original_img_size, crop_size, min_overlap_ratio=0.4):
    """
    从变换后的图像中裁剪出有效区域。

    Args:
        transformed_img: ndarray
            变换后的图像。
        H: ndarray, shape=(3,3)
            单应性变换矩阵（原始 -> 变换后）。
        original_img_size: tuple(width, height)
            原始图像的尺寸 (宽度, 高度)。
        crop_size: tuple(width, height)
            需要裁剪的区域大小 (宽度, 高度)。
        in_pts_num: int, optional
            要求至少有多少个裁剪区域的角点（变换回原始图像后）必须在原始图像边界内。
            默认为 4（所有角点都必须在内）。取值范围 1 到 4。

    Returns:
        cropped_img: ndarray or None
            裁剪后的图像区域。如果找不到有效区域则返回 None。
            shape = (crop_h, crop_w, c)
        tl_point: tuple(x, y) or None
            裁剪区域左上角在变换后图像中的坐标 (宽度方向x, 高度方向y)。
            如果找不到有效区域则返回 None。

    Notes:
        - 函数通过随机采样方式寻找有效的裁剪区域。
        - 最多尝试 1000 次随机采样，每 100 次打印一次警告。
        - 通过 is_valid_crop 函数验证采样点是否有效。
    """
    # 获取变换后图像的高度和宽度
    transformed_h, transformed_w = transformed_img.shape[0:2]

    # 检查 crop_size 是否小于等于 transformed_img 尺寸
    if crop_size[0] > transformed_w or crop_size[1] > transformed_h:
        print(f"警告: 裁剪尺寸 {crop_size} 大于变换后图像尺寸 {(transformed_w, transformed_h)}。")
        return None, None

    # 计算有效的采样范围 (左上角点的最大坐标)
    # +1 是因为 randint 的上界是不包含的
    valid_w_range = transformed_w - crop_size[0]
    valid_h_range = transformed_h - crop_size[1]

    # 确保采样范围有效 (非负)
    if valid_w_range < 0 or valid_h_range < 0:
         print(f"错误: 变换后图像尺寸 {(transformed_w, transformed_h)} 小于裁剪尺寸 {crop_size}。")
         return None, None

    # 随机采样尝试找到有效区域
    idx = 0
    max_attempts = 1000
    for i in range(max_attempts):
        idx += 1
        # 随机选择左上角坐标点 (x, y) -> (宽度, 高度)
        # 注意 randint 的范围是 [low, high)
        selected_tl_w = np.random.randint(0, valid_w_range + 1) if valid_w_range >= 0 else 0
        selected_tl_h = np.random.randint(0, valid_h_range + 1) if valid_h_range >= 0 else 0
        # 验证该点是否可以作为有效的裁剪区域
        # 注意传递 in_pts_num 参数
        is_valid, overlap_ratio = is_valid_crop(original_img_size, (selected_tl_w, selected_tl_h), H, crop_size, min_overlap_ratio)
        if is_valid:
            # 返回裁剪后的图像和左上角坐标 (x, y)
            # 图像索引是 [高度, 宽度]
            return transformed_img[selected_tl_h : selected_tl_h + crop_size[1],
                                 selected_tl_w : selected_tl_w + crop_size[0]], (selected_tl_w, selected_tl_h)

        # 每 100 次采样打印一次警告
        if idx % 100 == 0 and idx > 0:
            print('selected_tl_w, selected_tl_h ', selected_tl_h, selected_tl_w)
            print(f'cropsize:',crop_size)
            print(f'original_img_size:',original_img_size)
            print(f'警告: 随机采样寻找有效裁剪区域已尝试 {idx} 次...')

    # 如果没找到有效区域则返回 None
    print(f"警告: 在 {max_attempts} 次尝试后未能找到满足条件的有效裁剪区域。")
    return None, None

def get_4_pts(tl_gt,size_sar,H_matrix,inner_dis=0,scale_factor=1):
    """
    tl_gt [bs,2]
    size_sar [2] 

    return [bs,4,2]
    """
    inner_dis //= scale_factor
    sar_h, sar_w = size_sar[0] // scale_factor , size_sar[1] // scale_factor
    tr_gt = torch.stack([tl_gt[:,0]+inner_dis, tl_gt[:,1] + sar_w-1-inner_dis], dim = 1) #[bs,2]
    bl_gt = torch.stack([tl_gt[:,0]+sar_h-1-inner_dis, tl_gt[:,1] + inner_dis], dim = 1) #[bs,2]
    br_gt = torch.stack([tl_gt[:,0]+sar_h-1-inner_dis, tl_gt[:,1] + sar_w-1-inner_dis], dim = 1) #[bs,2]
    tl_gt = torch.stack([tl_gt[:,0]+inner_dis, tl_gt[:,1] + inner_dis], dim = 1)
    temp_pts = torch.stack([tl_gt,tr_gt,bl_gt,br_gt],dim=1) #[bs,4,2]

    tl_gt = restore_single_point(tl_gt,H_matrix)
    bl_gt = restore_single_point(bl_gt,H_matrix)
    tr_gt = restore_single_point(tr_gt,H_matrix)
    br_gt = restore_single_point(br_gt,H_matrix)
    
    pts = torch.stack([tl_gt,tr_gt,bl_gt,br_gt],dim=1) #[bs,4,2]

    return pts, temp_pts

def random_flip_and_adjust_H_matrix(img1, img2, H_matrix, p_horizontal=0.5, p_vertical=0.5):
    """
    对img1和img2进行同步的水平翻转和垂直翻转，并更新对应的单应性矩阵H
    
    参数:
    img1: tensor, 形状为 [C, H1, W1]
    img2: tensor, 形状为 [C, H2, W2]
    H_matrix: tensor, 形状为 [3, 3], 从img1到img2的映射
    p_horizontal: float, 水平翻转的概率，默认为0.5
    p_vertical: float, 垂直翻转的概率，默认为0.5
    
    返回:
    augmented_img1: tensor, 增强后的img1
    augmented_img2: tensor, 增强后的img2
    augmented_H: tensor, 更新后的单应性矩阵
    """
    
    # 获取图像尺寸
    _, H1, W1 = img1.shape
    _, H2, W2 = img2.shape
    
    # 根据指定概率决定是否进行水平翻转和垂直翻转
    flip_horizontal = torch.rand(1).item() < p_horizontal
    flip_vertical = torch.rand(1).item() < p_vertical
    
    # 如果不需要任何翻转，直接返回原始数据
    if not flip_horizontal and not flip_vertical:
        return img1, img2, H_matrix
    
    # 优化：预先定义角点坐标，减少重复计算
    device = H_matrix.device
    dtype = H_matrix.dtype
    corners1 = torch.tensor([[0, 0], [W1-1, 0], [0, H1-1], [W1-1, H1-1]], 
                            dtype=dtype, device=device)
    
    # 使用H_matrix计算img2中对应的点
    corners2 = kornia.geometry.transform_points(H_matrix.unsqueeze(0), corners1.unsqueeze(0)).squeeze(0)
    
    # 对图像进行翻转
    if flip_horizontal:
        img1 = torch.flip(img1, [2])  # 水平翻转
        img2 = torch.flip(img2, [2])  # 水平翻转
        corners1[:, 0] = W1 - 1 - corners1[:, 0]
        corners2[:, 0] = W2 - 1 - corners2[:, 0]
    
    if flip_vertical:
        img1 = torch.flip(img1, [1])  # 垂直翻转
        img2 = torch.flip(img2, [1])  # 垂直翻转
        corners1[:, 1] = H1 - 1 - corners1[:, 1]
        corners2[:, 1] = H2 - 1 - corners2[:, 1]
    
    # 使用Kornia的get_perspective_transform计算新的H矩阵
    augmented_H = kornia.geometry.transform.get_perspective_transform(corners1.unsqueeze(0), corners2.unsqueeze(0)).squeeze(0)
    
    return img1, img2, augmented_H

    """
    对img1和img2进行完全独立的水平和垂直翻转，并通过矩阵变换更新单应性矩阵H

    参数:
    img1: tensor, 形状为 [C, H1, W1]
    img2: tensor, 形状为 [C, H2, W2]
    H_matrix: tensor, 形状为 [3, 3], 从img1到img2的映射
    p_horizontal: float, 水平翻转的概率（各自独立），默认为0.5
    p_vertical: float, 垂直翻转的概率（各自独立），默认为0.5

    返回:
    augmented_img1: tensor, 增强后的img1
    augmented_img2: tensor, 增强后的img2
    augmented_H: tensor, 更新后的单应性矩阵
    """
    # 获取图像尺寸
    _, H1, W1 = img1.shape
    _, H2, W2 = img2.shape

    # 独立决定四个翻转状态
    flip_img1_h = torch.rand(1).item() < p_horizontal
    flip_img1_v = torch.rand(1).item() < p_vertical
    flip_img2_h = torch.rand(1).item() < p_horizontal
    flip_img2_v = torch.rand(1).item() < p_vertical

    # 构造水平翻转矩阵（img1和img2独立）
    F1_h = torch.eye(3, dtype=H_matrix.dtype, device=H_matrix.device)
    if flip_img1_h:
        F1_h[0, 0] = -1
        F1_h[0, 2] = W1 - 1.0

    F2_h = torch.eye(3, dtype=H_matrix.dtype, device=H_matrix.device)
    if flip_img2_h:
        F2_h[0, 0] = -1
        F2_h[0, 2] = W2 - 1.0

    # 构造垂直翻转矩阵（img1和img2独立）
    F1_v = torch.eye(3, dtype=H_matrix.dtype, device=H_matrix.device)
    if flip_img1_v:
        F1_v[1, 1] = -1
        F1_v[1, 2] = H1 - 1.0

    F2_v = torch.eye(3, dtype=H_matrix.dtype, device=H_matrix.device)
    if flip_img2_v:
        F2_v[1, 1] = -1
        F2_v[1, 2] = H2 - 1.0

    # 计算总变换矩阵（先水平后垂直）- 修正矩阵乘法顺序
    F1_total = F1_h @ F1_v  # img1的复合变换：先水平再垂直（矩阵顺序修正）
    F2_total = F2_h @ F2_v  # img2的复合变换：先水平再垂直（矩阵顺序修正）

    # 调整H矩阵：H' = F2_total * H_original * inv(F1_total)
    F1_total_inv = torch.linalg.inv(F1_total)
    augmented_H = F2_total @ H_matrix @ F1_total_inv

    # 应用图像翻转（先水平后垂直）
    if flip_img1_h:
        img1 = torch.flip(img1, dims=[2])  # 水平翻转
    if flip_img1_v:
        img1 = torch.flip(img1, dims=[1])  # 垂直翻转

    if flip_img2_h:
        img2 = torch.flip(img2, dims=[2])  # 水平翻转
    if flip_img2_v:
        img2 = torch.flip(img2, dims=[1])  # 垂直翻转

    return img1, img2, augmented_H

def calculate_quadrilateral_area(points):
    """
    计算四边形面积，自动处理负坐标的情况
    
    Parameters:
    points: Array of four corner points' coordinates, shape [4, 2]
           [0] - Top-left coordinate (x, y)
           [1] - Top-right coordinate (x, y) 
           [2] - Bottom-left coordinate (x, y)
           [3] - Bottom-right coordinate (x, y)
    
    Returns:
    float: Area of the quadrilateral
    """
    # 找到最小坐标值，用于平移
    min_x = points[:, 0].min()
    min_y = points[:, 1].min()
    
    # 平移所有点，使最小坐标变为0
    translated_points = points.clone()
    translated_points[:, 0] = points[:, 0] - min_x
    translated_points[:, 1] = points[:, 1] - min_y
    
    # 使用平移后的坐标计算面积
    ordered_points = [translated_points[0], translated_points[1], translated_points[3], translated_points[2]]
    
    # 使用鞋带公式计算面积
    area = 0
    n = len(ordered_points)
    
    for i in range(n):
        j = (i + 1) % n
        area += ordered_points[i][0] * ordered_points[j][1]
        area -= ordered_points[j][0] * ordered_points[i][1]
    
    return abs(area) / 2

def transform_points_with_homography_hw(points, H):
    """
    将(h, w)格式的点通过一个基于(w, h)坐标系的单应性矩阵H进行变换。
    
    参数:
        points: 形状为(B,N,2)的张量，表示B批次，每批次N个2D点。
                每个点的坐标顺序假定为 (h, w) 即 (y, x)。
        H: 形状为(B,3,3)的张量，表示B个单应性矩阵。
           !!! 重要: 该矩阵 H 必须是基于 (w, h) 或 (x, y) 坐标系计算得出的 !!!
           即，它期望输入的齐次坐标是 [w, h, 1]^T。
        
    返回:
        变换后的点，形状为(B,N,2)，坐标顺序恢复为 (h, w)。
    """
    B, N, _ = points.shape

    points = points[:,:,[1,0]]
    # 将点转换为齐次坐标 (B,N,3)
    ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
    points_homogeneous = torch.cat([points, ones], dim=-1)
    
    # 转置点以便于矩阵乘法 (B,N,3) -> (B,3,N)
    points_homogeneous = points_homogeneous.transpose(1, 2)
    
    # 应用单应性变换 (B,3,3) @ (B,3,N) -> (B,3,N)
    transformed_points = torch.bmm(H, points_homogeneous)
    
    # 转置回来 (B,3,N) -> (B,N,3)
    transformed_points = transformed_points.transpose(1, 2)
    
    # 转换回非齐次坐标 (除以第三个坐标)
    # 避免除以零
    denominator = transformed_points[:, :, 2:3]
    transformed_points_2d = transformed_points[:, :, :2] / (denominator+1e-10)
    
    return transformed_points_2d[:,:,[1,0]]

