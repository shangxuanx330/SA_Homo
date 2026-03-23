import torch
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import kornia

def draw_lines(image_batch, points_batch, line_color="white", line_width=8):
    """
    在批量的Torch Tensor图像上根据每张图对应的四个角点绘制指定的线条，支持灰度图和RGB图像。

    参数:
    - image_batch (Tensor): 需要绘制线条的图像批次，形式为 (B, C, H, W)，其中C可以是1（灰度图）或3（RGB图像）
    - points_batch (Tensor): 每张图对应的四个角点坐标，形式为 (B, 4, 2),[:,:,0]为纵坐标，[:,:,1]为横坐标
    - line_color (str): 绘制线条的颜色，默认为白色
    - line_width (int): 线条的宽度，默认为8

    返回:
    - Tensor: 绘制了线条的图像批次Tensor
    """
    # 初始化一个列表来保存处理后的图像
    processed_images = []

    # 遍历批次中的每个图像和对应的坐标
    for image_tensor, points in zip(image_batch, points_batch):
        # 检查通道数
        if image_tensor.size(0) == 1:
            # 如果是灰度图，将其转换为RGB图像
            image_tensor = image_tensor.expand(3, -1, -1)
        
        # 将Tensor转换为PIL Image
        pil_image = TF.to_pil_image(image_tensor)
        
        # 创建一个draw对象
        draw = ImageDraw.Draw(pil_image)

        # 对四个点按照纵坐标排序（升序）
        sorted_points = sorted(points, key=lambda p: p[0])
        # 根据纵坐标分为上、下两组
        top_points = sorted(sorted_points[:2], key=lambda p: p[1])  # 横坐标排序区分左右
        bottom_points = sorted(sorted_points[2:], key=lambda p: p[1])  # 横坐标排序区分左右

        # 左上、右上、左下、右下
        top_left = (int(top_points[0][1]), int(top_points[0][0]))
        top_right = (int(top_points[1][1]), int(top_points[1][0]))
        bottom_left = (int(bottom_points[0][1]), int(bottom_points[0][0]))
        bottom_right = (int(bottom_points[1][1]), int(bottom_points[1][0]))

        # 绘制指定的线条
        draw.line([top_left, top_right], fill=line_color, width=line_width)    # 左上角到右上角
        draw.line([top_left, bottom_left], fill=line_color, width=line_width)  # 左上角到左下角
        draw.line([bottom_left, bottom_right], fill=line_color, width=line_width)  # 左下角到右下角
        draw.line([top_right, bottom_right], fill=line_color, width=line_width)  # 右上角到右下角

        # 将PIL Image转换回Tensor
        processed_image = TF.to_tensor(pil_image)
        # 添加到结果列表中
        processed_images.append(processed_image)

    # 将列表转换回Tensor (B, C, H, W)
    return torch.stack(processed_images).to(torch.float32)

def highlight_pts(image, pts, color, radius=5):
    """
    pts 先h后w
    """
    # 初始化一个列表来保存处理后的图像
    processed_images = []

    # 遍历每一张图像和对应的坐标
    for image_tensor, points in zip(image, pts):
        # 检查通道数
        if image_tensor.size(0) == 1:
            # 如果是灰度图，转换为RGB图像
            image_tensor = image_tensor.expand(3, -1, -1)

        # 将Tensor转换为PIL Image
        pil_image = TF.to_pil_image(image_tensor)

        # 创建一个draw对象
        draw = ImageDraw.Draw(pil_image)

        # 遍历每个点，在图像上绘制圆圈
        for point in points:
            y,x = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        # 将处理后的PIL图像转换回Tensor
        processed_image = TF.to_tensor(pil_image)

        # 添加到结果列表中
        processed_images.append(processed_image)

    # 将列表转换回Tensor (B, C, H, W)
    return torch.stack(processed_images).to(torch.float32)

def create_checker_mixed_image(imgs_search, imgs_template, H_pred_by_usanc, dev, checker_size=16):
    """
    创建棋盘状混合图像
    
    Args:
        imgs_search: 搜索图像 [B, C, H, W]
        imgs_template: 模板图像 [B, C, H_t, W_t]  
        H_pred_by_usanc: 预测的单应性矩阵 [B, 3, 3]
        dev: 设备
        checker_size: 棋盘方格大小，默认16像素
        
    Returns:
        mixed_image: 棋盘状混合图像 [B, C, H, W]
    """
    # 将模板图像变换到搜索图像坐标系
    template_2_search = kornia.geometry.transform.warp_perspective(
        imgs_template, torch.inverse(H_pred_by_usanc), 
        imgs_search.shape[-2:], mode='bicubic',padding_mode='zeros',
        align_corners=True
    )
    
    # 创建棋盘状mask
    h, w = imgs_search.shape[-2:]
    checker_mask = torch.zeros((h, w), device=dev)
    
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            # 创建棋盘模式：(i+j)为偶数时显示search img，奇数时显示warped template
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                checker_mask[i:min(i+checker_size, h), j:min(j+checker_size, w)] = 1
    
    # 扩展mask到所有batch和通道
    checker_mask = checker_mask.unsqueeze(0).unsqueeze(0).expand(imgs_search.shape[0], imgs_search.shape[1], -1, -1)
    
    # 创建混合图像：在重叠区域应用棋盘模式
    template_valid_mask = (template_2_search.abs().sum(dim=1, keepdim=True) > 0.01).float()  # 检测有效的template区域
    
    # 混合图像：在有效区域使用棋盘模式，其他地方显示原始search img
    mixed_image = imgs_search.clone()
    mixed_image = mixed_image * (1 - template_valid_mask * (1 - checker_mask)) + \
                 template_2_search * template_valid_mask * (1 - checker_mask)
    
    return mixed_image

def create_template_replaced_image(imgs_search, imgs_template, H_pred_by_usanc, dev):
    """
    创建template区域被直接替换的图像
    
    Args:
        imgs_search: 搜索图像 [B, C, H, W]
        imgs_template: 模板图像 [B, C, H_t, W_t]  
        H_pred_by_usanc: 预测的单应性矩阵 [B, 3, 3]
        dev: 设备
        
    Returns:
        replaced_image: template区域被替换的图像 [B, C, H, W]
    """
    # 将模板图像变换到搜索图像坐标系
    template_2_search = kornia.geometry.transform.warp_perspective(
        imgs_template, torch.inverse(H_pred_by_usanc), 
        imgs_search.shape[-2:], mode='bicubic',padding_mode='border',align_corners=True
    )
    
    # 创建template的有效区域mask
    # 先在原始template上创建一个全1的mask，然后变换这个mask
    template_mask = torch.ones_like(imgs_template[:, :1, :, :])  # 只需要一个通道
    template_mask_transformed = kornia.geometry.transform.warp_perspective(
        template_mask, torch.inverse(H_pred_by_usanc), 
        imgs_search.shape[-2:], mode='bicubic',padding_mode='zeros',align_corners=True
    )
    
    # 只在mask值较高的地方进行替换，避免插值产生的低值区域
    template_valid_mask = (template_mask_transformed > 0.8).float()
    
    # 创建替换图像：在有效区域使用template内容，其他地方保持search图像不变
    replaced_image = imgs_search * (1 - template_valid_mask) + template_2_search * template_valid_mask
    
    return replaced_image