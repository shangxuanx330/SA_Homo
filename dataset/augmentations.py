from typing import Tuple
import random
import torch
from torch import Tensor, nn
from torchvision.transforms import functional_tensor as F

class RandomCrop(nn.Module):
    def __init__(
        self,
        size_optical: Tuple[int, int],
        size_sar: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.size_optical = size_optical
        self.size_sar = size_sar

    def forward(
        self,
        img_optical: Tensor,
        img_sar: Tensor,
        box_tl: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        h0_old, w0_old = img_optical.shape[-2:]
        h1_old, w1_old = img_sar.shape[-2:]

        h0_new, w0_new = self.size_optical
        h1_new, w1_new = self.size_sar

        # Randomly crop a patch within the old SAR image.
        offset_y_sar = random.randint(0, h1_old - h1_new)
        offset_x_sar = random.randint(0, w1_old - w1_new)

        img_sar_cropped = img_sar[
            ...,
            offset_y_sar: offset_y_sar + h1_new,
            offset_x_sar: offset_x_sar + w1_new,
        ]

        t, l = int(box_tl[0]), int(box_tl[1])
        t += offset_y_sar
        l += offset_x_sar

        # Randomly crop a patch within the old optical image.
        # Make sure the newly cropped SAR image patch is included.
        offset_y_optical = random.randint(
            max(0, t + h1_new - h0_new),
            min(h0_old - h0_new, t)
        )
        offset_x_optical = random.randint(
            max(0, l + w1_new - w0_new),
            min(w0_old - w0_new, l)
        )

        img_optical_cropped = img_optical[
            ...,
            offset_y_optical: offset_y_optical + h0_new,
            offset_x_optical: offset_x_optical + w0_new,
        ]

        # Map the top-left coordinate of the SAR image patch
        # into the cropped optical image path.
        box_tl[0] += offset_y_sar - offset_y_optical
        box_tl[1] += offset_x_sar - offset_x_optical

        return img_optical_cropped, img_sar_cropped, box_tl

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(
        self,
        img_optical: Tensor,
        img_sar: Tensor,
        box_tl: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        w0 = img_optical.shape[-1]
        w1 = img_sar.shape[-1]

        if random.random() < self.p:
            img_optical = img_optical.flip(-1)
            img_sar = img_sar.flip(-1)
            box_tl[1] = w0 - w1 - box_tl[1]

        return img_optical, img_sar, box_tl

class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(
        self,
        img_optical: Tensor,
        img_sar: Tensor,
        box_tl: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        h0 = img_optical.shape[-2]
        h1 = img_sar.shape[-2]

        if random.random() < self.p:
            img_optical = img_optical.flip(-2)
            img_sar = img_sar.flip(-2)
            box_tl[0] = h0 - h1 - box_tl[0]

        return img_optical, img_sar, box_tl

class RandomRot90(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(
        self,
        img_optical: Tensor,
        img_sar: Tensor,
        box_tl: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        t, l = box_tl
        h0, w0 = img_optical.shape[-2:]
        h1, w1 = img_sar.shape[-2:]

        if random.random() < self.p:
            k = random.randint(1, 3)
            # Rotate both images counter-clockwise
            # by 90 degrees k times.
            img_optical = img_optical.rot90(k, dims=(-2, -1))
            img_sar = img_sar.rot90(k, dims=(-2, -1))

            if k == 1:
                t, l = w0 - w1 - l, t
            elif k == 2:
                t, l = h0 - h1 - t, w0 - w1 - l
            elif k == 3:
                t, l = l, h0 - h1 - t

        return img_optical, img_sar, torch.tensor((t, l))

class RandomBrightness(nn.Module):
    def __init__(self, ratio: float) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, img: Tensor) -> Tensor:
        factor = float(torch.empty(1).uniform_(1.0 - self.ratio, 1.0 + self.ratio))
        img = F.adjust_brightness(img, factor)

        return img

def get_perspective_bounds(H_inv, original_size):
    """
    计算透视变换后的图像边界
    Args:
        H_inv: S到画布的变换矩阵 [3, 3]
        original_size: 原始图像尺寸 (H, W)
    Returns:
        min_x, max_x, min_y, max_y: 变换后图像的边界
    """
    h, w = original_size
    corners = torch.tensor([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ], dtype=torch.float32).to(H_inv.device)
    
    corners_transformed = (H_inv @ corners.T).T
    corners_transformed = corners_transformed / corners_transformed[:, 2:]
    
    min_x = torch.floor(corners_transformed[:, 0].min()).int()
    max_x = torch.ceil(corners_transformed[:, 0].max()).int()
    min_y = torch.floor(corners_transformed[:, 1].min()).int()
    max_y = torch.ceil(corners_transformed[:, 1].max()).int()
    
    return min_x, max_x, min_y, max_y

def warp_and_shuffle_v2(S, T, H, k1, k2):
    """
    Args:
        S: 大图 [B, C, H, W]
        T: 小图 [B, C, h, w]
        H: S到T的单应性矩阵 [B, 3, 3]
        k1, k2: 分块数量
    Returns:
        result: 数据增强后的图像
    """
    B, C, H_s, W_s = S.shape
    B, C, H_t, W_t = T.shape
    
    # 确保尺寸可以被k1,k2整除
    assert H_t % k1 == 0 and W_t % k2 == 0, f"T size ({H_t},{W_t}) must be divisible by k1={k1}, k2={k2}"
    
    # 1. 计算H的逆矩阵（S到画布的变换）
    H_inv = torch.inverse(H)
    
    # 2. 计算变换后需要的画布大小
    min_x, max_x, min_y, max_y = get_perspective_bounds(H_inv[0], (H_s, W_s))
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    
    # 创建平移矩阵
    translation = torch.eye(3, device=H.device).unsqueeze(0).repeat(B, 1, 1)
    translation[:, 0, 2] = -min_x
    translation[:, 1, 2] = -min_y
    
    # 组合变换矩阵（S到画布的完整变换）
    H_full = translation @ H_inv
    
    # 3. 在更大的画布上进行变换
    S_warp = K.warp_perspective(S, H_full, (canvas_h, canvas_w))
    
    # 4. 截取T大小的区域
    x1, y1 = 0, 0
    x2, y2 = W_t, H_t
    S_sub = S_warp[:, :, y1:y2, x1:x2]
    
    # 5. 计算patch大小
    patch_h, patch_w = H_t // k1, W_t // k2
    
    # 6. 将S_sub和T重塑为patches
    # [B, C, H, W] -> [B, C, k1, patch_h, k2, patch_w]
    S_reshaped = S_sub.view(B, C, k1, patch_h, k2, patch_w)
    T_reshaped = T.view(B, C, k1, patch_h, k2, patch_w)
    
    # [B, C, k1, patch_h, k2, patch_w] -> [B, k1*k2, C, patch_h, patch_w]
    S_patches = S_reshaped.permute(0, 2, 4, 1, 3, 5).reshape(B, k1*k2, C, patch_h, patch_w)
    T_patches = T_reshaped.permute(0, 2, 4, 1, 3, 5).reshape(B, k1*k2, C, patch_h, patch_w)
    
    # 7. 对每个batch使用相同的随机序列进行打乱
    for b in range(B):
        idx = torch.randperm(k1 * k2)
        S_patches[b] = S_patches[b, idx]
        T_patches[b] = T_patches[b, idx]
    
    # 8. 重建图像
    # [B, k1*k2, C, patch_h, patch_w] -> [B, k1, k2, C, patch_h, patch_w]
    S_shuffled = S_patches.view(B, k1, k2, C, patch_h, patch_w)
    
    # [B, k1, k2, C, patch_h, patch_w] -> [B, C, H, W]
    S_sub_shuffled = S_shuffled.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H_t, W_t)
    
    # 9. 替换回原区域
    S_warp_new = S_warp.clone()
    S_warp_new[:, :, y1:y2, x1:x2] = S_sub_shuffled
    
    # 10. 变换回原始空间
    H_full_inv = torch.inverse(H_full)
    result = K.warp_perspective(S_warp_new, H_full_inv, (H_s, W_s))
    
    return result

def test_warp_and_shuffle():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    S = torch.randn(2, 3, 480, 640).to(device)  # batch_size=2
    T = torch.randn(2, 3, 120, 160).to(device)
    # H现在是S到T的变换矩阵
    H = torch.eye(3).unsqueeze(0).repeat(2, 1, 1).to(device)
    H[:, 0:2, 2] = torch.tensor([10., 20.])
    
    k1, k2 = 4, 4
    result = warp_and_shuffle_v2(S, T, H, k1, k2)
    
    # 验证输出尺寸
    assert result.shape == S.shape, f"Shape mismatch: {result.shape} != {S.shape}"
    return result

def augment_with_points(image: torch.Tensor, points: torch.Tensor):
    """
    对图像进行随机平移，确保平移后的图像包含所有给定点
    
    Args:
        image: shape为[c,h,w]的tensor
        points: shape为[4,2]的tensor, 每行为[h,w]坐标
    
    Returns:
        shifted_image: 平移后的图像
        shifted_points: 平移后的点坐标
    """
    _, h, w = image.shape
    
    # 计算点的边界框
    min_h, min_w = torch.min(points, dim=0)[0]
    max_h, max_w = torch.max(points, dim=0)[0]
    
    # 计算可移动的最大范围
    max_shift_up = int(min_h)
    max_shift_down = int(h - max_h - 1)
    max_shift_left = int(min_w)
    max_shift_right = int(w - max_w - 1)
    
    # 随机生成平移量
    shift_h = random.randint(-max_shift_up, max_shift_down)
    shift_w = random.randint(-max_shift_left, max_shift_right)
    
    # 创建新图像
    shifted_image = torch.zeros_like(image)
    
    # 计算源图像和目标图像的切片
    if shift_h >= 0:
        src_h_start, src_h_end = 0, h - shift_h
        dst_h_start, dst_h_end = shift_h, h
    else:
        src_h_start, src_h_end = -shift_h, h
        dst_h_start, dst_h_end = 0, h + shift_h
        
    if shift_w >= 0:
        src_w_start, src_w_end = 0, w - shift_w
        dst_w_start, dst_w_end = shift_w, w
    else:
        src_w_start, src_w_end = -shift_w, w
        dst_w_start, dst_w_end = 0, w + shift_w
    
    # 执行平移
    shifted_image[:, dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
        image[:, src_h_start:src_h_end, src_w_start:src_w_end]
    
    # 更新点坐标
    shifted_points = points.clone()
    shifted_points[:, 0] += shift_h  # 更新h坐标
    shifted_points[:, 1] += shift_w  # 更新w坐标
    
    return shifted_image, shifted_points

if __name__ == "__main__":
    # Test random crop.
    size_optical = (768, 1024)
    size_sar = (512, 512)
    size_optical_cropped = (480, 640)
    size_sar_cropped = (320, 320)

    tl = torch.tensor((120, 240), dtype=torch.int32)

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.misc import face

    img_optical = torch.from_numpy(face()).permute((2, 0, 1)).contiguous()
    img_sar = img_optical[
        ...,
        tl[0]: tl[0] + size_sar[0],
        tl[1]: tl[1] + size_sar[1],
    ]

    # transform = RandomVerticalFlip(p=1)

    transform = RandomCrop(
        size_optical=size_optical_cropped,
        size_sar=size_sar_cropped
    )

    img_optical_cropped, img_sar_cropped, tl = transform(
        img_optical,
        img_sar,
        tl
    )

    size_sar_cropped = img_sar_cropped.shape[-2:]
    img_sar_cropped_expected = img_optical_cropped[
        ...,
        tl[0]: tl[0] + size_sar_cropped[0],
        tl[1]: tl[1] + size_sar_cropped[1],
    ]

    assert torch.allclose(
        img_sar_cropped_expected,
        img_sar_cropped
    ), "Not matched."

    img_optical_cropped: Tensor = img_optical_cropped.permute((1, 2, 0))
    img_sar_cropped: Tensor = img_sar_cropped.permute((1, 2, 0))
    img_sar_cropped_expected: Tensor = img_sar_cropped_expected.permute(
        (1, 2, 0))

    fig, axes = plt.subplots(1, 3, num="Cropped", figsize=(18, 6))
    axes[0].imshow(img_optical_cropped)
    axes[1].imshow(img_sar_cropped)
    axes[2].imshow(img_sar_cropped_expected)

    axes[0].set_title("Optical")
    axes[1].set_title("SAR")
    axes[2].set_title("SAR - Expected")

    box = Rectangle(
        xy=(tl[1], tl[0]),
        height=size_sar_cropped[0],
        width=size_sar_cropped[1],
        color="red",
        fill=False
    )
    axes[0].add_patch(box)
    # plt.savefig("foo.png", dpi=300)
    plt.show()