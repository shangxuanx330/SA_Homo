import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms 
import torch
import numpy as np
from dataset.dataset_utils import get_4_pts,crop_valid_region,calculate_quadrilateral_area
import cv2
import random
import kornia

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
    
    # 定义img1的四个角点
    corners1 = torch.tensor([[0, 0], [W1-1, 0], [0, H1-1], [W1-1, H1-1]], 
                            dtype=H_matrix.dtype, device=H_matrix.device)
    
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

def generate_homo(img1, img2, homo_parameter):
    """
    生成单应性变换的训练数据
    
    Args:
        img1: 源图像
        img2: 目标图像
        homo_parameter: 参数字典，包含:
            - patch_size: (h,w) 图像块的大小
            - marginal: (marginal_x, marginal_y) 或 int 边缘padding大小
            - perturb(x,y): 扰动范围
            - height: 图像高度
            - width: 图像宽度
    
    Returns:
        patch_img1: ndarray, shape=(patch_h, patch_w, c)
            变换后的图像1的patch区域，与patch_img2对应
        
        patch_img2: ndarray, shape=(patch_h, patch_w, c)
            原始图像2中的patch区域，作为参考区域
        
        ground_truth: torch.Tensor, shape=(4, 2)
            四个角点的位移向量，记录了变换前后角点的偏移量
            用于训练网络预测变换参数
        
        org_pts: torch.Tensor, shape=(4, 2) 先w后h
            原始四个角点的坐标(包含marginal偏移)
            顺序为：[左上, 右上, 左下, 右下]
        
        dst_pts: torch.Tensor, shape=(4, 2) 先w后h
            变换后的四个角点坐标(包含marginal偏移)
            在原始点基础上添加了随机扰动
        
        warped_img1: ndarray, shape=(h, w, c)
            整个图像1经过单应性变换后的完整结果
        
        img2: ndarray, shape=(h, w, c)
            原始参考图像，未经任何变换
        
        new_H_inverse: ndarray, shape=(3, 3)
            去除marginal偏移的逆单应性矩阵
            用于将变换后的坐标映射回原始坐标系
    """
    
    # 处理marginal参数，支持单一值或(x,y)分离值
    marginal = homo_parameter["marginal"]

    marginal_x, marginal_y = marginal
    perturb, patch_size = homo_parameter["perturb"], homo_parameter["patch_size"]
    height, width = homo_parameter["height"], homo_parameter["width"]
    
    # 使用分离的marginal值来随机选择patch位置
    x = random.randint(marginal_x, width - marginal_x - patch_size[1])
    y = random.randint(marginal_y, height - marginal_y - patch_size[0])

    top_left = (x, y)
    bottom_left = (x, patch_size[0] + y - 1)
    bottom_right = (patch_size[1] + x - 1, patch_size[0] + y - 1)
    top_right = (patch_size[1] + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    
    # 使用分离的marginal值来裁剪图像
    img1 = img1[top_left[1]-marginal_y:bottom_right[1]+marginal_y+1, 
                top_left[0]-marginal_x:bottom_right[0]+marginal_x+1, :] 
    img2 = img2[top_left[1]-marginal_y:bottom_right[1]+marginal_y+1, 
                top_left[0]-marginal_x:bottom_right[0]+marginal_x+1, :] 

    # 调整坐标系，将top_left设置为(marginal_x, marginal_y)
    four_pts = four_pts - four_pts[np.newaxis, 0] + np.array([marginal_x, marginal_y])
    (top_left, top_right, bottom_left, bottom_right) = four_pts
    
    perturb_x, perturb_y = perturb
    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb_x, perturb_x)
            t2 = random.randint(-perturb_y, perturb_y)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 =   perturb_x // (i + 1)
            t2 = - perturb_y // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    
    warped_img1 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
    if(warped_img1.ndim==2):
        warped_img1 = warped_img1[..., np.newaxis]
    
    patch_img1 = warped_img1[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :] 
    patch_img2 = img2[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :] 

    new_dst_pts = np.array(dst_pts - np.array([marginal_x, marginal_y]), dtype=np.float32)
    H_patch_img2_to_warped_img1 = cv2.getPerspectiveTransform(new_dst_pts, org_pts)

    new_org_pts = np.array(org_pts - np.array([marginal_x, marginal_y]), dtype=np.float32)
    H_patch_img2_to_patch_img1 = cv2.getPerspectiveTransform(new_dst_pts, new_org_pts)

    # 优化：一次性转换所有numpy数组为tensor，减少重复转换开销
    ground_truth = torch.from_numpy(ground_truth).to(torch.float32)
    org_pts = torch.from_numpy(org_pts).to(torch.float32)
    dst_pts = torch.from_numpy(dst_pts).to(torch.float32)

    """
    img1中dst_pts在warped_img1中是org_pts

    new_H_inverse代表的是从patch_img2到warped_img1的映射
    """
    return patch_img1, patch_img2, ground_truth, org_pts, dst_pts, warped_img1, img2 , H_patch_img2_to_warped_img1, H_patch_img2_to_patch_img1

class SA_Homo_Homography_Dataset(Dataset):
    def __init__(self, 
                 root_list,  # List of dataset paths
                 split='train',  # 'train' or 'val'
                 search_size=(768,960), 
                 template_patch_size=(256,320),
                 min_scale_diff=1,
                 max_scale_diff=4,
                 margin=(0,0),
                 transform=transforms.ToTensor(), 
                 x_flip=0, 
                 y_flip=0,
                 color='gray',
                 uni_model=False,
                 min_overlap_ratio=0.4,
                 is_val_static=False,  # New parameter to control static validation set
                 val_dataset_folder_name='val_adaptive_resolution'
                 ) -> None:
        super().__init__()
        """
        CV2 implementation of the homography dataset
        """
        self.search_size = search_size
        self.template_patch_size = template_patch_size
        self.transform = transform
        self.x_flip = x_flip
        self.min_overlap_ratio = min_overlap_ratio
        self.y_flip = y_flip
        self.uni_model = uni_model
        self.min_scale_diff = min_scale_diff
        self.max_scale_diff = max_scale_diff
        self.homo_parameter = {"marginal": (margin[1], margin[0]), "perturb": (search_size[1] // 4, search_size[0] // 4), "patch_size": search_size}
        self.homo_parameter["height"], self.homo_parameter["width"] = (self.search_size[0] + margin[0]*2, self.search_size[1] + margin[1]*2)

        self.color = color
        self.split = split
        self.root_list = root_list
        self.is_val_static = is_val_static
        self.val_dataset_folder_name = val_dataset_folder_name
        
        # 预计算常用的tensor，避免重复创建
        self._init_cached_tensors()
        
        # Initialize lists to store samples
        self.samples = []
        
        # Load data (train or val)
        self._load_data()
        
        print(f"Loaded {len(self.samples)} samples from {len(root_list)} datasets")
    
    def _init_cached_tensors(self):
        """预计算常用的tensor，减少运行时开销"""
        h_t, w_t = self.template_patch_size
        
        # 缓存固定的corner points tensor
        self.cached_org_pts = torch.tensor([[0, 0],
                                          [0, w_t-1],
                                          [h_t-1, 0],
                                          [h_t-1, w_t-1]], dtype=torch.float32).unsqueeze(dim=0)
        
        # 缓存固定的gt_tl tensor
        self.cached_zero_tl = torch.tensor([0, 0], dtype=torch.float32)
        

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def _load_data(self):
        """Load data from multiple datasets"""
        for dataset_idx, dataset_root in enumerate(self.root_list):
            if self.split == 'val' and self.is_val_static:
                # Validation mode with static data - load from anno.txt
                dataset_name = os.path.basename(os.path.normpath(dataset_root))
                val_folder = os.path.join(dataset_root, self.val_dataset_folder_name)
                anno_file = os.path.join(val_folder, 'anno.txt')
                
                if not os.path.exists(val_folder) or not os.path.exists(anno_file):
                    print(f"Warning: {val_folder} or {anno_file} does not exist, skipping...")
                    continue
                
                # 获取相对路径部分作为dataset_name的一部分
                val_subfolder = val_folder.replace(dataset_root, '').lstrip('/')
                
                with open(anno_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    fields = line.strip().split()
                    if len(fields) != 16:
                        print(f"Warning: Invalid line in {anno_file}: {line.strip()}, skipping...")
                        continue
                    
                    # Parse annotation line
                    search_name, template_name, tl_x, tl_y = fields[:4]
                    H_matrix = np.array(fields[4:13], dtype=np.float32).reshape(3, 3)
                    search_scale = np.array(fields[13:15], dtype=np.float32)
                    scale_diff = np.array(fields[15], dtype=np.float32)
                    
                    # Construct full paths
                    search_path = os.path.join(val_folder, search_name)
                    template_path = os.path.join(val_folder, template_name)
                    
                    if not os.path.exists(search_path) or not os.path.exists(template_path):
                        print(f"Warning: Image missing - search: {search_path}, template: {template_path}, skipping...")
                        continue
                    
                    # Generate enhanced dataset name with size information
                    enhanced_dataset_name = f"{dataset_name}/{val_subfolder}/{search_path.split('/')[-3]}"
                    
                    self.samples.append({
                        'template_path': template_path,
                        'search_path': search_path,
                        'gt_tl': torch.tensor([float(tl_x), float(tl_y)], dtype=torch.float32),
                        'H_matrix': torch.from_numpy(H_matrix).to(torch.float32),
                        'search_scale': torch.from_numpy(search_scale).to(torch.float32),
                        'scale_diff': torch.from_numpy(scale_diff).to(torch.float32),
                        'dataset_name': enhanced_dataset_name
                    })
            else:
                # Training data - original logic
                template_folder = os.path.join(dataset_root, self.split, 'template')
                search_folder = os.path.join(dataset_root, self.split, 'search')
                
                if not os.path.exists(template_folder) or not os.path.exists(search_folder):
                    print(f"Warning: {template_folder} or {search_folder} does not exist, skipping...")
                    continue
                    
                template_imgs_names = sorted(os.listdir(template_folder))
                search_imgs_names = sorted(os.listdir(search_folder))
                
                # Ensure template and search images match
                if len(template_imgs_names) != len(search_imgs_names):
                    print(f"Warning: Number of template and search images do not match in {dataset_root}, skipping...")
                    continue
                    
                # Add full paths to the list
                for t_name, s_name in zip(template_imgs_names, search_imgs_names):
                    template_path = os.path.join(template_folder, t_name)
                    search_path = os.path.join(search_folder, s_name)
                    
                    self.samples.append({
                        'template_path': template_path,
                        'search_path': search_path,
                        'dataset_name': os.path.basename(dataset_root)
                    })

    def crop_patch_from_trans_template(self, transformed_image, H, search_size, crop_size):
        

        template_croped,tl = crop_valid_region(transformed_image, H,search_size, crop_size,min_overlap_ratio=self.min_overlap_ratio)

        if  template_croped is not None and tl is not None:
            gt_tl = torch.tensor([tl[1],tl[0]],dtype = torch.float)
            return template_croped , gt_tl
        else:
            w, h,_ = transformed_image.shape
            center_x, center_y = (w-crop_size[0])//2 , (h - crop_size[1])//2
            template_croped = transformed_image[center_y:center_y + crop_size[1], center_x : center_x + crop_size[0]]
            gt_tl = torch.tensor([center_y,center_x],dtype = torch.float)
            return template_croped , gt_tl

    def synchronized_resize(self, template_img, search_img, target_size):
        """
        同步裁剪并调整两张图像大小，保持相同的裁剪区域
        
        Args:
            template_img: PIL图像对象
            search_img: PIL图像对象
            target_size: 目标尺寸 (height, width)
        
        Returns:
            调整后的template_img和search_img
        """
        # 获取原始尺寸
        w, h = search_img.size  # PIL图像的size返回的是(width, height)
        target_h, target_w = target_size
        
        # 计算原始和目标的纵横比
        orig_aspect = w / h
        target_aspect = target_w / target_h
        
        # 如果纵横比相同或接近，直接调整大小
        if abs(orig_aspect - target_aspect) < 0.05:
            return template_img.resize((target_w, target_h), Image.BICUBIC), \
                search_img.resize((target_w, target_h), Image.BICUBIC)
        
        # 根据纵横比决定裁剪方式
        if orig_aspect > target_aspect:
            # 原图更宽，需要在宽度上裁剪
            new_w = int(h * target_aspect)
            max_x_offset = w - new_w
            x_offset = random.randint(0, max_x_offset)
            
            # 同步裁剪两个图像
            template_img = template_img.crop((x_offset, 0, x_offset + new_w, h))
            search_img = search_img.crop((x_offset, 0, x_offset + new_w, h))
        else:
            # 原图更高，需要在高度上裁剪
            new_h = int(w / target_aspect)
            max_y_offset = h - new_h
            y_offset = random.randint(0, max_y_offset)
            
            # 同步裁剪两个图像
            template_img = template_img.crop((0, y_offset, w, y_offset + new_h))
            search_img = search_img.crop((0, y_offset, w, y_offset + new_h))
        
        # 调整到目标尺寸
        template_img = template_img.resize((target_w, target_h), Image.BICUBIC)
        search_img = search_img.resize((target_w, target_h), Image.BICUBIC)
        
        return template_img, search_img
    
    def synchronized_resize_cv2(self, template_img, search_img, target_size):
        """
        Synchronously crop and resize two images using OpenCV
        """
        # Get original dimensions
        h, w = search_img.shape[:2]  # CV2 image shape returns (height, width)
        target_h, target_w = target_size
        
        # Resize to target size
        template_img = cv2.resize(template_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        search_img = cv2.resize(search_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        return template_img, search_img
    
    def get_random_crop_size(self):
        """
        uniform sampling scale from [min_scale, max_scale]
        """
        # Randomly select width using uniform distribution for continuous values
        scale_diff = random.uniform(self.min_scale_diff, self.max_scale_diff)
        
        # Calculate corresponding height based on aspect ratio
        hc = int(self.search_size[0] / np.abs(scale_diff))
        wc = int(self.search_size[1] / np.abs(scale_diff))

        return hc, wc, scale_diff
    
    def __getitem__(self, i):
        sample = self.samples[i]
        dataset_name = sample['dataset_name']
        
        if self.split == 'val' and self.is_val_static:
            # 验证模式的处理保持不变，但优化图像读取
            return self._get_validation_item(sample, dataset_name)
        
        h_crop, w_crop, scale_diff = self.get_random_crop_size()
        
        
        template_img_path = sample['template_path']
        search_img_path = sample['search_path']

        if self.uni_model:
            template_img_path = search_img_path

        # 优化图像读取：并行读取两张图像（如果路径不同）
        template_img, search_img = self._load_image_pair(template_img_path, search_img_path)
        
        h_s_ori, w_s_ori = search_img.shape[:2]
        h_t_ori, w_t_ori = template_img.shape[:2]
        h_s, w_s = self.search_size
        h_t, w_t = self.template_patch_size
        if (h_s_ori, w_s_ori) != (self.homo_parameter["height"], self.homo_parameter["width"]) or (h_t_ori, w_t_ori) != (self.template_patch_size[0], self.template_patch_size[1]):
            template_img, search_img = self.synchronized_resize_cv2(template_img, search_img, (self.homo_parameter["height"], self.homo_parameter["width"]))
        is_gray = len(template_img.shape) == 2

        if is_gray:
            template_img = template_img[..., np.newaxis]
            search_img = search_img[..., np.newaxis]
       
        template_croped, patch_search, four_gt, org_pts, dst_pts, large_img1_warp, large_img2,  H_matrix_patch_img2_to_warped_img1 ,H_matrix_patch_img2_to_patch_img1= generate_homo(template_img, search_img, homo_parameter=self.homo_parameter)
    
        if self.min_scale_diff == 1 and self.max_scale_diff == 1:
            H_matrix_patch_img2_to_patch_img1 = torch.from_numpy(H_matrix_patch_img2_to_patch_img1).to(torch.float)
            gt_tl = torch.tensor([0,0],dtype=torch.float)
            dst_pts, org_pts = get_4_pts(gt_tl.unsqueeze(dim=0), (h_crop, w_crop), H_matrix_patch_img2_to_patch_img1.unsqueeze(dim=0), inner_dis=0, scale_factor=1)
        else:
            H_matrix_patch_img2_to_warped_img1 = torch.from_numpy(H_matrix_patch_img2_to_warped_img1).to(torch.float)
            template_croped, gt_tl = self.crop_patch_from_trans_template(large_img1_warp, H_matrix_patch_img2_to_warped_img1, (w_s, h_s), (w_crop, h_crop))
            dst_pts, org_pts = get_4_pts(gt_tl.unsqueeze(dim=0), (h_crop, w_crop), H_matrix_patch_img2_to_warped_img1.unsqueeze(dim=0), inner_dis=0, scale_factor=1)

        org_pts = self.cached_org_pts.clone()

        H_matrix_new = kornia.geometry.transform.get_perspective_transform(dst_pts[:,:,[1,0]], org_pts[:,:,[1,0]]).squeeze(dim=0)
        gt_tl = self.cached_zero_tl.clone()
        
        if (w_crop, h_crop)!=(h_t, w_t):
            template_croped = cv2.resize(template_croped, (w_t,h_t), interpolation=cv2.INTER_CUBIC)
        
        search_scale_to_ori = torch.stack([torch.tensor(h_s_ori/self.homo_parameter["height"], dtype=torch.float32), 
                                    torch.tensor(w_s_ori/self.homo_parameter["width"], dtype=torch.float32)])
        
        if self.transform:
            if is_gray:
                template_croped = template_croped.squeeze()
                patch_search = patch_search.squeeze()
            
            # Convert numpy arrays to PIL Images for torchvision transforms
            template_croped = Image.fromarray(template_croped)
            patch_search = Image.fromarray(patch_search)
            template_croped = self.transform(template_croped)
            patch_search = self.transform(patch_search)
        
        if self.x_flip != 0 or self.y_flip != 0:
                patch_search, template_croped, H_matrix_new = random_flip_and_adjust_H_matrix(patch_search, template_croped, H_matrix_new, self.x_flip, self.y_flip)
        

        area_in_search  = calculate_quadrilateral_area(dst_pts.squeeze(dim=0)[:,[1,0]])
        actual_scale_diff = torch.sqrt((self.template_patch_size[0] * self.template_patch_size[1]) / area_in_search)
        if actual_scale_diff < 1:
            actual_scale_diff = 1/actual_scale_diff

        return patch_search, template_croped, gt_tl, H_matrix_new, search_scale_to_ori, actual_scale_diff, dataset_name,search_img_path, template_img_path

    def _load_image_pair(self, template_path, search_path):
        """优化的图像对加载函数"""
        if self.color == 'gray':
            # 如果是同一张图像，只读取一次
            if template_path == search_path:
                img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                return img.copy(), img
            else:
                template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
                return template_img, search_img
        else:
            # 如果是同一张图像，只读取一次
            if template_path == search_path:
                img = cv2.imread(template_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img.copy(), img
            else:
                template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
                search_img = cv2.imread(search_path, cv2.IMREAD_COLOR)
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
                search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
                return template_img, search_img
    
    def _get_validation_item(self, sample, dataset_name):
       
        search_path = sample['search_path']
        template_path = sample['template_path']
        gt_tl = sample['gt_tl']
        H_matrix_new = sample['H_matrix']
        search_scale = sample['search_scale']
        scale_diff = sample['scale_diff']
        
        template_croped, search_img = self._load_image_pair(template_path, search_path)
        
        is_gray = len(search_img.shape) == 2
        if is_gray:
            search_img = search_img.squeeze()
            template_croped = template_croped.squeeze()

        # Apply transform
        if self.transform:
            search_img = Image.fromarray(search_img)
            template_croped = Image.fromarray(template_croped)
            search_img = self.transform(search_img)
            template_croped = self.transform(template_croped)

        return search_img, template_croped, gt_tl, H_matrix_new, search_scale, scale_diff, dataset_name, search_path, template_path

