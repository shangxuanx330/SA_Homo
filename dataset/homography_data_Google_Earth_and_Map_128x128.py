import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import Dataset
import os, json, cv2
from glob import glob
import random
import numpy as np
import cv2
import torch
import kornia
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),              # 将PIL Image或numpy.ndarray转换为tensor，并归一化至[0,1]
    transforms.Normalize((0.5,), (0.5,)), # 使得像素值分布在[-1,1]之间
])
import kornia

def generate_homo(img1, img2, homo_parameter):

    # define corners of image patch
    marginal, perturb, patch_size = homo_parameter["marginal"], homo_parameter["perturb"], homo_parameter["patch_size"]
    height, width = homo_parameter["height"], homo_parameter["width"]
    x = random.randint(marginal, width - marginal - patch_size)
    y = random.randint(marginal, height - marginal - patch_size)
    top_left = (x, y)
    bottom_left = (x, patch_size + y - 1)
    bottom_right = (patch_size + x - 1, patch_size + y - 1)
    top_right = (patch_size + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    
    img1 = img1[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :] #(192,192)
    img2 = img2[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :] #(192,192)

    four_pts = four_pts - four_pts[np.newaxis, 0] + marginal # 将top_left设置为(marginal, marginal)
    (top_left, top_right, bottom_left, bottom_right) = four_pts
    
    
    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb, perturb)
            t2 = random.randint(-perturb, perturb)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 =   perturb // (i + 1)
            t2 = - perturb // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    
    warped_img1 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
    
    patch_img1 = warped_img1[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :] #(128,128)
    patch_img2 = img2[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :] #(128,128)
    ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
    org_pts = torch.tensor(org_pts, dtype=torch.float32)
    dst_pts = torch.tensor(dst_pts, dtype=torch.float32)

    new_org_four_pts = np.array(org_pts - marginal, dtype=np.float32)
    new_dst_four_pts = np.array(dst_pts - marginal, dtype=np.float32)

    new_H_inverse = cv2.getPerspectiveTransform(new_dst_four_pts, new_org_four_pts)

    return patch_img1, patch_img2, ground_truth, org_pts, dst_pts, warped_img1, img2 , new_H_inverse

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

class GoogleMapAndEarth_static_return_homo(Dataset):
    def __init__(self, dataset_type,split='train',
                 rho=32,retransformation_range=(-16,16),
                 is_retransformation=False, 
                 x_flip=0, 
                 y_flip=0,
                 transform=transform, 
                 is_validation=False):
        
        self.split = split
        self.transform = transform
        self.retransformation_range = retransformation_range
        self.is_retransformation = is_retransformation
        self.x_flip = x_flip
        self.y_flip = y_flip
        self.is_validation = is_validation
        self.dataset_type = dataset_type

        if dataset_type in ['GoogleEarth']:
            if split == 'train':
                self.img1_path = 'datasets/GoogleEarth/train2014_template/'
                self.img2_path = 'datasets/GoogleEarth/train2014_input/'
                self.label_path = 'datasets/GoogleEarth/train2014_label/'
            else:
                self.img1_path = 'datasets/GoogleEarth/val2014_template/'
                self.img2_path = 'datasets/GoogleEarth/val2014_input/'
                self.label_path = 'datasets/GoogleEarth/val2014_label/'

      

        self.img_name = os.listdir(self.img1_path)

    def __len__(self):  
        return len(self.img_name)

    def retransformation(self,org_pts,patch_img1_warp):
        low,high = self.retransformation_range
        random_int_tensor = ((torch.rand(size=(4, 2))) * (high-low) + low)
        org_pts_perturbation = org_pts + random_int_tensor
        img_shape = patch_img1_warp.shape[1:]

        H = kornia.geometry.transform.get_perspective_transform(org_pts.unsqueeze(dim=0),org_pts_perturbation.unsqueeze(dim=0))
        img_warp = kornia.geometry.transform.warp_perspective(patch_img1_warp.unsqueeze(dim=0),H,img_shape).squeeze(dim=0)

        return img_warp , random_int_tensor

    def __getitem__(self, index):  
        
        patch_img1_warp_path = self.img1_path + self.img_name[index]
        large_img2_path = self.img2_path + self.img_name[index]
        patch_img1_warp = Image.open(patch_img1_warp_path)
        large_img2 = Image.open(large_img2_path)

        if self.transform:
            large_img2 = self.transform(large_img2)
            patch_img1_warp = self.transform(patch_img1_warp)

        with open(self.label_path + self.img_name[index].split('.')[0] + '_label.txt', 'r') as outfile:
            data = json.load(outfile)

        top_left = [data['location'][0]['top_left_u'], data['location'][0]['top_left_v']]
        top_right = [data['location'][1]['top_right_u'], data['location'][1]['top_right_v']]
        bottom_left = [data['location'][2]['bottom_left_u'], data['location'][2]['bottom_left_v']]
        bottom_right = [data['location'][3]['bottom_right_u'], data['location'][3]['bottom_right_v']]
        
        org_pts = torch.tensor([[32,32], [159,32], [32,159], [159,159]], dtype=torch.float32)
        dst_pts = torch.tensor([top_left, top_right, bottom_left, bottom_right], dtype=torch.float32)
        four_gt = (dst_pts - org_pts)

        org_pts = torch.tensor([[0,0], [128 - 1,0], [0,128 - 1], [128 - 1,128 - 1]], dtype=torch.float32)
        dst_pts = four_gt + org_pts
        
        if self.is_retransformation:
            patch_img1_warp , random_int_tensor = self.retransformation(org_pts, patch_img1_warp)
            org_pts += random_int_tensor
            four_gt = (dst_pts - (org_pts + random_int_tensor))

        patch_img2 = large_img2[:, 32:160, 32:160]
        tl_gt = torch.tensor([0, 0],dtype=torch.float32)
        H_matrix = kornia.geometry.transform.get_perspective_transform(dst_pts.unsqueeze(dim=0), org_pts.unsqueeze(dim=0)).squeeze(dim=0)
        patch_img2, patch_img1_warp, H_matrix = random_flip_and_adjust_H_matrix( patch_img2, patch_img1_warp, H_matrix, self.x_flip, self.y_flip)

        search_scale_to_ori = torch.stack([torch.tensor(1.0, dtype=torch.float32), 
                                    torch.tensor(1.0, dtype=torch.float32)])

        actual_scale_diff = torch.tensor(1.0, dtype=torch.float32)
        
        return patch_img2, patch_img1_warp, tl_gt, H_matrix,search_scale_to_ori,actual_scale_diff,self.dataset_type,large_img2_path,patch_img1_warp_path
    
class GoogleMapAndEarth_dynamic_return_homo(Dataset):
    def __init__(self, split,dataset_type,rho = 32,x_flip=0, y_flip=0, transform=transform, is_validation=False):
        self.rho = rho
        self.transform=transform 
        self.dataset_type = dataset_type
        self.is_validation = is_validation
        self.homo_parameter = {"marginal":32, "perturb":rho, "patch_size":128}
        self.x_flip = x_flip
        self.y_flip = y_flip

        if dataset_type in ['GoogleMap',]:
            if split == 'train':
                root_img1 = 'datasets/GoogleMap/train2014_input/search'
                root_img2 = 'datasets/GoogleMap/train2014_template_original'        
            else:
                root_img1 = 'datasets/GoogleMap/val2014_input'
                root_img2 = 'datasets/GoogleMap/val2014_template_original'
      
       
        self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.jpg')))
        self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.jpg')))                

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img_size = self.homo_parameter["patch_size"] + 2 * self.homo_parameter["marginal"]
        img1 = cv2.resize(img1, (img_size, img_size))
        img2 = cv2.resize(img2, (img_size, img_size))
            
        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape
        
        patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, large_img1_warp, large_img2,H_matrix = generate_homo(img1, img2, homo_parameter=self.homo_parameter)

        if self.transform:
            patch_img2 = Image.fromarray(patch_img2)
            patch_img1_warp =  Image.fromarray(patch_img1_warp)
            patch_img2 = self.transform(patch_img2)
            patch_img1_warp = self.transform(patch_img1_warp)

        tl_gt = torch.tensor([0, 0],dtype=torch.float32)
        H_matrix = torch.from_numpy(H_matrix).to(torch.float32)
        patch_img2, patch_img1_warp, H_matrix = random_flip_and_adjust_H_matrix(patch_img2, patch_img1_warp, H_matrix,self.x_flip, self.y_flip)

        search_scale_to_ori = torch.stack([torch.tensor(1.0, dtype=torch.float32), 
                                    torch.tensor(1.0, dtype=torch.float32)])

        actual_scale_diff = torch.tensor(1.0, dtype=torch.float32)

        return patch_img2, patch_img1_warp, tl_gt, H_matrix,search_scale_to_ori,actual_scale_diff,self.dataset_type,self.image_list_img2[index],self.image_list_img1[index]
            

    