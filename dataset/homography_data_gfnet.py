import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import kornia.geometry.transform as KGT
import kornia
import torch.nn.functional as F


def random_four_points(deform_area, w, h, img, bi=False, H=None):
    topleft = [
        torch.randint(0, deform_area, size=(1, )),
        torch.randint(0, deform_area, size=(1, ))
    ]
    topright = [
        torch.randint(w-deform_area, w, size=(1, )),
        torch.randint(0, deform_area, size=(1, ))
    ]
    botright = [
        torch.randint(w-deform_area, w, size=(1, )),
        torch.randint(h-deform_area, h, size=(1, ))
    ]
    botleft = [
        torch.randint(0, deform_area, size=(1, )),
        torch.randint(h-deform_area, h, size=(1, ))
    ]
    tgt_points = torch.tensor([[deform_area//2, deform_area//2], [w-deform_area//2-1, deform_area//2], [w-deform_area//2-1, h-deform_area//2-1], [deform_area//2, h-deform_area//2-1]]).float() ## 4x2
    
    if bi:
        src_points = torch.tensor([topleft, topright, botright, botleft]).float() ## 4x2
    else:
        src_points = tgt_points
    if H is None:
        H = KGT.get_perspective_transform(src_points.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0) ## 3x3
    # flow_points = src_points - tgt_points
    
    warped_img = KGT.warp_perspective(img.unsqueeze(0), H.unsqueeze(0), (h, w)).squeeze(0) # C H W
    warped_img = warped_img[:, deform_area//2:h-deform_area//2, deform_area//2:w-deform_area//2] ## C 128 128
    
    return H, warped_img

def randomH(img1, img2, crop_size, input_size, deformation_ratio=0.33, bi=True):
    c1, h1, w1 = img1.shape
    c2, h2, w2 = img2.shape
    assert c1 == c2
    assert h1 == h2
    assert w1 == w2
    
    if w1<=crop_size or h1<=crop_size:
        size = crop_size + 10
        o_resize = transforms.Resize(size=size, interpolation=3, antialias=None) ##3 means bicubic
        img1, img2 = o_resize(img1), o_resize(img2)
    c1, h1, w1 = img1.shape    
    crop_top_left = [torch.randint(0, w1-crop_size, size=(1,)),
                     torch.randint(0, h1-crop_size, size=(1,))
                     ]
    img1 = img1[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    img2 = img2[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    
    c, h_original, w_original = img1.shape
    deform_area = int(w_original * deformation_ratio)
    
    H_1t, img1 = random_four_points(deform_area, w_original, h_original, img1, bi=True)
    H_2t, img2 = random_four_points(deform_area, w_original, h_original, img2, bi=bi)
        
    H_1t2t = H_2t @ H_1t.inverse()

    src_points = torch.tensor([[deform_area//2, deform_area//2], [w_original-deform_area//2-1, deform_area//2], [w_original-deform_area//2-1, h_original-deform_area//2-1], [deform_area//2, h_original-deform_area//2-1]]).float() ## 4x2
    tgt_points = kornia.geometry.transform_points(H_1t2t.unsqueeze(0), src_points.unsqueeze(0)).squeeze(0)
    flow = tgt_points - src_points
    _, h, w = img1.shape
    src_points = torch.tensor([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).float()
    tgt_points = src_points + flow
    
    H_s2t = KGT.get_perspective_transform(src_points.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0) ## 3x3
    
    if input_size.size[0] != h or input_size.size[1] != w:
        img1, img2 = input_size(img1), input_size(img2)
        _, h_input, w_input = img1.shape

        H_s2t = torch.diag(torch.tensor([h_input/h, h_input/h, 1.])).float() @ \
        H_s2t @ \
        torch.diag(torch.tensor([w_input/w, w_input/w, 1.])).float().inverse()
    else:
        _, h_input, w_input = img1.shape
    
    warped_src_return = KGT.warp_perspective(img1.unsqueeze(0), H_s2t.unsqueeze(0), (h_input, w_input)).squeeze(0) # C H W
    
    return img2, img1, H_s2t, warped_src_return ## img1: src，img2: tgt

def crop(img1, img2, crop_size=512):
    c1, h1, w1 = img1.shape
    c2, h2, w2 = img2.shape
    assert c1 == c2
    assert h1 == h2
    assert w1 == w2
    
    if w1<=crop_size or h1<=crop_size:
        size = min(w1, h1) + 3
        o_resize = transforms.Resize(size=size, antialias=None)
        img1, img2 = o_resize(img1), o_resize(img2)
    c1, h1, w1 = img1.shape    
    crop_top_left = [torch.randint(0, w1-crop_size, size=(1,)),
                     torch.randint(0, h1-crop_size, size=(1,))
                     ]
    img1 = img1[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    img2 = img2[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    
    return img1, img2  
    
class HomographyDataset_gfnet(Dataset):
    def __init__(self,
                 dataset,
                 split,
                 gfnet_datasets_folder='datasets/gfnet_dronevehicle',
                 input_resolution=(448,448),
                 search_size=(768,768), 
                 template_patch_size=(768,768),
                 initial_transforms=None,
                 bi=True,
                 normalize=False,
                 deformation_ratio=[0.3],
                 **kwargs):
        super().__init__()

        self.split = split
        self.dataset = dataset
        self.gfnet_datasets_folder = gfnet_datasets_folder
        assert input_resolution is not None, 'you should provide an input resolution.'
        self.input_resolution = input_resolution
        self.search_size = search_size
        self.template_patch_size = template_patch_size
        self.initial_transforms = initial_transforms
        self.bi = bi
        self.input_resize = transforms.Resize(size=self.input_resolution, interpolation=3, antialias=None)

        self.normalize = normalize
        self.input_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.deformation_ratio = deformation_ratio
        imgs0 = []
        imgs1 = []
        
        if split == 'train':
            if dataset == 'gfnet_dronevehicle':
                path = f'{self.gfnet_datasets_folder}/train/VIS-IR-drone'
                test_list = open(f'{path}/test_list_original.txt').read().split('\n')
                all_list = os.listdir(f'{path}/train/trainimg/')
                train_list = [x for x in all_list if x not in test_list][:5000]
                for image_name in train_list:
                    if torch.rand(1)>0.5:
                        imgs0.append(f'{path}/train/trainimg/' + image_name)
                        imgs1.append(f'{path}/train/trainimgr/' + image_name) ## r
                    else:
                        imgs0.append(f'{path}/train/trainimgr/' + image_name) ## r
                        imgs1.append(f'{path}/train/trainimg/' + image_name)
            elif dataset == 'googlemap':
                path = f'{self.gfnet_datasets_folder}/train/GoogleMap'
                train_list = os.listdir(f'{path}/map/')[:5000]
                for image_name in train_list:
                    if torch.rand(1)>0.5:
                        imgs0.append(f'{path}/satellite/' + image_name)
                        imgs1.append(f'{path}/map/' + image_name)
                    else:
                        imgs0.append(f'{path}/map/' + image_name)
                        imgs1.append(f'{path}/satellite/' + image_name)
            elif dataset == 'glunet_448x448_occlusion':
                path = f'{self.gfnet_datasets_folder}/train/glunet_448x448_occlusion/target'
                train_list = glob.glob(os.path.join(path, '*'))
                self.H_stg = []
                self.mask = []
                for image_path in train_list:
                    image_name = image_path.split('/')[-1]
                    imgs0.append(image_path)
                    imgs1.append(os.path.join(path.replace('target', 'source'), image_name))
                    self.mask.append(os.path.join(path.replace('target', 'mask'), image_name))
                    self.H_stg.append(os.path.join(path.replace('target', 'H_s2t'), image_name.replace('jpg', 'json')))
        elif self.split == 'test':
            self.input_resize = transforms.Compose  ([
                transforms.Resize(size=input_resolution, interpolation=3),
                transforms.ToTensor()
                ])                      
            if dataset == 'gfnet_dronevehicle':
                path = f'{self.gfnet_datasets_folder}/test/visir_1k_448x448/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('png', 'json')) for i in test_list]
            elif dataset == 'googlemap':
                path = f'{self.gfnet_datasets_folder}/test/googlemap_1k_448x448_new/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('jpg', 'json')) for i in test_list]
            elif dataset == 'googlemap_224x224':
                path = f'{self.gfnet_datasets_folder}/test/googlemap_1k_224x224/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('jpg', 'json')) for i in test_list]
            elif dataset == 'googlemap_672x672':
                path = f'{self.gfnet_datasets_folder}/test/googlemap_1k_672x672/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('jpg', 'json')) for i in test_list]            
            elif dataset == 'mscoco':
                path = f'{self.gfnet_datasets_folder}/test/mscoco_1k_448x448/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('png', 'json')) for i in test_list]
                test_list = os.listdir(path)
            imgs0 = [os.path.join(path, i) for i in test_list] ## target
            imgs1 = [os.path.join(path.replace('target', 'source'), i) for i in test_list] ## source
            
        self.imgs0 = imgs0
        self.imgs1 = imgs1

    def __len__(self):
        return len(self.imgs0)

    def resize_tensor(self, img, size):
        img = img.unsqueeze(0)  # [1, C, H, W]
        img = F.interpolate(
            img,
            size=size,           # (H, W) 或 int
            mode='bicubic',      # interpolation=3 对应 bicubic
            align_corners=False
        )
        return img.squeeze(0)
        
    def __getitem__(self, index, visualization=False):

        img0 = Image.open(self.imgs0[index]) ## target
        if img0.mode != 'RGB':
            img0 = img0.convert('RGB')
        img1 = Image.open(self.imgs1[index])
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if self.split == 'train':
            if self.dataset == 'gfnet_dronevehicle':
                img0 = np.array(img0) ## H W C
                img1 = np.array(img1)
                h0, w0 = img0.shape[:2]
                h1, w1 = img1.shape[:2]
                assert h0 == h1
                assert w0 == w1
                img0 = Image.fromarray(img0[100:-100, 100:-100])
                img1 = Image.fromarray(img1[100:-100, 100:-100])
            if self.dataset == 'googlemap':
                img0 = np.array(img0) ## H W C
                img1 = np.array(img1)
                h0, w0 = img0.shape[:2]
                h1, w1 = img1.shape[:2]
                assert h0 == h1
                assert w0 == w1
                img0 = Image.fromarray(img0[:-100, :])
                img1 = Image.fromarray(img1[:-100, :])
            
            img0, img1 = self.initial_transforms(img0), self.initial_transforms(img1)

            bi = self.bi
            
            if 'glunet_' not in self.dataset:
                ## online generation
                deformation_ratio = float(random.sample(self.deformation_ratio, 1)[0])
                crop_size = int(self.input_resolution[0]/(1-deformation_ratio))
                img0, img1, H_s2t, _ = randomH(img0, img1, crop_size=crop_size, input_size=self.input_resize, deformation_ratio=deformation_ratio, bi=bi)
            else:
                ## offline generation
                with open(self.H_stg[index], 'r') as json_file:
                    data = json.load(json_file)
                H_s2t = torch.tensor(data['H']).float() ##  3 3
                  
            if self.normalize:
                img0, img1 = self.input_norm(img0), self.input_norm(img1)
                  
        elif self.split == 'test':
            
            w0_original, h0_original = img0.size
            w1_original, h1_original = img1.size
            img0, img1 = self.initial_transforms(img0), self.initial_transforms(img1)
            
            with open(self.H_stg[index], 'r') as json_file:
                data = json.load(json_file)
            H_s2t = torch.tensor(data['H']).float() ##  3 3

        
            res_h, res_w = self.input_resolution
            H_s2t = torch.diag(torch.tensor([res_w/w1_original, res_h/h1_original, 1.])).float() @ \
            H_s2t @ \
            torch.diag(torch.tensor([res_w/w0_original, res_h/h0_original, 1.])).float().inverse()
        
        H_t2s = torch.inverse(H_s2t)
        tl_gt = torch.tensor([0, 0], dtype=torch.float32)
        
        # Resize img0 to search_size and img1 to template_patch_size
        img0 = self.resize_tensor(img0, self.search_size)
        img1 = self.resize_tensor(img1, self.template_patch_size)
        
        # Adjust H_t2s for the new image sizes
        scale_target = torch.tensor([self.search_size[1] / self.input_resolution[1], 
                                      self.search_size[0] / self.input_resolution[0], 1.0])
        scale_source = torch.tensor([self.template_patch_size[1] / self.input_resolution[1], 
                                      self.template_patch_size[0] / self.input_resolution[0], 1.0])
        H_t2s = torch.diag(scale_source) @ H_t2s @ torch.diag(1.0 / scale_target)
        
        default_scale_factor = torch.tensor((self.input_resolution[0]) / (self.search_size[0]), dtype=torch.float32)
        scale_diff = torch.tensor(1.0, dtype=torch.float32)
    
        return img0, img1, tl_gt,H_t2s, default_scale_factor,scale_diff, self.dataset,self.imgs0[index],self.imgs1[index]
