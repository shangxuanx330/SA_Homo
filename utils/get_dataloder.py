import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.homography_data_Google_Earth_and_Map_128x128 import  GoogleMapAndEarth_static_return_homo,GoogleMapAndEarth_dynamic_return_homo
from utils.augmentation_utils import get_train_transform_fn, get_val_transform_fn
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.homography_data_SA_Homo import SA_Homo_Homography_Dataset
import os
from torch.utils.data import DataLoader
from dataset.homography_data_gfnet import HomographyDataset_gfnet

def get_val_dataloder(config,args,split='val'):
    val_transform_fn  = get_val_transform_fn(config)
    
    if not split=='val':
        split='test' 
    if config["dataset_augmentations"]['dataset_type'] in ['GoogleMap']:
        dataset_val = GoogleMapAndEarth_dynamic_return_homo(
            transform=val_transform_fn,
            split=split,
            dataset_type=config["dataset_augmentations"]['dataset_type'], 
            rho=config["dataset_augmentations"]['rho'],
            x_flip = 0,
            y_flip = 0,
        )

    elif config["dataset_augmentations"]['dataset_type'] in ['GoogleEarth']:
            dataset_val = GoogleMapAndEarth_static_return_homo(
                transform=val_transform_fn,
                split=split,
                rho=config["dataset_augmentations"]['rho'],
                dataset_type=config["dataset_augmentations"]['dataset_type'], 
                retransformation_range=(0,0),
                is_retransformation=False,
                x_flip = 0,
                y_flip = 0,
            )
 
    elif config["dataset_augmentations"]['dataset_type'] in ['cv2_multi_datasets']:
        dataset_val = SA_Homo_Homography_Dataset(
            root_list = config["dataset_augmentations"]["dataset_list"],
            split=split,
            search_size = config["training_search_img_size"],
            template_patch_size = config["training_template_img_size"],
            min_scale_diff=config["dataset_augmentations"]["min_scale_diff"],
            max_scale_diff=config["dataset_augmentations"]["max_scale_diff"],
            min_overlap_ratio=config["dataset_augmentations"]["min_overlap_ratio"],
            transform = val_transform_fn,
            x_flip = config['x_flip'],
            y_flip = config['y_flip'],
            color  = config['imgs_color'],
            uni_model=False,
            is_val_static=config["dataset_augmentations"]["is_val_static"],
            val_dataset_folder_name=config["dataset_augmentations"]["val_dataset_folder_name"],
            margin=config["dataset_augmentations"]["margin"],
        )
    elif config["dataset_augmentations"]['dataset_type'] in ['gfnet_dronevehicle']:
        dataset_val = HomographyDataset_gfnet(
            dataset=config["dataset_augmentations"]['dataset_type'],
            split=split,
            initial_transforms=val_transform_fn,
            search_size = config["training_search_img_size"],
            template_patch_size = config["training_template_img_size"],
        )

    
    is_distributed = config.get("distributed", {}).get("enabled", False)
    # 验证集也需要分布式采样器（但不shuffle）
    if is_distributed:
        val_sampler = DistributedSampler(dataset_val, shuffle=False)
    else:
        val_sampler = None

    validation_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=min(os.cpu_count(), args.num_workers),
        pin_memory=True,
        collate_fn=None,
        drop_last=True,
    )

    # 如果使用分布式训练，也返回sampler引用以便设置epoch
    if is_distributed:
        return  validation_loader,  val_sampler
    else:
        return validation_loader, None

def get_train_dataloder(config,args,split='train'):
    train_transform_fn  = get_train_transform_fn(config)
    
    if config["dataset_augmentations"]['dataset_type'] in ['GoogleMap']:  
        dataset_train = GoogleMapAndEarth_dynamic_return_homo(
            transform=train_transform_fn,
            dataset_type=config["dataset_augmentations"]['dataset_type'], 
            split=split,
            rho=config["dataset_augmentations"]['rho'],
            x_flip = config['x_flip'],
            y_flip = config['y_flip'],
        )

    elif config["dataset_augmentations"]['dataset_type'] in ['GoogleEarth']:
            dataset_train = GoogleMapAndEarth_static_return_homo(
                transform=train_transform_fn,
                split=split,
                rho=config["dataset_augmentations"]['rho'],
                dataset_type=config["dataset_augmentations"]['dataset_type'], 
                retransformation_range=config["dataset_augmentations"]["retransformation_range"],
                is_retransformation=config["dataset_augmentations"]["is_retransformation"],
                x_flip = config['x_flip'],
                y_flip = config['y_flip'],
            )
    
    elif config["dataset_augmentations"]['dataset_type'] in ['cv2_multi_datasets']:
         dataset_train = SA_Homo_Homography_Dataset(
            root_list = config["dataset_augmentations"]["dataset_list"],
            split=split,
            search_size = config["training_search_img_size"],
            template_patch_size = config["training_template_img_size"],
            min_scale_diff=config["dataset_augmentations"]["min_scale_diff"],
            max_scale_diff=config["dataset_augmentations"]["max_scale_diff"],
            min_overlap_ratio=config["dataset_augmentations"]["min_overlap_ratio"],
            transform = train_transform_fn,
            x_flip = config['x_flip'],
            y_flip = config['y_flip'],
            color  = config['imgs_color'],
            is_val_static=config["dataset_augmentations"]["is_val_static"],
            uni_model=False,
            val_dataset_folder_name=config["dataset_augmentations"]["val_dataset_folder_name"],
            margin=config["dataset_augmentations"]["margin"],
         )
    elif config["dataset_augmentations"]['dataset_type'] in ['gfnet_dronevehicle']:
        dataset_train = HomographyDataset_gfnet(
            dataset=config["dataset_augmentations"]['dataset_type'],
            split=split,
            initial_transforms=train_transform_fn,
            search_size = config["training_search_img_size"],
            template_patch_size = config["training_template_img_size"],
        )

    # 设置分布式采样器
    is_distributed = config.get("distributed", {}).get("enabled", False)
    if is_distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        shuffle = False  # 使用sampler时不能shuffle
    else:
        train_sampler = None
        shuffle = True

    training_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=min(os.cpu_count(), args.num_workers),
        pin_memory=True,
        collate_fn=None,
        drop_last=True
    )

    # 如果使用分布式训练，也返回sampler引用以便设置epoch
    if is_distributed:
        return training_loader, train_sampler
    else:
        return training_loader, None

def get_dataloader(config,args):

    is_distributed = config.get("distributed", {}).get("enabled", False)

    training_loader,train_sampler = get_train_dataloder(config,args,split='train')
    validation_loader,val_sampler = get_val_dataloder(config,args,split='val')

    # 如果使用分布式训练，也返回sampler引用以便设置epoch
    if is_distributed:
        return training_loader, validation_loader, train_sampler, val_sampler
    else:
        return training_loader, validation_loader, None, None
