
import sys
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import device as Device
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from utils.logger import setup_logger 
from utils.cfg_management import merge_args_into_config
from utils.utils import get_obj_from_str
from utils.get_dataloder import get_val_dataloder
from train.utils.get_args import parse_args
import json
from train.utils.get_pts import  get_grid_like_k_pts_gt_include_margin,get_4_pts_in_template_and_search_img,get_4_corner_in_search_img_by_indexes,get_pred_4_corner_in_search_img_by_dlt,get_pred_4_corner_in_search_img_by_usanc
from models.box_parser import get_k_pts_BoxParser
import kornia
from train.utils.get_error import get_mace
from train.utils.get_pts import  *
from utils.Evaluation import *
from inference.statistic_fn import analyze_and_save_results
from utils.get_seeds import set_seed
from utils.loading import load_checkpoint_validation

def statistic(
    args,
    config,    
    SCEM_model,
    IHERM_model,
    dataloader: DataLoader,
    dev: Device
):
    k = config['num_of_predited_pts']
    scale_factor = 1
    iter_each_scale = config['iter_each_scale']
    total_iter = sum(iter_each_scale)
    SCEM_model.eval()
    IHERM_model.eval()
    template_size = config['training_template_img_size']
 
    mace_by_IHERM_list =[]
    dataset_name_list = []
    img_path_list= []
    scale_diff_list = []
  
    idx = 0
    with torch.no_grad():
        for i in range(args.eval_loop_times):
            for imgs_search, imgs_template, tl_gt, H_matrix,search_scale, scale_diffs ,dataset_name, search_path, template_path in tqdm(
                dataloader,
                total=len(dataloader),
                unit="batch",
                desc="[ val ]",
                colour="blue",
            ):
                idx += 1
                four_corner_in_template_corresponding_pts_in_search_img_gt, four_corner_in_template = get_4_pts_in_template_and_search_img(tl_gt,template_size,H_matrix,0,scale_factor=scale_factor)
                four_corner_in_template_corresponding_pts_in_search_img_gt = four_corner_in_template_corresponding_pts_in_search_img_gt.to(dev, non_blocking=True)
                four_corner_in_template = four_corner_in_template.to(dev, non_blocking=True)
                search_scale = search_scale.to(dev, non_blocking=True).unsqueeze(dim=1)
                dataset_name_list.extend(dataset_name)
                img_path_list.extend(search_path)
                scale_diff_list.extend(scale_diffs)
                
                key_pts_gt_in_search_img, key_pts_gt_in_template_img = get_grid_like_k_pts_gt_include_margin(tl_gt,template_size,H_matrix,k)
                key_pts_gt_in_search_img = key_pts_gt_in_search_img.to(dev, non_blocking=True)
                key_pts_gt_in_template_img = key_pts_gt_in_template_img.to(dev, non_blocking=True)

                # Transfer data to target device.
                imgs_search = imgs_search.to(dev, non_blocking=True)
                imgs_template = imgs_template.to(dev, non_blocking=True)
                tl_gt = tl_gt.to(dev, non_blocking=True) #shape[1,2]
                H_matrix = H_matrix.to(dev, non_blocking=True)
                pred_sim_matrix,pred_score_map, pred_offset_map,x_search,x_template = SCEM_model(imgs_search, imgs_template)
                get_pred_pts = get_k_pts_BoxParser(stride=config["SECM_downsample_factor"])
                pts_pred_by_SCEM_model = get_pred_pts(pred_score_map, pred_offset_map).to(dev, non_blocking=True)
                if config['get_4_corner_method']=='get_4_corner_via_indexes':
                    four_corner_pred_by_first_stage = get_4_corner_in_search_img_by_indexes(pts_pred_by_SCEM_model,k)
                elif config['get_4_corner_method']=='get_4_corner_via_dlt':
                    four_corner_pred_by_first_stage = get_pred_4_corner_in_search_img_by_dlt(imgs_template.shape[-2:],pts_pred_by_SCEM_model,key_pts_gt_in_template_img)
                elif config['get_4_corner_method']=='get_4_corner_via_usanc':
                    four_corner_pred_by_first_stage = get_pred_4_corner_in_search_img_by_ransac(pts_pred_by_SCEM_model,key_pts_gt_in_template_img,imgs_template.shape[-2:])


                four_corner_gt_via_indexes = get_4_corner_in_search_img_by_indexes(key_pts_gt_in_template_img,k)
            
                H_SCEM_pred_search_2_template = kornia.geometry.transform.get_perspective_transform(
                                four_corner_pred_by_first_stage[:, :, [1, 0]], four_corner_gt_via_indexes[:, :, [1, 0]],
                            ).detach()


                imgs_search_warpped = kornia.geometry.transform.warp_perspective(imgs_search, H_SCEM_pred_search_2_template, \
                                                                            imgs_template.shape[-2:], padding_mode='border', \
                                                                            align_corners=True,  mode='bilinear')
            
                init_four_disp = torch.zeros_like(four_corner_pred_by_first_stage)
    
                
                with torch.no_grad():
                    _, x_search = SCEM_model.module.search_module(imgs_search_warpped)
                x_search, x_template = [x_search[i].detach() for i in range(len(x_search))], [x_template[i].detach() for i in range(len(x_template))]
                pred_delta = IHERM_model(x_search, x_template, init_four_disp)
            
                
                four_corner_pred_by_second_stage = transform_points_with_homography_hw(four_corner_in_template + pred_delta[total_iter-1],torch.inverse(H_SCEM_pred_search_2_template))
                mace_second_stage = get_mace(four_corner_pred_by_second_stage, four_corner_in_template_corresponding_pts_in_search_img_gt,4,search_scale=search_scale)
                mace_by_IHERM_list.extend(mace_second_stage.tolist())

    return mace_by_IHERM_list, dataset_name_list, img_path_list, scale_diff_list
             
def print_config(config, indent=0):
    """
    递归打印配置字典，支持嵌套字典
    每个顶级键占一行，子字典的键值对缩进显示
    """
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(' ' * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            logger.info(' ' * indent + f"{key}: {value}")

def load_dataset(args, config):
    validation_loader,  _ = get_val_dataloder(config,args,'val')
    return validation_loader

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)

    logger = setup_logger("SA-Homo", args.save_dir, 0)

    # Load model config and hyperparameters config.
    with open(args.config, "r") as f:
        config = json.load(f)

    merge_args_into_config(args, config)
    logger.info('=================================================================================')
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info('=================================================================================')
    print_config(config)
    logger.info('=================================================================================')

    # Try to use GPU.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare stateful training modules.
    epoch: int = 0
    batch_size: int = args.batch_size
    num_epochs: int = config["epochs"]
    
    SCEM_model_obj = get_obj_from_str(config['SCEM_model_name'])
    SCEM_model = SCEM_model_obj(
        num_features=config["SCEM_features_dim"],
        downsampling=config["SECM_downsample_factor"],
        color_format_search=config['imgs_color'],
        template_size=config['training_template_img_size'],
        search_size=config['training_search_img_size'],
        max_shape=config["max_shape"],
        d_model=config["SCEM_d_model"],
        n_heads=config["n_heads"],
        layer_names=['self', 'cross'] * config['cross_attention_layers_num'],
        head_kernel_size=config['head_kernel_size'],
        num_backbone_layes=config['num_backbone_layes'],
        num_head_layers=config['num_head_layers'],
        num_of_tasktoken=config['num_of_task_token'],
        num_of_predited_pts=config['num_of_predited_pts'],
        attention_drop_out_rate=config['attention_drop_out_rate'],
        cnn_act=config['cnn_activation'],
        att_act=config['att_activation'],
        cnn_norm=config['cnn_norm'],
        cnn_conv2d_has_bias=config['cnn_conv2d_has_bias'],
        cnn_bn_has_bias=config['cnn_bn_has_bias'],
        att_scales=config['att_scales'],
        att_norm=config['att_norm'],
        att_conv2d_has_bias=config['att_conv2d_has_bias'],
        att_bn_has_bias=config['att_bn_has_bias'],
        attn_type=config['attn_type'],
        use_share_encoder=config['use_share_encoder'],
        kernel_fn=config['kernel_fn'],
        SCEM_num_features_predition_head=config['SCEM_num_features_predition_head'],
        SCEM_feature_encoder_name=config['SCEM_feature_encoder_name'],
        dual_softmax_method=config['dual_softmax_method'],
    ).to(dev)

    IHERM_model_obj = get_obj_from_str(config['IHERM_model_name'])

    IHERM_model = IHERM_model_obj(
        num_features=config["IHERM_num_features"],
        downsampling=config["IHERM_downsample_factor"],
        imgs_color=config['imgs_color'],
        template_size=config['training_template_img_size'],
        search_size=config['training_search_img_size'],
        d_model=config["IHERM_d_model"],
        cnn_act = config['cnn_activation'],
        cnn_norm = config['cnn_norm'],
        cnn_conv2d_has_bias=config['cnn_conv2d_has_bias'],
        cnn_bn_has_bias=config['cnn_bn_has_bias'],
        scales = config['scales'],
        n_warp_each_scale = config['iter_each_scale'],
        num_backbone_layes = config['num_backbone_layes'],
        use_share_encoder=config['use_share_encoder'],
        num_of_predicted_pts=config['num_of_predited_pts'],
        ).to(dev)
    
   #多GPU并行计算
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs!", torch.cuda.device_count())
        SCEM_model = nn.DataParallel(SCEM_model)
        IHERM_model = nn.DataParallel(IHERM_model)
    
    if args.secm_resume:
        SCEM_model = load_checkpoint_validation(SCEM_model, args.secm_resume, "1/2 SCEM", dev)

    if args.IHERM_resume:
        IHERM_model = load_checkpoint_validation(IHERM_model, args.IHERM_resume, "2/2 IHERM", dev)
    
    args.batch_size = 1

    validation_loader = load_dataset(args, config)
        
    mace_list, dataset_name_list, img_path_list, scale_diff_list = statistic(
            args=args,
            config=config,    
            SCEM_model=SCEM_model,
            IHERM_model=IHERM_model,
            dataloader=validation_loader,
            dev=dev,
            )
    scale_diff_ranges = [[1, 1.5], [1.5, 4], [4, 8]]

    results = analyze_and_save_results(
        mace_list=mace_list,
        dataset_name_list=dataset_name_list, 
        img_path_list=img_path_list,
        scale_diff_list=scale_diff_list,
        scale_diff_ranges=scale_diff_ranges,
        save_dir=args.save_dir,
        dataset=args.dataset,
        logger=logger
    )
        
   
    
