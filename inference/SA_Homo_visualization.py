
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
from train.utils.get_pts import (
    get_grid_like_k_pts_gt_include_margin,
    get_4_pts_in_template_and_search_img,
    get_4_corner_in_search_img_by_indexes,
    get_pred_4_corner_in_search_img_by_dlt,
    get_pred_4_corner_in_search_img_by_ransac,
    transform_points_with_homography_hw,
)
from models.box_parser import get_k_pts_BoxParser
import kornia
from train.utils.get_error import get_mace
import torchvision.utils as utils
from utils.draw import draw_lines,highlight_pts
import torch.nn.functional as F
from models.common import reconstruct_from_sim_matrix
from models.homography_transformed import get_gt_mask
from utils.draw import create_checker_mixed_image
from utils.get_seeds import set_seed
from utils.loading import load_checkpoint_validation
import os
import torch
import kornia
from tqdm import tqdm
import torch.nn.functional as F
from utils.draw import create_template_replaced_image
import time

def save_img_func(pred_score_map, save_path, search_img_size, pred_offset_map, 
                  imgs_search, template_img_size, imgs_template, scale_factor,
                  idx, tl_gt, H_matrix, batch_size,
                  pred_4_corner_by_usanc, gt_4_corner, sim_img, 
                  H_pred_by_usanc, config, dev, line_width=None, mace_value=None):
    vis_dir = os.path.join(save_path, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 保存mace值到txt文件
    if mace_value is not None:
        mace_txt_path = os.path.join(vis_dir, f'{idx}_mace.txt')
        with open(mace_txt_path, 'w') as f:
            f.write(f"MACE: {mace_value:.6f}\n")
    box_parser = get_k_pts_BoxParser(stride=config["SECM_downsample_factor"])
    pts_pred = box_parser(pred_score_map, pred_offset_map)
    
    if line_width is None:
        line_width = int(imgs_search.shape[2]/1536*8)
    
    imgs_search_drawed =  draw_lines(imgs_search, gt_4_corner,"green",line_width=line_width).to(dev)
    imgs_search_drawed =  draw_lines(imgs_search_drawed, pred_4_corner_by_usanc,"red",line_width=line_width).to(dev)
    utils.save_image(imgs_search_drawed, os.path.join(vis_dir,f'{idx}_line_graph.png'))
    imgs_search_drawed =  highlight_pts(imgs_search_drawed,pts_pred,'red',line_width).to(dev)
   
    imgs_template_expand = torch.ones_like(imgs_search).to(dev)
    
    for b in range(imgs_template_expand.shape[0]):
        h,w = (search_img_size[0]-template_img_size[0])//2, (search_img_size[1]-template_img_size[1])//2
        sar_h,sar_w = template_img_size[0], template_img_size[1]
        imgs_template_expand[b,0,h:h+sar_h,w:w+sar_w] = imgs_template[b,0,:,:]

    gt_mask  =  get_gt_mask(template_img_size=template_img_size,
                            search_img_size = search_img_size,
                            gt_tl= tl_gt,
                            H_matrix = H_matrix, 
                            scale_factor_4_sub_img = 1,
                            stride = 1,
                            device=dev)
    
    search_2_template = kornia.geometry.transform.warp_perspective(imgs_search, H_pred_by_usanc, template_img_size, padding_mode='zeros')

    # 创建棋盘状混合图像
    mixed_image = create_checker_mixed_image(imgs_search, imgs_template, H_pred_by_usanc, dev, checker_size=64)
    replaced_image = create_template_replaced_image(imgs_search, imgs_template, H_pred_by_usanc, dev)

    combine = torch.cat([imgs_search_drawed, gt_mask.expand(-1, 3, -1, -1),imgs_template_expand.expand(-1, 3, -1, -1),search_2_template.expand(-1, 3, -1, -1), mixed_image.expand(-1, 3, -1, -1)],dim=0)
    combined_save_path = os.path.join(vis_dir,f'{idx}.png')
    utils.save_image(combine,combined_save_path,nrow=batch_size, padding=20,pad_value = 0.7)
    utils.save_image(imgs_search, os.path.join(vis_dir,f'{idx}_search.png'))
    imgs_template1 = F.interpolate(imgs_template, size = (template_img_size[0] // scale_factor, template_img_size[1] // scale_factor))
    utils.save_image(imgs_template1, os.path.join(vis_dir,f'{idx}_template.png'))
    utils.save_image(search_2_template, os.path.join(vis_dir,f'{idx}_search_2_template.png'))
    
    # 单独保存棋盘状混合图像
    utils.save_image(mixed_image.expand(-1, 3, -1, -1), os.path.join(vis_dir,f'{idx}_checker_mixed.png'))
    utils.save_image(replaced_image.expand(-1, 3, -1, -1), os.path.join(vis_dir,f'{idx}_template_replaced.png'))

def visualization(
    args,
    config,
    SCEM_model,
    IHERM_model,
    dataloader: DataLoader,
    dev: Device,
):
    batch_size = args.batch_size
    scale_factor = args.scale_factor
    total_iter = sum(args.iter_each_scale)
    template_size = config['training_template_img_size']
    search_size = config['training_search_img_size']
    k = config['num_of_predited_pts']
    stride = config['SECM_downsample_factor']
    idx = 0
    with torch.no_grad():
        for imgs_search, imgs_template, tl_gt, H_matrix,search_scale, scale_diff ,dataset_type, search_path, template_path in tqdm(
            dataloader,
            total=len(dataloader),
            unit="batch",
            desc="[ val ]",
            colour="blue",
        ):
            four_corner_in_template_corresponding_pts_in_search_img_gt, four_corner_in_template = get_4_pts_in_template_and_search_img(tl_gt,template_size,H_matrix,0,scale_factor=scale_factor)
            four_corner_in_template_corresponding_pts_in_search_img_gt = four_corner_in_template_corresponding_pts_in_search_img_gt.to(dev, non_blocking=True)
            four_corner_in_template = four_corner_in_template.to(dev, non_blocking=True)
            search_scale = search_scale.to(dev, non_blocking=True).unsqueeze(dim=1)
            scale_diff = scale_diff.to(dev, non_blocking=True).unsqueeze(dim=1)

            # 如果设置了metric_no_scale，则将search_scale和scale_diff设置为1
            if args.metric_no_scale:
                search_scale = torch.ones_like(search_scale)
                scale_diff = torch.ones_like(scale_diff)

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
           
            H_pred_by_usanc_search_2_template = kornia.geometry.transform.get_perspective_transform(
                            four_corner_pred_by_second_stage[:, :, [1, 0]], four_corner_in_template[:, :, [1, 0]],
                        ).detach()

            sim_img = reconstruct_from_sim_matrix(
                                                    pred_sim_matrix,
                                                    (int(search_size[0] / stride), int(search_size[1] / stride)),
                                                    (int(template_size[0] / stride), int(template_size[1] / stride))
                                                )
            
            save_img_func(
                    pred_score_map=pred_score_map,
                    save_path=args.save_dir,
                    search_img_size=search_size,
                    pred_offset_map=pred_offset_map,
                    imgs_search=imgs_search*0.5+0.5,
                    imgs_template=imgs_template*0.5+0.5,
                    template_img_size=template_size,
                    scale_factor=scale_factor,
                    idx=idx,
                    tl_gt=tl_gt,
                    H_matrix=H_matrix,
                    batch_size=batch_size,
                    pred_4_corner_by_usanc=four_corner_pred_by_second_stage,
                    gt_4_corner=four_corner_in_template_corresponding_pts_in_search_img_gt,
                    sim_img=sim_img,
                    H_pred_by_usanc=H_pred_by_usanc_search_2_template,
                    config=config,
                    dev=dev,
                    line_width=getattr(args, 'line_width', None),
                    mace_value=mace_second_stage.mean().item()
                )
            idx += 1
            
def cal_time(
    args,
    config,    
    SCEM_model,
    IHERM_model,
    dataloader: DataLoader,
    dev: Device
):

    k = config['num_of_predited_pts']
    scale_factor = 1
    SCEM_model.eval()
    IHERM_model.eval()
    template_size = config['training_template_img_size']
   

    # 时间统计列表
    SCEM_times = []
    ransac_times = []
    IHERM_times = []
    feature_extract_times = []
    warp_times = []
    total_times = []

    for _ in range(args.eval_loop_times):
        with torch.no_grad():
            for imgs_search, imgs_template, tl_gt, H_matrix,search_scale, scale_diff ,dataset_type, search_path, template_path in tqdm(
                dataloader,
                total=len(dataloader),
                unit="batch",
                desc="[ 测量推理时间 ]",
                colour="green",
            ):
                # 数据预处理（不计入推理时间）
                four_corner_in_template_corresponding_pts_in_search_img_gt, four_corner_in_template = get_4_pts_in_template_and_search_img(tl_gt,template_size,H_matrix,0,scale_factor=scale_factor)
                four_corner_in_template_corresponding_pts_in_search_img_gt = four_corner_in_template_corresponding_pts_in_search_img_gt.to(dev, non_blocking=True)
                four_corner_in_template = four_corner_in_template.to(dev, non_blocking=True)
                search_scale = search_scale.to(dev, non_blocking=True).unsqueeze(dim=1)
                scale_diff = scale_diff.to(dev, non_blocking=True).unsqueeze(dim=1)


                if args.metric_no_scale:
                    search_scale = torch.ones_like(search_scale)
                    scale_diff = torch.ones_like(scale_diff)

                key_pts_gt_in_search_img, key_pts_gt_in_template_img = get_grid_like_k_pts_gt_include_margin(tl_gt,template_size,H_matrix,k)
                key_pts_gt_in_search_img = key_pts_gt_in_search_img.to(dev, non_blocking=True)
                key_pts_gt_in_template_img = key_pts_gt_in_template_img.to(dev, non_blocking=True)

                imgs_search = imgs_search.to(dev, non_blocking=True)
                imgs_template = imgs_template.to(dev, non_blocking=True)
                tl_gt = tl_gt.to(dev, non_blocking=True)
                H_matrix = H_matrix.to(dev, non_blocking=True)

                total_start = time.time()
                # =============  1. 第一阶段推理时间测量 =============
                SCEM_start = time.time()
            
                pred_sim_matrix,pred_score_map, pred_offset_map,x_search,x_template = SCEM_model(imgs_search, imgs_template)
        
                SCEM_end = time.time()
                SCEM_time = SCEM_end - SCEM_start
                SCEM_times.append(SCEM_time)

                # =============  2. RANSAC获得H矩阵时间测量 =============
                ransac_start = time.time()
                
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
            
                ransac_end = time.time()
                ransac_time = ransac_end - ransac_start
                ransac_times.append(ransac_time)

                # =============  3. 第二阶段推理时间测量 =============
                warp_start = time.time()
                imgs_search_warpped = kornia.geometry.transform.warp_perspective(imgs_search, H_SCEM_pred_search_2_template, \
                                                                            imgs_template.shape[-2:], padding_mode='border', \
                                                                            align_corners=True,  mode='bilinear')
                warp_end = time.time()
                warp_time = warp_end - warp_start
                warp_times.append(warp_time)
                
                init_four_disp = torch.zeros_like(four_corner_pred_by_first_stage)
                
                IHERM_start = time.time()
                
              
                with torch.no_grad():
                    feature_extract_time_start = time.time()
                    _, x_search = SCEM_model.module.search_module(imgs_search_warpped)
                    feature_extract_time_end = time.time()
                    feature_extract_time = feature_extract_time_end - feature_extract_time_start
                    feature_extract_times.append(feature_extract_time)

                x_search, x_template = [x_search[i].detach() for i in range(len(x_search))], [x_template[i].detach() for i in range(len(x_template))]
                pred_delta = IHERM_model(x_search, x_template, init_four_disp)
               
                
                IHERM_end = time.time()
                IHERM_time = IHERM_end - IHERM_start - feature_extract_time     
                IHERM_times.append(IHERM_time)

                # 记录总时间结束
                total_end = time.time()
                total_time = total_end - total_start
                total_times.append(total_time)
            
    print("\n" + "="*50)
    print("Inference Time Statistics:")
    print("="*50)
    print(f"Number of Test Samples: {len(SCEM_times)}")
    print(f"SCEM Average Time: {sum(SCEM_times)/len(SCEM_times)*1000:.2f} ms")
    print(f"RANSAC Average Time: {sum(ransac_times)/len(ransac_times)*1000:.2f} ms")  
    print(f"IHERM Average Time: {sum(IHERM_times)/len(IHERM_times)*1000:.2f} ms")
    print(f"Total Inference Time (including preprocessing): {sum(total_times)/len(total_times)*1000:.2f} ms")
    print(f"Core Inference Time (sum of three stages): {(sum(SCEM_times)+sum(ransac_times)+sum(IHERM_times))/len(SCEM_times)*1000:.2f} ms")
    
    # Save time statistics to file
    time_stats_path = os.path.join(args.save_dir, 'timing_statistics.txt')
    with open(time_stats_path, 'w') as f:
        f.write("Inference Time Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Number of Test Samples: {len(SCEM_times)}\n")
        f.write(f"SCEM Average Time: {sum(SCEM_times)/len(SCEM_times)*1000:.2f} ms\n")
        f.write(f"RANSAC Average Time: {sum(ransac_times)/len(ransac_times)*1000:.2f} ms\n")
        f.write(f"Warp Average Time: {sum(warp_times)/len(warp_times)*1000:.2f} ms\n")
        f.write(f"Feature Extract Average Time: {sum(feature_extract_times)/len(feature_extract_times)*1000:.2f} ms\n")
        f.write(f"IHERM Average Time: {sum(IHERM_times)/len(IHERM_times)*1000:.2f} ms\n")
        f.write(f"Total Inference Time : {sum(total_times)/len(total_times)*1000:.2f} ms\n")
        f.write("\nDetailed Time List:\n")
        f.write("LM Times (ms): " + ", ".join([f"{t*1000:.2f}" for t in SCEM_times]) + "\n")
        f.write("RANSAC Times (ms): " + ", ".join([f"{t*1000:.2f}" for t in ransac_times]) + "\n")
        f.write("IHERM Times (ms): " + ", ".join([f"{t*1000:.2f}" for t in IHERM_times]) + "\n")
    
    print(f"Time statistics saved to: {time_stats_path}")
    print("="*50)

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

    validation_loader,  _ = get_val_dataloder(config,args,'test')

    return validation_loader
    # return validation_loader

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
        
    if args.exe_visualization:
        
        args.batch_size = 1
        validation_loader = load_dataset(args, config)
           
        visualization(
                    args=args,
                    config=config,    
                    SCEM_model=SCEM_model,
                    IHERM_model=IHERM_model,
                    dataloader=validation_loader,
                    dev=dev,
                )

    if args.exe_time:
        
        args.batch_size = 1
        validation_loader = load_dataset(args, config)

        cal_time(
            args=args,
            config=config,    
            SCEM_model=SCEM_model,
            IHERM_model=IHERM_model,
            dataloader=validation_loader,
            dev=dev,
        )

   
    
