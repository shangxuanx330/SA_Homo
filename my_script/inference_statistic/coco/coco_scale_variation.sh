
export CUDA_VISIBLE_DEVICES="5,7"

save_path="results/coco/scale_variation"
echo $save_path
script_dir=$(cd "$(dirname "$0")" && pwd)
script_path="$script_dir/$(basename "$0")"
SCEM_model=ckpts/coco/scale_variation/scem.pt
IHERM_model=ckpts/coco/scale_variation/iherm.pt
config_path=configs/general_config.json

python inference/SA_Homo_inference.py \
  --batch_size=2 \
  --sh_file_path $script_path \
  --save_dir=$save_path \
  --config=$config_path \
  --num_workers=8 \
  --SCEM_features_dim=96 \
  --SCEM_d_model=256 \
  --SCEM_num_features_predition_head 256 \
  --IHERM_num_features=96 \
  --IHERM_d_model=96 \
  --num_of_predited_pts 144 \
  --SECM_downsample_factor 16 \
  --IHERM_downsample_factor 16 \
  --att_scales 3 5 \
  --scales 16 8 4 \
  --iter_each_scale 2 2 2  \
  --training_search_img_size 768 768 \
  --training_template_img_size 768 768  \
  --cross_attention_layers_num 2 \
  --imgs_color "rgb" \
  --cnn_norm "instancenorm" \
  --att_norm "instancenorm" \
  --cnn_activation "gelu" \
  --att_activation "gelu" \
  --kernel_fn "elu" \
  --dataset_type "cv2_multi_datasets" \
  --dataset_list "datasets/coco" \
  --val_dataset_folder_name "crop_ratio_1_5_margin_0_0" \
  --attn_type MLA \
  --get_4_corner_method get_4_corner_via_usanc \
  --use_share_encoder \
  --is_val_static \
  --SCEM_feature_encoder_name "Backbone" \
  --dual_softmax_method sinkhorn \
  --secm_resume $SCEM_model \
  --IHERM_resume $IHERM_model \
  --SCEM_model_name "models.networks.SCEM" \
  --IHERM_model_name "models.networks.IHERM" \
  --eval_loop_times 5 \
  



echo $save_path