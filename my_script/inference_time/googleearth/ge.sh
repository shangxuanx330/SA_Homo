export CUDA_VISIBLE_DEVICES="5,7"

save_path="results/googleearth/runtime"
echo $save_path
script_dir=$(cd "$(dirname "$0")" && pwd)
script_path="$script_dir/$(basename "$0")"
SCEM_model=ckpts/googleearth/scem.pt
IHERM_model=ckpts/googleearth/iherm.pt
config_path=configs/general_config.json

python inference/SA_Homo_visualization.py \
  --batch_size=1 \
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
  --SECM_downsample_factor 4 \
  --IHERM_downsample_factor 4 \
  --att_scales 3 5 \
  --scales 4 2 1 \
  --iter_each_scale 2 2 2   \
  --training_search_img_size 128 128 \
  --training_template_img_size 128 128  \
  --cross_attention_layers_num 2 \
  --imgs_color "rgb" \
  --cnn_norm "instancenorm" \
  --att_norm "instancenorm" \
  --cnn_activation "gelu" \
  --att_activation "gelu" \
  --kernel_fn "elu" \
  --dataset_type "GoogleEarth" \
  --attn_type MLA \
  --get_4_corner_method get_4_corner_via_usanc \
  --use_share_encoder \
  --SCEM_feature_encoder_name "Backbone_421" \
  --dual_softmax_method sinkhorn \
  --SCEM_model_name "models.networks.SCEM" \
  --IHERM_model_name "models.networks.IHERM" \
  --eval_loop_times 5 \
  --secm_resume $SCEM_model \
  --IHERM_resume $IHERM_model \
  --exe_time \
  

echo $save_path