import torch
from pathlib import Path
import os
def load_checkpoint_SA_Homo(args, SCEM_model, SCEM_optimizer, SCEM_scheduler,
                    IHERM_model, IHERM_optimizer, IHERM_scheduler, dev):
    """
    Load checkpoints for first and IHERM models, with optional encoder-only loading.
    
    Args:
        args: Argument object containing resume paths and loading options
        SCEM_model: SCEM model
        SCEM_optimizer: SCEM optimizer
        SCEM_scheduler: SCEM scheduler
        IHERM_model: IHERM model
        IHERM_optimizer: IHERM optimizer
        IHERM_scheduler: IHERM scheduler
        dev: Device for mapping loaded checkpoint
    
    Returns:
        epoch: Loaded epoch for SCEM (if applicable), None otherwise
    """
    def load_module_params(current_state, model_state, module_prefixes, loaded_params, skipped_params):
        """Helper function to load specific module parameters."""
        for key in current_state.keys():
            if any(key.startswith(prefix) or key.startswith(f"module.{prefix}") for prefix in module_prefixes):
                checkpoint_key = key
                fallback_key = key[7:] if key.startswith('module.') else f"module.{key}"
                
                # 尝试找到匹配的checkpoint key
                source_key = None
                source_param = None
                if checkpoint_key in model_state:
                    source_key = checkpoint_key
                    source_param = model_state[checkpoint_key]
                elif fallback_key in model_state:
                    source_key = fallback_key
                    source_param = model_state[fallback_key]
                
                if source_param is not None:
                    # 检查参数尺寸是否匹配
                    current_param = current_state[key]
                    if current_param.shape == source_param.shape:
                        current_state[key] = source_param
                        loaded_params.append(f"{source_key} -> {key}")
                    else:
                        skipped_params.append(f"{key} (shape mismatch: checkpoint {source_param.shape} vs model {current_param.shape})")
                else:
                    skipped_params.append(f"{key} (not found in checkpoint)")
                    # 跳过不匹配的encoder参数，保持当前模型的初始化值
        return current_state

    epoch = 0

    # SCEM loading
    path_checkpoint = Path(args.secm_resume) if args.secm_resume else None
    if path_checkpoint and path_checkpoint.exists():
        print(f"[0/3] Found checkpoint at {path_checkpoint}")
        try:
            checkpoint = torch.load(path_checkpoint, map_location=dev)
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            return None

        if "model_state" not in checkpoint:
            print(f"[ERROR] Checkpoint missing 'model_state' key")
            return None

        if args.SCEM_only_load_encoder:
            print("[INFO] Only loading encoder part (search_module and template_module)")
            model_state = checkpoint["model_state"]
            current_model_state = SCEM_model.state_dict()
            loaded_params, skipped_params = [], []

            current_model_state = load_module_params(
                current_model_state, model_state,
                ['search_module.', 'template_module.'],
                loaded_params, skipped_params
            )

            SCEM_model.load_state_dict(current_model_state, strict=False)
            print(f"[INFO] Successfully loaded {len(loaded_params)} encoder parameters:")
            for param in loaded_params[:5]:
                print(f"       ✓ {param}")
            if len(loaded_params) > 5:
                print(f"       ... and {len(loaded_params) - 5} more parameters")

            if skipped_params:
                print(f"[INFO] Skipped encoder parts ({len(skipped_params)} parameters):")
                for param in skipped_params[:5]:
                    print(f"       ○ {param}")
                if len(skipped_params) > 5:
                    print(f"       ... and {len(skipped_params) - 5} more parameters")
                print(f"[INFO] These encoder parameters will keep their initialized values")
            else:
                print(f"[INFO] All encoder parameters found and loaded from checkpoint")
            
            print(f"[1/3] Successfully loaded encoder-only checkpoint (skipped non-matching parts)")
            print(f"[INFO] Optimizer and scheduler states are NOT loaded in encoder-only mode")
        else:
            print("[INFO] Loading complete model with parameter matching")
            model_state = checkpoint["model_state"]
            current_model_state = SCEM_model.state_dict()
            loaded_params, skipped_params = [], []

            # 加载所有匹配的参数
            for key in current_model_state.keys():
                # 尝试多种可能的checkpoint key匹配方式
                possible_keys = [
                    key,  # 直接匹配
                    key[7:] if key.startswith('module.') else f"module.{key}",  # module前缀转换
                ]
                
                # 找到匹配的checkpoint key
                source_key = None
                source_param = None
                for candidate_key in possible_keys:
                    if candidate_key in model_state:
                        source_key = candidate_key
                        source_param = model_state[candidate_key]
                        break
                
                if source_param is not None:
                    # 检查参数尺寸是否匹配
                    current_param = current_model_state[key]
                    if current_param.shape == source_param.shape:
                        current_model_state[key] = source_param
                        loaded_params.append(f"{source_key} -> {key}")
                    else:
                        skipped_params.append(f"{key} (shape mismatch: checkpoint {source_param.shape} vs model {current_param.shape})")
                else:
                    skipped_params.append(f"{key} (not found in checkpoint)")

            SCEM_model.load_state_dict(current_model_state, strict=False)
            print(f"[INFO] Successfully loaded {len(loaded_params)} parameters:")
            for param in loaded_params[:5]:
                print(f"       ✓ {param}")
            if len(loaded_params) > 5:
                print(f"       ... and {len(loaded_params) - 5} more parameters")

            if skipped_params:
                print(f"[INFO] Skipped parameters ({len(skipped_params)} parameters):")
                for param in skipped_params[:5]:
                    print(f"       ○ {param}")
                if len(skipped_params) > 5:
                    print(f"       ... and {len(skipped_params) - 5} more parameters")
                print(f"[INFO] These parameters will keep their initialized values")
            else:
                print(f"[INFO] All parameters found and loaded from checkpoint")
            
            print(f"[1/3] Successfully loaded complete model with parameter matching")

        if not args.not_load_epoch and "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            print(f"[2/3] Loading checkpoint from epoch {epoch}")
            if "optimizer_state" in checkpoint and "scheduler_state" in checkpoint:
                SCEM_optimizer.load_state_dict(checkpoint["optimizer_state"])
                SCEM_scheduler.load_state_dict(checkpoint["scheduler_state"])
                print(f"[3/3] Successfully loaded model, optimizer, and scheduler states")
            else:
                print(f"[WARNING] Missing optimizer or scheduler state in checkpoint")
        else:
            print(f"[3/3] Successfully loaded model (optimizer and scheduler skipped)")
    else:
        print("No SCEM model checkpoint found")

    # IHERM loading
    path_checkpoint = Path(args.IHERM_resume) if args.IHERM_resume else None
    if path_checkpoint and path_checkpoint.exists():
        print(f"[1/2] Found checkpoint at {path_checkpoint}")
        try:
            checkpoint = torch.load(path_checkpoint, map_location=dev)
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            return epoch

        if "model_state" not in checkpoint:
            print(f"[ERROR] Checkpoint missing 'model_state' key")
            return epoch

        print("[INFO] Loading IHERM model with parameter matching")
        model_state = checkpoint["model_state"]
        current_model_state = IHERM_model.state_dict()
        loaded_params, skipped_params = [], []

        # 加载所有匹配的参数
        for key in current_model_state.keys():
            # 尝试多种可能的checkpoint key匹配方式
            possible_keys = [
                key,  # 直接匹配
                key[7:] if key.startswith('module.') else f"module.{key}",  # module前缀转换
            ]
            
            # 找到匹配的checkpoint key
            source_key = None
            source_param = None
            for candidate_key in possible_keys:
                if candidate_key in model_state:
                    source_key = candidate_key
                    source_param = model_state[candidate_key]
                    break
            
            if source_param is not None:
                # 检查参数尺寸是否匹配
                current_param = current_model_state[key]
                if current_param.shape == source_param.shape:
                    current_model_state[key] = source_param
                    loaded_params.append(f"{source_key} -> {key}")
                else:
                    skipped_params.append(f"{key} (shape mismatch: checkpoint {source_param.shape} vs model {current_param.shape})")
            else:
                skipped_params.append(f"{key} (not found in checkpoint)")

        IHERM_model.load_state_dict(current_model_state, strict=False)
        print(f"[INFO] Successfully loaded {len(loaded_params)} IHERM parameters:")
        for param in loaded_params[:5]:
            print(f"       ✓ {param}")
        if len(loaded_params) > 5:
            print(f"       ... and {len(loaded_params) - 5} more parameters")

        if skipped_params:
            print(f"[INFO] Skipped IHERM parameters ({len(skipped_params)} parameters):")
            for param in skipped_params[:5]:
                print(f"       ○ {param}")
            if len(skipped_params) > 5:
                print(f"       ... and {len(skipped_params) - 5} more parameters")
            print(f"[INFO] These parameters will keep their initialized values")
        else:
            print(f"[INFO] All IHERM parameters found and loaded from checkpoint")
        if not args.not_load_epoch and "optimizer_state" in checkpoint and "scheduler_state" in checkpoint:
            IHERM_optimizer.load_state_dict(checkpoint["optimizer_state"])
            IHERM_scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"[2/2] Successfully loaded model, optimizer, and scheduler states")
        else:
            print(f"[2/2] Successfully loaded model (optimizer and scheduler skipped)")
    else:
        print("No IHERM model checkpoint found")

    return epoch

def load_checkpoint_validation(model, path_checkpoint, stage_name, device):
    if not os.path.isfile(path_checkpoint):
        raise FileNotFoundError(f"[{stage_name}] Checkpoint file not found: {path_checkpoint}")

    print(f"[{stage_name}] Found checkpoint at {path_checkpoint}")
    try:
        checkpoint = torch.load(path_checkpoint, map_location=device)
    except Exception as e:
        raise RuntimeError(f"[{stage_name}] Failed to load checkpoint file: {path_checkpoint}\nError: {e}")

    if "model_state" not in checkpoint:
        raise KeyError(f"[{stage_name}] 'model_state' not found in checkpoint: {path_checkpoint}")

    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception as e:
        raise RuntimeError(f"[{stage_name}] Failed to load model state_dict from {path_checkpoint}\nError: {e}")

    print(f"[{stage_name}] Successfully loaded checkpoint.")
    return model