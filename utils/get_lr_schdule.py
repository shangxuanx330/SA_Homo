import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, StepLR

def get_scheduler(optimizer, args, stage='first_stage'):
    """
    根据args配置和训练阶段返回学习率调度器（按epoch调度）
    
    Args:
        optimizer: PyTorch优化器
        args: 命令行参数
        stage: 训练阶段标识 ('first_stage' 或 'second_stage')
    
    Returns:
        scheduler: 学习率调度器对象（或None）
    """
    # 获取对应阶段的参数（带前缀）
    lr_schedule_type = getattr(args, f'{stage}_lr_schedule_type')
    
    if lr_schedule_type == 'warmup_cosine':
        warmup_epochs = getattr(args, f'{stage}_warmup_epochs')
        cosine_epochs = args.epochs - warmup_epochs  # 使用统一的args.epochs
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=getattr(args, f'{stage}_warmup_start_factor'),
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=getattr(args, f'{stage}_eta_min')
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
    elif lr_schedule_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=getattr(args, f'{stage}_lr_decrease_period'),  
            gamma=getattr(args, f'{stage}_lr_gamma')
        )
        
    elif lr_schedule_type == 'none':
        # 返回一个恒定学习率的调度器（LambdaLR with lambda=1.0）
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0
        )
        
    else:
        raise ValueError(f"未知的{stage} lr_schedule_type: {lr_schedule_type}")
    
    return scheduler
