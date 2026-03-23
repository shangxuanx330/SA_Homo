import torch
from pathlib import Path
from typing import Dict, Any

def save_best_checkpoint(
    current_loss: float,
    best_loss: float,
    epoch: int,
    num_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    save_dir_checkpoint,
    save_dir_only_model,
    logger: Any
) -> float:
    """
    Save the best checkpoint if the current loss is lower than the best loss so far.

    Args:
    current_loss (float): The current validation loss
    best_loss (float): The best loss achieved so far
    epoch (int): Current epoch number
    num_epochs (int): Total number of epochs
    model (torch.nn.Module): The model to save
    optimizer (torch.optim.Optimizer): The optimizer to save
    scheduler (Any): The learning rate scheduler to save
    save_dir (Path): Directory to save the checkpoint
    logger (Any): Logger object for logging information

    Returns:
    float: The updated best loss
    """
    if current_loss < best_loss:
        best_loss = current_loss
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }

        logger.info(
            f"[{epoch + 1:3d}/{num_epochs:3d}] Saving best checkpoint to best.pt."
        )
        torch.save(checkpoint, save_dir_checkpoint)
        torch.save(model, save_dir_only_model)

    return best_loss