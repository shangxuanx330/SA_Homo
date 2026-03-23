from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary

class ModelWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

def print_one_model_summaries(model, logger):
    """Print model summaries using PyTorch Lightning."""
    # Wrap models with Lightning Module
    pl_first_stage = ModelWrapper(model)
    pl_second_stage = ModelWrapper(model)
    
    # Generate model summaries
    logger.info("\n==== SCEM MODEL SUMMARY ====")
    summary1 = ModelSummary(pl_first_stage, max_depth=-1)
    logger.info(f"{summary1}")
    
    # Calculate and print total parameters
    total_params_first = sum(p.numel() for p in model.parameters())
    
    logger.info(f"\nTotal parameters in SCEM model: {total_params_first:,}")

   

def print_two_model_summaries(SCEM_model, IHERM_model, logger):
    """Print model summaries using PyTorch Lightning."""
    # Wrap models with Lightning Module
    pl_first_stage = ModelWrapper(SCEM_model)
    pl_second_stage = ModelWrapper(IHERM_model)
    
    # Generate model summaries
    logger.info("\n==== SCEM MODEL SUMMARY ====")
    summary1 = ModelSummary(pl_first_stage, max_depth=-1)
    logger.info(f"{summary1}")
    
    logger.info("\n==== IHERM MODEL SUMMARY ====")
    summary2 = ModelSummary(pl_second_stage, max_depth=-1)
    logger.info(f"{summary2}")
    
    # Calculate and print total parameters
    total_params_first = sum(p.numel() for p in SCEM_model.parameters())
    total_params_second = sum(p.numel() for p in IHERM_model.parameters())
    
    logger.info(f"\nTotal parameters in SCEM model: {total_params_first:,}")
    logger.info(f"Total parameters in IHERM model: {total_params_second:,}")
    logger.info(f"Total parameters in both models: {total_params_first + total_params_second:,}")