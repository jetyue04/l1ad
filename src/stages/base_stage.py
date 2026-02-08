# src/stages/base_stage.py
import os
import yaml
import torch
from tqdm import tqdm
import wandb

class BaseStage:
    def __init__(self, stage_name, stage_cfg, model_cls, model_cfg_path, device=None):
        """
        stage_name: str, name of stage
        stage_cfg: dict, stage-specific training config (from pipeline.yaml)
        model_cls: class, subclass of BaseModel
        model_cfg_path: str, path to model YAML config
        """
        self.stage_name = stage_name
        self.stage_cfg = stage_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model config
        with open(model_cfg_path) as f:
            self.model_cfg = yaml.safe_load(f)
        
        # Build model
        self.model = model_cls(**self.model_cfg)
        self.model.to(self.device)

        # Setup optimizer
        optimizer_cls = getattr(torch.optim, stage_cfg.get("optimizer", "Adam"))
        self.optimizer = optimizer_cls(self.model.parameters(), lr=stage_cfg.get("lr", 1e-4))
        
        # Optional LR scheduler
        if "lr_scheduler" in stage_cfg and stage_cfg["lr_scheduler"]:
            scheduler_cls = getattr(torch.optim.lr_scheduler, stage_cfg["lr_scheduler"])
            self.lr_scheduler = scheduler_cls(self.optimizer, **stage_cfg.get("lr_scheduler_args", {}))
        else:
            self.lr_scheduler = None
        
        # WandB integration
        self.use_wandb = stage_cfg.get("wandb", None) is not None
        if self.use_wandb:
            wandb.init(project=stage_cfg["wandb"]["project"], name=self.stage_name)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        for batch in tqdm(dataloader, desc=f"Training {self.stage_name}"):
            self.optimizer.zero_grad()
            x = batch[0].to(self.device)
            loss, metrics = self.model.train_step(x)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if self.use_wandb and (n_batches % self.stage_cfg["wandb"].get("log_every_n_steps", 50) == 0):
                wandb.log(metrics)
        
        avg_loss = total_loss / max(1, n_batches)
        return avg_loss
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        n_batches = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validation {self.stage_name}"):
                x = batch[0].to(self.device)
                loss, metrics = self.model.train_step(x)  # or model.eval_step if you have it
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)
    
    def save_checkpoint(self, name="last_epoch"):
        path = os.path.join(self.stage_cfg.get("save_checkpoint", "checkpoints"), f"{name}.pt")
        self.model.save(path)
    
    def load_checkpoint(self, path):
        self.model.load(path)
