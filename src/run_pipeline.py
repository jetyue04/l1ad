# src/run_pipeline.py
import os
import yaml
import importlib
from pathlib import Path

# PyTorch
import torch

# Optional: wandb setup
import wandb

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class PipelineRunner:
    def __init__(self, pipeline_cfg_path="configs/pipeline.yaml", device=None):
        self.pipeline_cfg_path = pipeline_cfg_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pipeline config
        self.cfg = load_yaml(pipeline_cfg_path)
        self.stages_config = self.cfg["pipeline"]

        # Keep track of stage instances and pretrained models
        self.stage_instances = {}
        self.pretrained_models = {}

    def _instantiate_stage(self, stage_name):
        stage_cfg = self.cfg.get(stage_name, {})
        module_name = f"src.stages.{stage_name}"
        stage_class_name = "".join([x.capitalize() for x in stage_name.split("_")])  # e.g., vicreg_stage -> VicregStage

        stage_module = importlib.import_module(module_name)
        stage_class = getattr(stage_module, stage_class_name)

        # Pass pretrained models if stage config specifies
        if "input_models" in stage_cfg:
            # Replace file paths with loaded models
            loaded_inputs = []
            for path in stage_cfg["input_models"]:
                model = torch.load(path, map_location=self.device)
                loaded_inputs.append(model)
            stage_cfg["input_models"] = loaded_inputs

        # Instantiate stage
        stage_instance = stage_class(stage_cfg)
        return stage_instance

    def run(self):
        for stage_name in self.stages_config:
            print(f"\n=== Running Stage: {stage_name} ===")

            stage_instance = self._instantiate_stage(stage_name)
            self.stage_instances[stage_name] = stage_instance

            if stage_instance.stage_cfg.get("train", True):
                # Here you need to provide dataloaders or handle them inside the stage
                train_loader = getattr(stage_instance, "train_loader", None)
                val_loader = getattr(stage_instance, "val_loader", None)

                if train_loader is None:
                    raise ValueError(f"{stage_name} does not have a train_loader defined")

                for epoch in range(stage_instance.stage_cfg.get("n_epochs", 10)):
                    print(f"\nEpoch {epoch+1}/{stage_instance.stage_cfg.get('n_epochs')}")
                    train_loss = stage_instance.train_epoch(train_loader)
                    print(f"Training loss: {train_loss:.4f}")

                    if val_loader is not None:
                        val_loss = stage_instance.validate_epoch(val_loader)
                        print(f"Validation loss: {val_loss:.4f}")

                    # Save checkpoint each epoch if desired
                    if stage_instance.stage_cfg.get("save_checkpoint", None):
                        stage_instance.save_checkpoint(name=f"epoch_{epoch+1}")

            print(f"=== Finished Stage: {stage_name} ===\n")

        print("=== Pipeline Completed ===")


if __name__ == "__main__":
    pipeline = PipelineRunner(pipeline_cfg_path="configs/pipeline.yaml")
    pipeline.run()
