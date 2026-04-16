import torch
import os

class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, device=None):
        super().__init__()
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x):
        """Override in subclass"""
        raise NotImplementedError("Forward method must be implemented in subclass")

    def train_step(self, batch):
        """
        Single training step.
        Override if custom training step is needed.
        Returns: loss, dict of metrics
        """
        raise NotImplementedError("train_step must be implemented in subclass")
    
    def save(self, path: str, state_dict_only=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if state_dict_only:
            torch.save({"model_state_dict": self.state_dict()}, path)
        else:
            torch.save(self, path)
        print(f"Saved model checkpoint at {path}")
    
    def load(self, path: str, state_dict_only=True):
        if state_dict_only:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            loaded_model = torch.load(path, map_location=self.device)
            self.load_state_dict(loaded_model.state_dict())
        print(f"Loaded model from {path}")
