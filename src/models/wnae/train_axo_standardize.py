import json
import numpy as np
from torch.utils import data
import h5py
import torch
from example.trainer import TrainerWassersteinNormalizedAutoEncoder
from example.loader import Loader
from example.architectures import Encoder, Decoder
from wnae._logger import log
from pathlib import Path
import os
import shutil

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
batch_size = 1024
n_train_sample = 50000   # 50k for training
n_test_sample  = 20000   # 20k for validation

# Open HDF5 file
f = h5py.File("../training/v5/conditionsupdate_apr25.h5","r")

# Take sequential subset
train_data = f['data']["Background_data"]["Train"]["DATA"][:n_train_sample]
test_data  = f['data']["Background_data"]["Test"]["DATA"][:n_test_sample]
sig_data   = f['data']["Signal_data"]["GluGluHToBB_M-125"]["DATA"][:n_test_sample]

# Convert to torch tensors and flatten
x_train = torch.tensor(train_data.reshape(n_train_sample, -1), dtype=torch.float32, device=device)
x_test  = torch.tensor(test_data.reshape(n_test_sample, -1), dtype=torch.float32, device=device)
x_sig   = torch.tensor(sig_data.reshape(n_test_sample, -1), dtype=torch.float32, device=device)

# Calc mean and std for standardization
bias = x_train.mean(dim=0)
scale = x_train.std(dim=0)
scale[scale == 0] = 1.0  # prevent division by zero

x_train = (x_train - bias) / scale
x_test  = (x_test  - bias) / scale
x_sig   = (x_sig   - bias) / scale

# Create DataLoaders
train_loader = data.DataLoader(dataset=data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)
val_loader   = data.DataLoader(dataset=data.TensorDataset(x_test),  batch_size=batch_size, shuffle=True)
val_loader_no_batch = data.DataLoader(dataset=data.TensorDataset(x_test),  batch_size=x_test.shape[0], shuffle=True)
sig_loader   = data.DataLoader(dataset=data.TensorDataset(x_sig),   batch_size=batch_size, shuffle=True)

print(f"x_train: {x_train.shape}, x_test: {x_test.shape}, x_sig: {x_sig.shape}")

class MyLoader():
    def __init__(self, train_loader, val_loader, val_loader_no_batch, ood_loader) -> None:
        self.training_loader = train_loader
        self.validation_loader = val_loader
        self.validation_loader_no_batch = val_loader_no_batch
        self.ood_loader = ood_loader
        
loaders = MyLoader(train_loader, val_loader, val_loader_no_batch, sig_loader)

config = import_module("example.config")
config.training_params["batch_size"] = batch_size
config.training_params['n_epochs'] = 33

input_size = x_train.shape[-1]
intermediate_architecture_encoder = (28,15)
# intermediate_architecture_decoder = (24, 32, 64, 128, 57)
intermediate_architecture_decoder = (57, 128, 64, 32, 24)
bottleneck_size = 8
output_path = "output_full_10_14"
# output_path = "~/Desktop"

if os.path.exists(output_path) and os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Deleted directory: {output_path}")
else:
    print(f"Directory does not exist: {output_path}")
    
config_file = f"{output_path}/config.json"
Path(f"{output_path}/").mkdir(parents=True, exist_ok=True)
with open(config_file, "w") as file:
    json.dump(config.training_params, file, indent=4) 
    
print("Saving to ", output_path)
encoder = Encoder(
    input_size=input_size,
    intermediate_architecture=intermediate_architecture_encoder,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)
decoder = Decoder(
    output_size=input_size,
    intermediate_architecture=intermediate_architecture_decoder,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)

trainer = TrainerWassersteinNormalizedAutoEncoder(
    config=config,
    loader=loaders,
    encoder=encoder,
    decoder=decoder,
    device=device,
    output_path=output_path,
    loss_function="wnae",  # can change to "ae" or "nae"
)

trainer.train()
log.info("Saving...")
trainer.save_train_plot()
log.info("Done.")
