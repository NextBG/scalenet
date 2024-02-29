
'''
Train ScaleNet
'''

import os
import yaml
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import ScaleNet
from dataset import ScaleNetDataset

EVAL_CHECKPOINT = "latest"

INF_DIR = "inference"

def inference(config):
    '''
    Initializations
    '''
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ScaleNet(
        enc_dim=config["enc_dim"], 
        sa_layers=config["sa_layers"],
        sa_heads=config["sa_heads"],
        sa_ff_dim_factor=config["sa_ff_dim_factor"],
    ).to(device)

    # Checkpoint
    latest_path = os.path.join(config["log_dir"], EVAL_CHECKPOINT, "latest.pth")
    model.load_state_dict(torch.load(latest_path))
    print("Checkpoint loaded")

    # Image normalization
    norm_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()

    with torch.no_grad():

        # Eval loop
        for data in tqdm(
            eval_dataloader, 
            desc="Evaluation batches", 
            total= ((len(eval_dataset))+1) // 2,
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ):

            # Data
            obs_0, obs_1, scaled_trans = data

            # To GPU
            obs_0 = obs_0.to(device)
            obs_1 = obs_1.to(device)
            scaled_trans = scaled_trans.to(device)

            # Normalize images
            obs_0 = norm_transform(obs_0)
            obs_1 = norm_transform(obs_1)

            # Forward
            scale_pred = model(obs_0, obs_1, scaled_trans) # Size[B, 1], Float[0, 1]



if __name__ == "__main__":
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Inference
    inference(config)