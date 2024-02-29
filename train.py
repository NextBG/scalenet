
'''
Train ScaleNet
'''

import os
import wandb
import time
import yaml
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import torch.nn.functional as F

from model import ScaleNet
from dataset import ScaleNetDataset
from utils import create_partition, count_parameters

def train(config):
    '''
    Initializations
    '''
    # WandB
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project="scale_net",
            config=config,
            )
        wandb.run.name = "ScaleNet-" + time.strftime("%y%m%d-%H%M%S")
    
    # Random seed
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Log dir YYMMDD-HHMMSS
    log_dir = os.path.join(config["log_dir"], time.strftime("%y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Partition
    train_part_file = os.path.join(config["dataset_dir"], "partitions", "train.txt")
    if not os.path.exists(train_part_file):
        create_partition(config["seed"], config["dataset_dir"], config["eval_ratio"])

    # Dataset
    train_dataset = ScaleNetDataset(config["dataset_dir"], "train", config["max_scale_factor"], config["seed"])
    eval_dataset = ScaleNetDataset(config["dataset_dir"], "eval", config["max_scale_factor"], config["seed"])

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # Model
    model = ScaleNet().to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Checkpoint
    if config["checkpoint"] != "":
        latest_path = os.path.join(config["log_dir"], config["checkpoint"], "latest.pth")
        if os.path.exists(latest_path):
            model.load_state_dict(torch.load(latest_path))
            print("Checkpoint loaded")
        else:
            print("Checkpoint not found")
            return

    # Image normalization
    norm_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Log
    if config["use_wandb"]:
        config["num_params"] = count_parameters(model)
        wandb.config.update(config)
    
    '''
    Train-eval loop
    '''
    for epoch in range(config["epochs"]):
        print(f">>> Epoch {epoch + 1}/{config['epochs']} <<<")

        # Train
        model.train()

        # Accumulate loss
        total_loss = 0.0

        # Train loop
        for data in tqdm(
            train_dataloader, 
            desc="Train batches", 
            total= (len(train_dataset) // config["batch_size"])+1,
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ):
            
            # Data
            obs_0, obs_1, scaled_trans, n_scale_gt = data

            # To GPU
            obs_0 = obs_0.to(device)
            obs_1 = obs_1.to(device)
            scaled_trans = scaled_trans.to(device)
            n_scale_gt = n_scale_gt.to(device)

            # Normalize images
            obs_0 = norm_transform(obs_0) # C, H, W
            obs_1 = norm_transform(obs_1) # C, H, W

            # Forward
            scale_pred = model(obs_0, obs_1, scaled_trans) # Size[B, 1], Float[0, 1]
            
            # Loss
            loss = F.mse_loss(scale_pred, n_scale_gt) # Size[1]
            total_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            if config["use_wandb"]:
                wandb.log({"Train/Loss": loss.mean().item()})

        # Average loss
        avg_loss = total_loss / len(train_dataset) * config["batch_size"]

        # Log
        if config["use_wandb"]:
            wandb.log({
                "Train/Learning rate": scheduler.get_last_lr()[0],
                "Train/Batch avg loss": avg_loss,
                "Epoch": epoch + 1,
                })
        print(f"Train loss: {avg_loss}")

        # Step scheduler
        scheduler.step()

        # Eval
        if len(eval_dataset) == 0:
            continue
    
        model.eval()
        with torch.no_grad():

            # Accumulate loss
            total_loss = 0.0

            # Eval loop
            for data in tqdm(
                eval_dataloader, 
                desc="Evaluation batches", 
                total= (len(eval_dataset) // config["batch_size"])+1,
                bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                ):

                # Data
                obs_0, obs_1, scaled_trans, n_scale_gt = data

                # To GPU
                obs_0 = obs_0.to(device)
                obs_1 = obs_1.to(device)
                scaled_trans = scaled_trans.to(device)
                n_scale_gt = n_scale_gt.to(device)

                # Normalize images
                obs_0 = norm_transform(obs_0)
                obs_1 = norm_transform(obs_1)

                # Forward
                scale_pred = model(obs_0, obs_1, scaled_trans) # Size[B, 1], Float[0, 1]

                # Loss
                loss = F.mse_loss(scale_pred, n_scale_gt)
                total_loss += loss.item()

                # Log
                if config["use_wandb"]:
                    wandb.log({"Eval/Loss": loss.mean().item()})

            # Average loss
            avg_loss = total_loss / len(eval_dataset) * config["batch_size"]

            # Log
            if config["use_wandb"]:
                wandb.log({
                    "Eval/Batch avg loss": avg_loss, 
                    "Epoch": epoch + 1
                    })
            print(f"Eval loss: {avg_loss}")

        # Save checkpoint
        if epoch % config["save_interval"] == 0:
            numbered_path = os.path.join(log_dir, f"{epoch}.pth")
            latest_path = os.path.join(log_dir, "latest.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)
            print(f"Checkpoint {epoch} saved")

    # Finish
    if config["use_wandb"]:
        wandb.finish()
    print("Train finished > <")

if __name__ == "__main__":
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Train-eval loop
    train(config)