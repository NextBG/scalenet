
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
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ScaleNet
from dataset import ScaleNetDataset
from utils import create_partition, count_parameters

def train(rank, config):
    '''
    Initializations
    '''
    # DataParallel
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)

    # WandB
    if config["use_wandb"] and rank == 0:
        wandb.login()
        wandb.init(
            project="scale_net",
            config=config,
            )
        wandb.run.name = "ScaleNet-ddp-" + time.strftime("%y%m%d-%H%M%S")
    
    # Random seed
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Log dir YYMMDD-HHMMSS
    log_name = "ddp-" + time.strftime("%y%m%d-%H%M%S")
    log_dir = os.path.join("logs", log_name)
    os.makedirs(log_dir, exist_ok=True)

    # Dataset
    train_dataset = ScaleNetDataset(
        dataset_dir=config["dataset_dir"], 
        dataset_type="train", 
        max_scale_factor=config["max_scale_factor"], 
        offset_min=config["offset_min"], 
        offset_max=config["offset_max"] , 
        num_samples=config["num_samples"],
        seed=config["seed"]
    )
    eval_dataset = ScaleNetDataset(
        dataset_dir=config["dataset_dir"], 
        dataset_type="eval", 
        max_scale_factor=config["max_scale_factor"], 
        offset_min=config["offset_min"], 
        offset_max=config["offset_max"] , 
        num_samples=config["num_samples"],
        seed=config["seed"]
    )
    test_dataset = ScaleNetDataset(
        dataset_dir=config["dataset_dir"], 
        dataset_type="test", 
        max_scale_factor=config["max_scale_factor"], 
        offset_min=config["offset_min"], 
        offset_max=config["offset_max"] , 
        num_samples=config["num_samples"],
        seed=config["seed"]
    )

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        sampler=DistributedSampler(train_dataset, shuffle=True),
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=config["batch_size"], 
        sampler=DistributedSampler(eval_dataset, shuffle=True),
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        sampler=DistributedSampler(test_dataset, shuffle=True),
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )

    # Model
    model = ScaleNet(
        enc_dim=config["enc_dim"], 
        sa_layers=config["sa_layers"],
        sa_heads=config["sa_heads"],
        sa_ff_dim_factor=config["sa_ff_dim_factor"],
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Checkpoint
    if config["checkpoint"] is not None:
        latest_path = os.path.join("logs", config["checkpoint"], "latest.pth")
        if os.path.exists(latest_path):
            model.load_state_dict(torch.load(latest_path))
            if rank == 0:
                print("Checkpoint loaded")

    # Image normalization
    norm_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Log
    if config["use_wandb"] and rank == 0:
        config["num_params"] = count_parameters(model)
        wandb.config.update(config)
    
    '''
        Train
    '''
    for epoch in range(config["epochs"]):
        if rank == 0:
            print(f">>> Epoch {epoch + 1}/{config['epochs']} <<<")

        # Train
        model.train()

        # Sampler set epoch
        train_dataloader.sampler.set_epoch(epoch)

        # Accumulate loss
        total_loss = 0.0

        # Train loop
        for data in tqdm(
            train_dataloader, 
            desc="Train batches", 
            total=((len(train_dataset) // config["batch_size"])+1) // 2,
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            disable=rank!=0,
            ):
            
            # Data
            obs_0, obs_1, scaled_trans, gt_trans, n_scale_gt = data

            # To GPU
            obs_0 = obs_0.to(rank)
            obs_1 = obs_1.to(rank)
            scaled_trans = scaled_trans.to(rank)
            n_scale_gt = n_scale_gt.to(rank)

            # Normalize images
            obs_0 = norm_transform(obs_0) # C, H, W
            obs_1 = norm_transform(obs_1) # C, H, W

            # Forward
            scale_pred = model(obs_0, obs_1, scaled_trans) # Size[B, 1], Float[0, 1]
            
            # Loss
            loss = F.mse_loss(scale_pred, n_scale_gt) # Size[1]
            total_loss += loss.item() * config["batch_size"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            if config["use_wandb"] and rank == 0:
                wandb.log({"Train/Loss": loss.item()})

        # Average loss
        avg_loss = total_loss / ((len(train_dataset)+1) // 2) 

        # Log
        if rank == 0:
            print(f"Train loss: {avg_loss}")
            if config["use_wandb"]:
                wandb.log({
                    "Train/Learning rate": scheduler.get_last_lr()[0],
                    "Train/Batch avg loss": avg_loss,
                    "Epoch": epoch + 1,
                    })

        # Step scheduler
        scheduler.step()

        model.eval()

        # Sampler set epoch
        eval_dataloader.sampler.set_epoch(epoch)

        with torch.no_grad():
            '''
                Eval
            '''
            # Accumulate loss
            total_loss = 0.0

            # Eval loop
            for data in tqdm(
                eval_dataloader, 
                desc="Evaluation batches", 
                total= ((len(eval_dataset) // config["batch_size"])+1) // 2,
                bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                disable=rank!=0,
                ):

                # Data
                obs_0, obs_1, scaled_trans, gt_trans, n_scale_gt = data

                # To GPU
                obs_0 = obs_0.to(rank)
                obs_1 = obs_1.to(rank)
                scaled_trans = scaled_trans.to(rank)
                n_scale_gt = n_scale_gt.to(rank)


                # Normalize images
                obs_0 = norm_transform(obs_0)
                obs_1 = norm_transform(obs_1)

                # Forward
                scale_pred = model(obs_0, obs_1, scaled_trans) # Size[B, 1], Float[0, 1]

                # Loss
                loss = F.mse_loss(scale_pred, n_scale_gt)
                total_loss += loss.item() * config["batch_size"]

                # Log
                if config["use_wandb"] and rank == 0:
                    wandb.log({"Eval/Loss": loss.item()})

            # Average loss
            avg_loss = total_loss / ((len(eval_dataset)+1) // 2)

            # Log
            if rank == 0:
                print(f"Eval loss: {avg_loss}")
                if config["use_wandb"]:
                    wandb.log({
                        "Eval/Batch avg loss": avg_loss, 
                        "Epoch": epoch + 1
                        })
        
            '''
                Test
            '''
            # Accumulate loss
            total_loss = 0.0

            # Eval loop
            for data in tqdm(
                test_dataloader, 
                desc="Test batches", 
                total= ((len(test_dataset) // config["batch_size"])+1) // 2,
                bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                disable=rank!=0,
                ):

                # Data
                obs_0, obs_1, scaled_trans, gt_trans, n_scale_gt = data

                # To GPU
                obs_0 = obs_0.to(rank)
                obs_1 = obs_1.to(rank)
                scaled_trans = scaled_trans.to(rank)
                n_scale_gt = n_scale_gt.to(rank)


                # Normalize images
                obs_0 = norm_transform(obs_0)
                obs_1 = norm_transform(obs_1)

                # Forward
                scale_pred = model(obs_0, obs_1, scaled_trans) # Size[B, 1], Float[0, 1]

                # Loss
                loss = F.mse_loss(scale_pred, n_scale_gt)
                total_loss += loss.item() * config["batch_size"]

                # Log
                if config["use_wandb"] and rank == 0:
                    wandb.log({"Test/Loss": loss.item()})

            # Average loss
            avg_loss = total_loss / ((len(test_dataset)+1) // 2)

            # Log
            if rank == 0:
                print(f"Test loss: {avg_loss}")
                if config["use_wandb"]:
                    wandb.log({
                        "Test/Batch avg loss": avg_loss, 
                        "Epoch": epoch + 1
                        })

        # Save checkpoint
        if epoch % config["save_interval"] == 0 and rank == 0: 
            numbered_path = os.path.join(log_dir, f"{epoch}.pth")
            latest_path = os.path.join(log_dir, "latest.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)
            print(f"Checkpoint {epoch} saved")

    # Clean up
    dist.destroy_process_group()

    # Finish
    if config["use_wandb"] and rank == 0:
        wandb.finish()
    if rank == 0:
        print("Train finished > <")

if __name__ == "__main__":
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Partition
    train_part_file = os.path.join(config["dataset_dir"], "partitions", "train.txt")
    if not os.path.exists(train_part_file):
        create_partition(config["seed"], config["dataset_dir"], config["eval_ratio"])
    
    # Train-eval loop
    torch.multiprocessing.set_start_method("fork")
    mp.spawn(train, args=(config,), nprocs=2, join=True)