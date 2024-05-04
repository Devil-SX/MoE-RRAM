import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from dataloader import get_dataset
from model import get_model
from optim import get_scheduler
from utils import set_everything


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    model.train()
    print(f"Total epochs: {num_epochs}")
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                wandb.log({"Train Loss": loss.item()})

        last_lr_list = scheduler.get_last_lr()
        for i, lr in enumerate(last_lr_list):
            wandb.log({"lr_" + str(i): lr})
        scheduler.step()
        evaluate_model(model, val_loader, device)


def evaluate_model(model, val_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    wandb.log({"Test Acc": accuracy})


def init_exp(cfg: DictConfig) -> Path:
    set_everything(cfg.seed)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        project="MoE-RRAM",
        group=cfg.model,
        name=current_time,
        config=OmegaConf.to_container(cfg),
        save_code=True,
    )
    output_dir = Path("outputs") / current_time
    output_dir.mkdir(parents=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="cfg/finetune.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    out_dir = init_exp(cfg)

    model = get_model(cfg)
    cifar100_train, cifar100_test = get_dataset(cfg)

    train_loader = DataLoader(cifar100_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(cifar100_test, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.start_lr)
    scheduler = get_scheduler(cfg, optimizer)

    train(model, train_loader, test_loader, optimizer, scheduler, cfg.epoch, device)

    torch.save(model.state_dict(), out_dir / "checkpoint.pt")
