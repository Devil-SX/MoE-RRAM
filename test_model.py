import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataloader import get_dataset
from model import get_model


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
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--checkpoint_dir",
        type=str,
        default="/home/shucheng/workspaces/MoE-rram/outputs/2024-05-05-14-10-54",
    )
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    cfg.model = "vit-moe-tiny-relu"

    model = get_model(cfg)
    state_dict = torch.load(checkpoint_dir / "checkpoint.pt")
    print(state_dict.keys())
    for name, module in model.named_modules():
        print(name)
    model.load_state_dict(state_dict)

    _, cifar100_test = get_dataset(cfg)
    test_loader = DataLoader(
        cifar100_test, batch_size=cfg.batch_size, shuffle=False, num_workers=16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
