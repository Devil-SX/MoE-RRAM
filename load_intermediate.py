import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_intermediates(checkpoint_dir):
    intermediates_path = checkpoint_dir / "tensors.pt"
    intermediates = torch.load(intermediates_path)
    return intermediates

def plot_distribution(values, title, filepath, weights=None):
    if type(values[0]) is torch.Tensor:
        values = [value.cpu().numpy() for value in values]
    if weights:
        if type(weights[0]) is torch.Tensor:
            weights = [weight.cpu().numpy() for weight in weights]

    plt.figure(figsize=(10, 5))
    plt.hist(values, weights=weights, bins=100, alpha=0.75, color="blue", density=True)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def write_top_tensordata(tensor_data, measure, num, filepath):
    sorted_tensors = sorted(tensor_data.items(), key=lambda x: x[1][measure], reverse=True)
    with open(filepath, 'w') as f:
        f.write(f"Top {num} tensors by {measure}:\n")
        for name, data in sorted_tensors[:num]:
            f.write(f"{name}: {data[measure]}\n")
        f.write(f"Bottom {num} tensors by {measure}:\n")
        for name, data in sorted_tensors[-num:]:
            f.write(f"{name}: {data[measure]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory")
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    fig_dir = checkpoint_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    result_dir = checkpoint_dir / "results"
    result_dir.mkdir(exist_ok=True)

    intermediates = load_intermediates(checkpoint_dir)
    stds = []
    means = []
    sizes = []
    tensor_data = {}
    for module, output in intermediates.items():
        std = output.std()
        std = (std / (output.max() - output.min())).item()
        mean = output.mean().item()
        size = output.numel()
        stds.append(std)
        means.append(mean)
        sizes.append(size)
        tensor_data[module] = {"std": std, "mean": mean, "size": size}

    plot_distribution(stds, "Distribution of Normalized Std Devs (Equal Weight)", fig_dir / "normalized_std_eq.png")
    plot_distribution(stds, "Distribution of Normalized Std Devs (Size Weighted)", fig_dir / "normalized_std_weighted.png", sizes)
    plot_distribution(means, "Distribution of Means (Equal Weight)", fig_dir / "means_eq.png")
    plot_distribution(means, "Distribution of Means (Size Weighted)", fig_dir / "means_weighted.png", sizes)

    write_top_tensordata(tensor_data, "std", 5, result_dir / "top_bottom_std.txt")
    write_top_tensordata(tensor_data, "mean", 5, result_dir / "top_bottom_mean.txt")
