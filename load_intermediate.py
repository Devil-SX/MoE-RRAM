import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_intermediates(checkpoint_dir):
    intermediates_path = checkpoint_dir / "tensors.pt"
    intermediates = torch.load(intermediates_path)
    return intermediates


def plot_exceedance(tensor_ids, exceed_3_pct, exceed_6_pct, std, filepath):
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    std_norm = std
    # std_norm = [(val / max(std)) * 100 for val in std]

    tensor_ids = np.array(tensor_ids)
    exceed_3_pct = np.array(exceed_3_pct)
    exceed_6_pct = np.array(exceed_6_pct)
    sorted_indices = np.argsort(tensor_ids)
    tensor_ids = tensor_ids[sorted_indices]
    exceed_3_pct = exceed_3_pct[sorted_indices]
    exceed_6_pct = exceed_6_pct[sorted_indices]
    ax1.fill_between(tensor_ids, exceed_3_pct, label=">3%", color="mediumaquamarine")
    ax1.fill_between(tensor_ids, exceed_6_pct, label=">6%", color="gold")

    # ax1.bar(tensor_ids, exceed_3_pct, label=">3%", color="green")
    # ax1.bar(tensor_ids, exceed_6_pct, label=">6%", color="orange")

    ax1.set_xlabel("Tensor ID by Max σ Order")
    ax1.set_ylabel("Percentage")
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.90))
    
    ax2 = ax1.twinx()
    ax2.plot(std_norm, label="Max σ", color="dodgerblue")
    ax2.set_ylabel("Normlized Maximum Value(σ)")
    
    ax2.annotate(f'{max(std):.2f}', xy=(0, std_norm[0]), textcoords="offset points", xytext=(0,10), ha='center')
    ax2.legend(loc='upper right')
    
    ax1.set_title("Tiny-MoE on CIFAR-100")
    ax1.set_ylim(0, max(exceed_3_pct)*1.3)
    ax2.set_ylim(0, max(std)*1.3)

    # fig.legend(loc='upper right')
    plt.savefig(filepath)
    plt.close()


def plot_distribution(values, title, xlabel, filepath, weights=None):
    if type(values[0]) is torch.Tensor:
        values = [value.cpu().numpy() for value in values]
    if weights:
        if type(weights[0]) is torch.Tensor:
            weights = [weight.cpu().numpy() for weight in weights]

    plt.figure(figsize=(10, 5))
    plt.hist(values, weights=weights, bins=100, alpha=0.75, color="dodgerblue", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    # plt.grid(True)
    plt.savefig(filepath)
    plt.close()


def write_top_tensordata(tensor_data, measure, num, filepath):
    sorted_tensors = sorted(
        tensor_data.items(), key=lambda x: x[1][measure], reverse=True
    )
    with open(filepath, "w") as f:
        f.write(f"Top {num} tensors by {measure}:\n")
        for name, data in sorted_tensors[:num]:
            f.write(f"{name}: {data[measure]}\n")
        f.write(f"Bottom {num} tensors by {measure}:\n")
        for name, data in sorted_tensors[-num:]:
            f.write(f"{name}: {data[measure]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory",
        default=".",
    )
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    fig_dir = checkpoint_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    result_dir = checkpoint_dir / "results"
    result_dir.mkdir(exist_ok=True)

    plt.rcParams['font.size'] = 14

    intermediates = load_intermediates(checkpoint_dir)
    stds = []
    means = []
    sizes = []
    tensor_data = {}
    for module, output in intermediates.items():
        std = output.std().item()
        mean = output.mean().item()
        max_value = output.max().item()
        min_value = output.min().item()
        size = output.numel()
        standardized = (output - mean) / std
        abs_vals = torch.abs(standardized)
        exceed_3 = (abs_vals > 3).float().mean().item() * 100
        exceed_6 = (abs_vals > 6).float().mean().item() * 100
        stds.append(std)
        means.append(mean)
        sizes.append(size)

        tensor_data[module] = {
            "std": std,
            "mean": mean,
            "size": size,
            "exceed_3": exceed_3,
            "exceed_6": exceed_6,
            "max": max_value,
            "min": min_value,
        }

    tensor_id = [
        index
        for index, value in sorted(
            enumerate(tensor_data.values()), key=lambda x: x[1]["std"], reverse=True
        )
    ]
    exceed_3s = [data["exceed_3"] for data in tensor_data.values()]
    exceed_6s = [data["exceed_6"] for data in tensor_data.values()]
    sort_stds = [
        data["std"]
        for data in sorted(tensor_data.values(), key=lambda x: x["std"], reverse=True)
    ]

    plot_distribution(
        stds,
        "Distribution of Normalized σ (Equal Weight)",
        "Standard Deviation",
        fig_dir / "normalized_std_eq.pdf",
    )
    plot_distribution(
        stds,
        "Distribution of Normalized σ (Size Weighted)",
        "Standard Deviation",
        fig_dir / "normalized_std_weighted.pdf",
        sizes,
    )
    plot_distribution(
        means, "Distribution of μ (Equal Weight)", 
        "Mean",
        fig_dir / "means_eq.pdf"
    )
    plot_distribution(
        means,
        "Distribution of μ (Size Weighted)",
        "Mean",
        fig_dir / "means_weighted.pdf",
        sizes,
    )
    plot_exceedance(
        tensor_id, exceed_3s, exceed_6s, sort_stds, fig_dir / "exceedance.pdf"
    )

    write_top_tensordata(tensor_data, "std", 20, result_dir / "top_bottom_std.txt")
    write_top_tensordata(tensor_data, "mean", 20, result_dir / "top_bottom_mean.txt")
    write_top_tensordata(tensor_data, "max", 20, result_dir / "top_bottom_max.txt")   
    write_top_tensordata(tensor_data, "min", 20, result_dir / "top_bottom_min.txt")   
