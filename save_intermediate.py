import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataloader import get_dataset
from model import get_model


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []


def build_tree(names: list[str]) -> TreeNode:
    root = TreeNode("")
    for name in names:
        parts = name.split('.')
        current_node = root
        for part in parts:
            found_child = None
            for child in current_node.children:
                if child.name == part:
                    found_child = child
                    break
            if found_child:
                current_node = found_child
            else:
                new_child = TreeNode(part)
                current_node.children.append(new_child)
                current_node = new_child
    return root


def get_leaf_nodes(node, prefix="") -> list[str]:
    if not node.children:
        return [prefix + node.name]
    leaf_nodes = []
    for child in node.children:
        if node.name == "": # root node
            leaf_nodes.extend(get_leaf_nodes(child, ""))
        else:
            leaf_nodes.extend(get_leaf_nodes(child, prefix + node.name + "."))
    return leaf_nodes


def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            break


intermediates = {}
module2name = {}


def hook_fn(module, input, output):
    intermediates[module2name[module]] = output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--checkpoint_dir", type=str, default="/home/shucheng/workspaces/MoE-rram/outputs/2024-05-03-22-47-19")
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")

    model = get_model(cfg)
    state_dict = torch.load(checkpoint_dir / "checkpoint.pt")
    # for name, value in state_dict.items():
    # print(name)
    model.load_state_dict(state_dict)

    _, cifar100_test = get_dataset(cfg)
    test_loader = DataLoader(cifar100_test, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    named_modules = {name: module for name, module in model.named_modules() if name != ""}
    module2name = {module: name for name, module in named_modules.items()}
    root_node = build_tree(list(named_modules.keys()))
    leaf_nodes = get_leaf_nodes(root_node)

    for module_name in leaf_nodes:
        named_modules[module_name].register_forward_hook(hook_fn)

    evaluate_model(model, test_loader, device)

    torch.save(intermediates, checkpoint_dir / "tensors.pt")
