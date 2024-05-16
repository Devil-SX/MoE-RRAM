from transformers import ViTModel, ViTConfig
from transformers.modeling_outputs import ImageClassifierOutput
from torch import nn
from .moe import MoEModel
from pathlib import Path
import re
import torch
import copy

def get_model(cfg):
    assert hasattr(cfg, "model")
    match cfg.model:
        case "vit-tiny":
            return ViTTiny()
        case "vit-tiny-relu":
            return ViTTiny(is_relu=True)
        case "vit-moe-tiny":
            return MoETiny().from_pretrained(Path(cfg.resume_dir))
        case "vit-moe-tiny-relu":
            return MoETiny(is_relu=True).from_pretrained(Path(cfg.resume_dir))
        case _:
            raise NotImplementedError


class ViTTiny(nn.Module):
    def __init__(self, num_labels=100, is_relu=False):
        super().__init__()
        config = ViTConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        if is_relu:
            config.hidden_act = "relu"
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224", config=config)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None) -> ImageClassifierOutput:
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(
            outputs.last_hidden_state[:, 0, :]
        )  # Use the [CLS] token
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )


class MoETiny(nn.Module):
    def __init__(self, num_labels=100, is_relu:bool=False):
        """
        is_relu: True (ReLU) or False (GeLU)
        """
        super().__init__()
        config = ViTConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        if is_relu:
            config.hidden_act = "relu"
        self.vit = MoEModel(config)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None) -> ImageClassifierOutput:
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(
            outputs.last_hidden_state[:, 0, :]
        )  # Use the [CLS] token
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )

    def from_pretrained(self, resume_dir:Path):
        state_dict = torch.load(resume_dir / "checkpoint.pt")

        pattern = re.compile(r"vit\.encoder\.layer\.(\d+)\.(output|intermediate)\.dense\.(weight|bias)")

        new_dict = copy.deepcopy(state_dict)
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match:
                layer_number, module_name, para_name = match.groups()
                new_key_0 = f"vit.encoder.layer.{layer_number}.moe_ffn.{module_name}.0.dense.{para_name}"
                new_key_1 = f"vit.encoder.layer.{layer_number}.moe_ffn.{module_name}.1.dense.{para_name}"
                new_dict[new_key_0] = value
                new_dict[new_key_1] = value
                del new_dict[key]

        self.load_state_dict(new_dict, strict=False)
        return self

    def get_aux_loss(self):
        aux_loss = 0
        for name, module in self.named_modules():
            if hasattr(module, "aux_loss"):
                aux_loss += module.aux_loss
        return aux_loss