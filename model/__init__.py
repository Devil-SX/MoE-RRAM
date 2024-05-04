from transformers import ViTModel
from transformers.modeling_outputs import ImageClassifierOutput
from torch import nn


def get_model(cfg):
    assert hasattr(cfg, "model")
    match cfg.model:
        case "vit-tiny":
            return ViTTiny()
        case "vit-moe":
            return None
        case _:
            raise NotImplementedError


class ViTTiny(nn.Module):
    def __init__(self, num_labels=100):
        super().__init__()
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
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
