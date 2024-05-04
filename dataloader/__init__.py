import PIL.Image
from datasets import load_dataset
from torchvision.transforms import (Compose, Normalize, RandomResizedCrop,
                                    Resize, ToTensor)
from transformers import ViTImageProcessor


def get_dataset(cfg):
    processor = ViTImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")

    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    resize = Resize(size, interpolation=PIL.Image.BICUBIC)
    _transforms_train = Compose(
        [
            resize,
            RandomResizedCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    _transforms_test = Compose([resize, ToTensor(), normalize])

    def transforms_train(examples):
        examples["pixel_values"] = [
            _transforms_train(img.convert("RGB")) for img in examples["img"]
        ]
        del examples["img"]
        examples["labels"] = [label for label in examples["fine_label"]]
        del examples["fine_label"]
        del examples["coarse_label"]
        return examples

    def transforms_test(examples):
        examples["pixel_values"] = [
            _transforms_test(img.convert("RGB")) for img in examples["img"]
        ]
        del examples["img"]
        examples["labels"] = [label for label in examples["fine_label"]]
        del examples["fine_label"]
        del examples["coarse_label"]
        return examples

    cifar100_train = load_dataset("cifar100", split="train").with_transform(
        transforms_train
    )
    # cifar100.set_format(type="torch", columns=["img", "fine_label"])
    cifar100_test = load_dataset("cifar100", split="test").with_transform(
        transforms_test
    )

    return cifar100_train, cifar100_test
