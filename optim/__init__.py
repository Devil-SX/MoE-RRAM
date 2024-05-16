from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (
    LinearLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)


def get_optimizer(cfg, parameters):
    assert hasattr(cfg, "optimizer")
    match cfg.optimizer:
        case "AdamW":
            return AdamW(parameters, lr=cfg.start_lr)
        case "Adam":
            return Adam(parameters, lr=cfg.start_lr)
        case _:
            raise NotImplementedError


def get_scheduler(cfg, optimizer):
    assert hasattr(cfg, "scheduler")
    match cfg.scheduler:
        case "LinearLR":
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=(cfg.end_lr / cfg.start_lr),
                total_iters=cfg.epoch,
            )
        case "ExponentialLR":
            gamma = (cfg.end_lr / cfg.start_lr) ** (1.0 / (cfg.epoch))
            return ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)
        case "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=cfg.end_lr)
        case "CosineAnnealingWarmRestarts":
            return CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.min_lr
            )
        case _:
            raise NotImplementedError
