from torch.optim.lr_scheduler import LinearLR, ExponentialLR


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
            gamma = (cfg.end_lr / cfg.start_lr) ** (1./(cfg.epoch))
            return ExponentialLR(
                optimizer,
                gamma=gamma,
                last_epoch=-1
            )
        case _:
            raise NotImplementedError
