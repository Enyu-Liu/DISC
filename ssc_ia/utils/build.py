from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, OmegaConf, open_dict

from .tabular_logger import TabularLogger


def pre_build_callbacks(cfg: DictConfig):
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = 'auto'
    # with open_dict(cfg):
    #     cfg.trainer.enable_progress_bar = False

    if cfg.get('dataset'):
        cfg.data.datasets.type = cfg.dataset
    if cfg.get('data_root'):
        cfg.data.datasets.data_root = cfg.data_root
    if cfg.get('label_root'):
        cfg.data.datasets.label_root = cfg.label_root
    if cfg.get('depth_root'):
        cfg.data.datasets.depth_root = cfg.depth_root

    output_dir = 'outputs'
    name = cfg.save.get('logger_dir_name', None)
    version = cfg.save.get('logger_version_name', None)

    if not cfg.trainer.get('enable_progress_bar', True):
        logger = [TabularLogger(save_dir=output_dir, name=name, version=version)]
    else:
        logger = [loggers.TensorBoardLogger(save_dir=output_dir, name=name, version=version)]

    callback = [
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.ModelCheckpoint(
            dirpath=logger[0].log_dir,
            filename='e{epoch}_miou{val/mIoU:.4f}',
            monitor='val/mIoU',
            mode='max',
            auto_insert_metric_name=False),
        callbacks.EarlyStopping(
            monitor='val/mIoU',  # Metric to monitor (e.g., validation loss)
            min_delta=0.0001,     # Minimum change to qualify as an improvement
            patience=4,          # Number of epochs with no improvement after which training will be stopped
            verbose=True,        # Verbosity mode
            mode='max'           # Mode for the monitored metric ('min' for minimizing loss)
        ),
        callbacks.ModelSummary(max_depth=3)
    ]

    if cfg.trainer.get('enable_progress_bar', True):
        callback.append(callbacks.RichProgressBar())

    return cfg, dict(logger=logger, callbacks=callback)


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
        # cfg = OmegaConf.to_container(cfg, resolve=True) # cast to dict type
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
