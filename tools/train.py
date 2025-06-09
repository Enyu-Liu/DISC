import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import warnings
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf


from ssc_ia import LitModule, build_data_loaders, pre_build_callbacks, SetSeed


@hydra.main(config_path='../configs', config_name='config_sema', version_base=None)  # my_config
def main(cfg: DictConfig):
    # torch.backends.cudnn.benchmark = False 
    # torch.backends.cudnn.deterministic = True
    
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    
    SetSeed(seed=42)
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, **meta_info)
    trainer = L.Trainer(**cfg.trainer, **callbacks)
    ckpt_path = None if cfg.resume.ckpt_path=='None' else cfg.resume.ckpt_path
    trainer.fit(model, *dls[:2], ckpt_path=ckpt_path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, message=".*__floordiv__ is deprecated.*")
    main()
