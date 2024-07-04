import graph_tool as gt
import os
# import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src.diffusion_model_beta import BetaDiffusionBinaryEdge
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


def resume_training(cfg, model_kwargs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "0"
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]
    ckpt_path = os.path.join(root_dir, cfg.general.resume)
    print(f"loading from ckpt: {ckpt_path}")

    model = BetaDiffusionBinaryEdge.load_from_checkpoint(ckpt_path, cfg=cfg, **model_kwargs)

    cfg.general.resume = ckpt_path
    cfg.general.name = cfg.general.name + '_resume'

    return cfg, model


warnings.filterwarnings("ignore", category=PossibleUserWarning)

@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):

    dataset_config = cfg["dataset"]
    if dataset_config['name'] in ['gdss-comm20', 'gdss-ego', 'gdss-grid', 'drum-sbm']:
        from src.datasets.genericGraph_dataset import GenericGraphDataModule, GenericGraphDatasetInfos
        datamodule = GenericGraphDataModule(cfg)
        dataset_infos = GenericGraphDatasetInfos(datamodule, dataset_config)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos}
    
    else:
        raise ValueError("Undefined dataset '{}'".format(dataset_config['name']))

    # fix all random operations for reproducibility
    pl.seed_everything(32)

    # instantiate model
    if cfg.general.resume is not None:
        cfg, model = resume_training(cfg, model_kwargs)
    else:
        model = BetaDiffusionBinaryEdge(cfg=cfg, **model_kwargs)

    # define checkpoints
    # Set up the ModelCheckpoint callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.general.name}",
        filename='{epoch}',
        save_top_k=-1,
        every_n_epochs=cfg.train.ckpt_callback_steps
    )
    last_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.general.name}",
        filename='last-{epoch}', 
        every_n_epochs=1
    )
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
                      gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      enable_progress_bar=False,
                      callbacks=[checkpoint_callback, last_callback],
                      log_every_n_steps=1,
                      logger = []
    )

    trainer.fit(model, datamodule=datamodule)
    # trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)

    
    

if __name__ == '__main__':
    main()