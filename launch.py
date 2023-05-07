import os
import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    # group.add_argument("--export", action="store_true") # TODO: a separate export function

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpus = len(args.gpu.split(","))

    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

    # from typeguard import install_import_hook
    # install_import_hook(['threestudio'])
    import threestudio

    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.callbacks import CustomProgressBar
    from threestudio.utils.callbacks import ConfigSnapshotCallback, CodeSnapshotCallback

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras)

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    pl.seed_everything(cfg.seed)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system = threestudio.find(cfg.system_type)(cfg.system)
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CustomProgressBar(refresh_rate=1),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),  # TODO: better config saving
        ]

    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ]

    trainer = Trainer(
        callbacks=callbacks, logger=loggers, inference_mode=False, **cfg.trainer
    )

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
