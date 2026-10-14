import warnings
from shutil import copy2

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.datasets.preparation import prepare_data
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="code_generation")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    if config.get("data_preparation") is not None:
        prepare_data(config.data_preparation)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    epoch_len = config.trainer.get("epoch_len")
    if "T_max" in config.lr_scheduler and config.lr_scheduler.T_max is None:
        steps_per_epoch = (
            epoch_len if epoch_len is not None else len(dataloaders["train"])
        )
        config.lr_scheduler.T_max = config.trainer.n_epochs * steps_per_epoch

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    if config.get("data_preparation") is not None:
        run_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        copy2(config.model.tokenizer_path, run_dir / "tokenizer.json")

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
