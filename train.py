import warnings
from pathlib import Path
from shutil import copy2

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from src.datasets.data_utils import get_dataloaders
from src.datasets.preparation import prepare_data
from src.datasets.tokenizer import tokenizer_digest
from src.trainer import Trainer
from src.utils.checkpoint import load_resume_checkpoint
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
    if config.trainer.get("resume_from") and config.trainer.get("from_pretrained"):
        raise ValueError("Choose resume_from or from_pretrained, not both.")
    checkpoint = load_resume_checkpoint(config, ROOT_PATH)
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
        total_steps = config.trainer.n_epochs * steps_per_epoch
        if checkpoint is not None and config.trainer.get("reset_scheduler", False):
            completed = checkpoint["epoch"] - int(
                not checkpoint.get("epoch_complete", True)
            )
            total_steps -= completed * steps_per_epoch + (
                checkpoint.get("epoch_step", 0)
                if not checkpoint.get("epoch_complete", True)
                else 0
            )
        config.lr_scheduler.T_max = total_steps

    logger = setup_saving_and_logging(config)
    project_config = OmegaConf.to_container(config, resolve=True)
    run_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    if config.model.get("tokenizer_path") is not None:
        source = config.model.tokenizer_path
        if checkpoint is not None and checkpoint.get("tokenizer_sha256"):
            if tokenizer_digest(source) != checkpoint["tokenizer_sha256"]:
                raise ValueError(
                    "Checkpoint and prepared data use different tokenizers."
                )
        if (run_dir / "tokenizer.json").resolve() != Path(source).resolve():
            copy2(source, run_dir / "tokenizer.json")
    validation = getattr(dataloaders.get("val", None), "dataset", None)
    validation = getattr(validation, "data", None)
    if hasattr(validation, "save_to_disk") and not (run_dir / "validation").exists():
        validation.save_to_disk(str(run_dir / "validation"))
    writer = instantiate(config.writer, logger, project_config)

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
        resume_checkpoint=checkpoint,
    )
    del checkpoint  # release the CPU copy after state has been restored
    # Resume-only actions apply once; the saved config is ready for the next session.
    with open_dict(config.trainer):
        config.trainer.reset_scheduler = False
        config.trainer.reset_data_state = False
        if config.trainer.get("from_pretrained") is not None:
            config.trainer.from_pretrained = None
    OmegaConf.save(config, run_dir / "config.yaml")

    try:
        trainer.train()
    finally:
        finish = getattr(writer, "finish", None)
        if finish is not None:
            finish()


if __name__ == "__main__":
    main()
