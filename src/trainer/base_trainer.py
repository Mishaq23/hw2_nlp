from abc import abstractmethod
from itertools import islice

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.datasets.tokenizer import tokenizer_digest
from src.metrics.tracker import MetricTracker
from src.utils.checkpoint import atomic_save, dataset_identity, restore_rng, rng_state
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
        resume_checkpoint=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_loader = dataloaders["train"]
        self.train_batch_sampler = self.train_loader.batch_sampler
        self.train_dataloader = self.train_loader
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self._epoch_complete = True
        self._epoch_step = 0
        self.global_step = 0
        self.not_improved_count = 0
        self.max_epochs_per_session = self.cfg_trainer.get("max_epochs_per_session")
        if self.max_epochs_per_session is not None and self.max_epochs_per_session < 1:
            raise ValueError("max_epochs_per_session must be positive or null.")
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
            transforms={m.name: m.compute for m in self.metrics["train"]},
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
            transforms={m.name: m.compute for m in self.metrics["inference"]},
        )
        self.epoch_metrics = MetricTracker(
            *self.train_metrics.keys(), transforms=self.train_metrics.transforms
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path, resume_checkpoint)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, last_only=True)
            raise e
        finally:
            if hasattr(self.train_dataloader, "close"):
                self.train_dataloader.close()

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        session_epochs = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if epoch != self.start_epoch or self._epoch_complete:
                self._epoch_step = 0
                self.train_metrics.reset()
                self.epoch_metrics.reset()
            self._last_epoch = epoch
            self._epoch_complete = False
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, self.not_improved_count = self._monitor_performance(
                logs, self.not_improved_count
            )
            self._epoch_complete = True
            self._save_checkpoint(epoch, save_best=best, only_best=True)
            session_epochs += 1

            if stop_process:  # early_stop
                break
            if (
                self.max_epochs_per_session is not None
                and session_epochs >= self.max_epochs_per_session
            ):
                self.logger.info(
                    "Session limit reached; continue from checkpoint-last.pth."
                )
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.writer.set_step(self.global_step)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(
                islice(self.train_dataloader, self.epoch_len - self._epoch_step),
                desc="train",
                total=self.epoch_len,
                initial=self._epoch_step,
            ),
            start=self._epoch_step,
        ):
            if batch_idx >= self.epoch_len:
                break
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    self._mark_batch_consumed()
                    self._epoch_step = batch_idx + 1
                    if self._epoch_step >= self.epoch_len:
                        break
                    continue
                else:
                    raise e

            self._mark_batch_consumed()
            self.global_step += 1
            self._epoch_step = batch_idx + 1
            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if (batch_idx + 1) % self.log_step == 0 or batch_idx + 1 >= self.epoch_len:
                self.writer.set_step(self.global_step)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                self.epoch_metrics.merge(self.train_metrics)
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        # Include a remaining window if the final batch was skipped after OOM.
        self.epoch_metrics.merge(self.train_metrics)
        self.train_metrics.reset()
        logs = self.epoch_metrics.result()
        if hasattr(self.train_batch_sampler, "pass_index"):
            sampler = self.train_batch_sampler
            logs["data_passes"] = sampler.pass_index + (
                sampler.batch_offset * sampler.batch_size / sampler.size
            )
            self.writer.add_scalar("data_passes", logs["data_passes"])

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _mark_batch_consumed(self):
        if hasattr(self.train_batch_sampler, "mark_consumed"):
            self.train_batch_sampler.mark_consumed()

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(self.global_step, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch, rather than individual dataset samples.
        Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(
        self, epoch, save_best=False, only_best=False, last_only=False
    ):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
            "global_step": self.global_step,
            "epoch_len": self.epoch_len,
            "epoch_complete": self._epoch_complete,
            "epoch_step": self._epoch_step,
            "not_improved_count": self.not_improved_count,
            "rng_state": rng_state(),
            "data_state": (
                self.train_batch_sampler.state_dict()
                if hasattr(self.train_batch_sampler, "state_dict")
                else None
            ),
            "validation_identity": (
                dataset_identity(self.evaluation_dataloaders["val"].dataset)
                if "val" in self.evaluation_dataloaders
                else None
            ),
            "train_metrics": self.train_metrics.state_dict(),
            "epoch_metrics": self.epoch_metrics.state_dict(),
            "tokenizer_sha256": (
                tokenizer_digest(self.config.model.tokenizer_path)
                if self.config.model.get("tokenizer_path") is not None
                else None
            ),
        }
        atomic_save(state, self.checkpoint_dir / "checkpoint-last.pth")
        if last_only:
            return
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if (
            self.save_period > 0
            and epoch % self.save_period == 0
            and not (only_best and save_best)
        ):
            atomic_save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            atomic_save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, checkpoint=None):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        A checkpoint may come from the current run or a previous session's
        mounted output directory.

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        if checkpoint is None:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_epoch_len = checkpoint.get(
            "epoch_len", checkpoint["config"]["trainer"].get("epoch_len")
        )
        if saved_epoch_len is not None and saved_epoch_len != self.epoch_len:
            raise ValueError("Keep trainer.epoch_len unchanged when resuming.")
        self._epoch_complete = checkpoint.get("epoch_complete", True)
        self.start_epoch = checkpoint["epoch"] + int(self._epoch_complete)
        self._epoch_step = (
            0 if self._epoch_complete else checkpoint.get("epoch_step", 0)
        )
        self.global_step = checkpoint.get(
            "global_step", checkpoint["epoch"] * self.epoch_len
        )
        self.not_improved_count = checkpoint.get("not_improved_count", 0)
        self.mnt_best = checkpoint["monitor_best"]

        saved_model = dict(checkpoint["config"]["model"])
        current_model = dict(self.config.model)
        saved_model.pop("tokenizer_path", None)
        current_model.pop("tokenizer_path", None)
        if saved_model != current_model:
            self.logger.warning(
                "Model configuration changed; checkpoint weights must remain compatible."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        saved_optimizer = dict(checkpoint["config"]["optimizer"])
        current_optimizer = dict(self.config.optimizer)
        reset_scheduler = self.cfg_trainer.get("reset_scheduler", False)
        if reset_scheduler:
            saved_optimizer.pop("lr", None)
            current_optimizer.pop("lr", None)
        if saved_optimizer != current_optimizer:
            raise ValueError(
                "Optimizer configuration changed; use from_pretrained for weights only."
            )
        configured_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if reset_scheduler:
            for group, lr in zip(self.optimizer.param_groups, configured_lrs):
                group["lr"] = group["initial_lr"] = lr
            self.logger.info(
                "Optimizer restored; starting the configured new LR cycle."
            )
        elif checkpoint["config"]["lr_scheduler"] != self.config.lr_scheduler:
            raise ValueError(
                "Scheduler changed; set trainer.reset_scheduler=true for a new LR cycle."
            )
        else:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        validation = (
            dataset_identity(self.evaluation_dataloaders["val"].dataset)
            if "val" in self.evaluation_dataloaders
            else None
        )
        if checkpoint.get("validation_identity") != validation:
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.not_improved_count = 0
            self.logger.warning(
                "Validation changed or is unknown; resetting best metric."
            )
        if not self.cfg_trainer.get("reset_data_state", False):
            if checkpoint.get("data_state") is not None:
                self.train_batch_sampler.load_state_dict(checkpoint["data_state"])
            else:
                self.logger.warning(
                    "Legacy checkpoint has no data cursor; starting a new data pass."
                )
            if not self._epoch_complete:
                self.train_metrics.load_state_dict(checkpoint["train_metrics"])
                self.epoch_metrics.load_state_dict(checkpoint["epoch_metrics"])
        elif not self._epoch_complete:
            raise ValueError(
                "Finish the current training window before switching data groups."
            )
        if checkpoint.get("rng_state") is not None:
            restore_rng(checkpoint["rng_state"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device, weights_only=False)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
