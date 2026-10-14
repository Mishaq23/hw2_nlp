from time import perf_counter

import torch

from src.datasets.tokenizer import CodeTokenizer
from src.metrics.tracker import MetricTracker
from src.model.generation import greedy_generate
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        loss_weight = 1
        if batch["labels"].ndim == 2:
            ignore_index = self.config.loss_function.get("ignore_index", -100)
            loss_weight = batch["labels"].ne(ignore_index).sum().item()
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=loss_weight)

        for met in metric_funcs:
            met.update(metrics, **batch)
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        logs = super()._evaluation_epoch(epoch, part, dataloader)
        config = self.config.get("generation_validation")
        if config is not None and config.enabled and epoch % config.every_n_epochs == 0:
            generated_logs = self._evaluate_generation(dataloader, config)
            logs.update(generated_logs)
            for name, value in generated_logs.items():
                self.writer.add_scalar(name, value)
        return logs

    @torch.no_grad()
    def _evaluate_generation(self, dataloader, config):
        metric_funcs = self.metrics["generation"]
        tracker = MetricTracker(
            *[m.name for m in metric_funcs],
            transforms={m.name: m.compute for m in metric_funcs},
        )
        tokenizer = CodeTokenizer.from_file(self.config.model.tokenizer_path)
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        count = 0
        for batch in dataloader:
            size = min(config.max_examples - count, batch["encoder_input_ids"].size(0))
            if size <= 0:
                break
            generated_ids = greedy_generate(
                self.model,
                batch["encoder_input_ids"][:size].to(device),
                batch["encoder_attention_mask"][:size].to(device),
                config.max_new_tokens,
            )
            generated_code = [
                tokenizer.decode(ids) for ids in generated_ids.cpu().tolist()
            ]
            for met in metric_funcs:
                met.update(tracker, generated_code=generated_code)
            count += size
            if count >= config.max_examples:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            **tracker.result(),
            "generation_seconds": perf_counter() - start,
            "generated_examples": count,
        }

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
