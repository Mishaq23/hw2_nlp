from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.tokenizer import CodeTokenizer
from src.metrics.tracker import MetricTracker
from src.model.generation import generate
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, **kwargs):
        self.generation_best_scores = {}
        self.selected_decoding = {"num_beams": 1, "length_penalty": 1.0}
        super().__init__(*args, **kwargs)
        self.selected_decoding.setdefault(
            "max_new_tokens", self.config.generation_validation.max_new_tokens
        )
        if self.monitor != "off" and self.mnt_metric == "val_pass_at_1":
            if not self.config.generation_validation.enabled:
                raise ValueError(
                    "val_pass_at_1 requires generation_validation.enabled=true; "
                    "use trainer.monitor='min val_loss' to disable generation."
                )
        self._compiled_forward = None
        self.compile_config = self.config.get("compile", {})
        if self.compile_config.get("enabled", False):
            try:
                # Keep the original module for state_dict and autoregressive generation.
                self._compiled_forward = torch.compile(
                    self.model.forward,
                    backend=self.compile_config.get("backend", "inductor"),
                    mode=self.compile_config.get("mode", "default"),
                    dynamic=self.compile_config.get("dynamic"),
                )
                self.logger.info(
                    "torch.compile enabled for training; first batches include compilation."
                )
            except Exception as error:
                if not self.compile_config.get("fallback_to_eager", True):
                    raise
                self.logger.warning("torch.compile unavailable; using eager: %s", error)

    def _forward_backward(self, batch):
        forward = self._compiled_forward if self.is_train else None
        inputs = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        outputs = forward(**inputs) if forward is not None else self.model(**inputs)
        losses = self.criterion(**batch, **outputs)
        if self.is_train:
            losses["loss"].backward()
        return {**outputs, **losses}

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
        ignore_index = self.config.loss_function.get("ignore_index", -100)
        batch["token_count"] = batch["labels"].ne(ignore_index).sum().item()
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad(set_to_none=True)

        try:
            outputs = self._forward_backward(batch)
        except Exception as error:
            compiler_error = type(error).__module__.startswith(
                ("torch._dynamo", "torch._inductor", "triton")
            )
            if not (
                self.is_train
                and self._compiled_forward is not None
                and compiler_error
                and self.compile_config.get("fallback_to_eager", True)
            ):
                raise
            self.logger.warning(
                "torch.compile failed; retrying this batch in eager: %s", error
            )
            self._compiled_forward = None
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self._forward_backward(batch)
        batch.update(outputs)

        if self.is_train:
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        loss_weight = batch["token_count"]
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=loss_weight)
        if batch.get("mean_nll") is not None:
            batch["mean_nll"] = batch["loss"].detach().item()

        for met in metric_funcs:
            met.update(metrics, **batch)
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        final = epoch == self.epochs or self._time_limit_reached
        every = self.cfg_trainer.get("validation_every_n_epochs", 1)
        logs = {}
        if epoch == 0 or final or epoch % every == 0:
            start = perf_counter()
            logs = super()._evaluation_epoch(epoch, part, dataloader)
            logs["teacher_forcing_seconds"] = perf_counter() - start
            self.writer.add_scalar(
                "teacher_forcing_seconds", logs["teacher_forcing_seconds"]
            )
        config = self.config.get("generation_validation")
        if (
            config is not None
            and config.enabled
            and (epoch % config.every_n_epochs == 0 or final)
        ):
            self.is_train = False
            self.model.eval()
            self.writer.set_step(self.completed_steps, part)
            generated_logs = self._evaluate_generation(dataloader, config)
            logs.update(generated_logs)
            for name, value in generated_logs.items():
                self.writer.add_scalar(name, value)
        return logs

    @torch.no_grad()
    def _evaluate_generation(self, dataloader, config):
        modes = {}
        for num_beams in config.get("num_beams", [1]):
            name = "greedy" if num_beams == 1 else f"beam{num_beams}"
            modes[name] = self._evaluate_generation_mode(dataloader, config, num_beams)
        # Prefer the first configured mode (greedy by default) on equal scores.
        winner = max(modes, key=lambda name: modes[name].get("pass_at_1", 0.0))
        num_beams = 1 if winner == "greedy" else int(winner.removeprefix("beam"))
        self.selected_decoding = {
            "num_beams": num_beams,
            "length_penalty": config.get("length_penalty", 1.0),
            "max_new_tokens": config.max_new_tokens,
        }
        return {
            **modes[winner],
            "selected_num_beams": num_beams,
            **{
                f"{metric}_{name}": value
                for name, results in modes.items()
                for metric, value in results.items()
            },
        }

    @torch.no_grad()
    def _evaluate_generation_mode(self, dataloader, config, num_beams):
        metric_funcs = self.metrics["generation"]
        tracker = MetricTracker(
            *[name for metric in metric_funcs for name in metric.names],
            transforms={m.name: m.compute for m in metric_funcs},
        )
        generator = torch.Generator().manual_seed(config.get("seed", 42))
        indices = torch.randperm(len(dataloader.dataset), generator=generator)[
            : config.max_examples
        ].tolist()
        dataloader = DataLoader(
            Subset(dataloader.dataset, indices),
            batch_size=config.get("batch_size") or dataloader.batch_size,
            collate_fn=dataloader.collate_fn,
            num_workers=0,
            generator=generator,
        )
        tokenizer = CodeTokenizer.from_file(self.config.model.tokenizer_path)
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        count = 0
        execution_seconds = 0.0
        examples = []
        for batch in dataloader:
            size = min(config.max_examples - count, batch["encoder_input_ids"].size(0))
            if size <= 0:
                break
            generated_ids = generate(
                self.model,
                batch["encoder_input_ids"][:size].to(device),
                batch["encoder_attention_mask"][:size].to(device),
                config.max_new_tokens,
                num_beams=num_beams,
                length_penalty=config.get("length_penalty", 1.0),
            )
            generated_code = [
                tokenizer.decode(ids) for ids in generated_ids.cpu().tolist()
            ]
            ignore_index = self.config.loss_function.get("ignore_index", -100)
            reference_code = [
                tokenizer.decode(labels[labels.ne(ignore_index)].tolist())
                for labels in batch["labels"][:size]
            ]
            execution_start = perf_counter()
            for met in metric_funcs:
                met.update(
                    tracker,
                    generated_code=generated_code,
                    reference_code=reference_code,
                )
            execution_seconds += perf_counter() - execution_start
            for index in range(min(size, 8 - len(examples))):
                labels = batch["labels"][index]
                ignore_index = self.config.loss_function.get("ignore_index", -100)
                examples.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "question": tokenizer.decode(
                            batch["encoder_input_ids"][index].tolist()
                        ),
                        "reference_code": tokenizer.decode(
                            labels[labels.ne(ignore_index)].tolist()
                        ),
                        "generated_code": generated_code[index],
                    }
                )
            count += size
            if count >= config.max_examples:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logs = {
            **tracker.result(),
            "generation_seconds": perf_counter() - start - execution_seconds,
            "execution_seconds": execution_seconds,
            "generated_examples": count,
        }
        if "reference_validity" in logs:
            logs["pass_at_1_examples"] = round(logs["reference_validity"] * count)
            if logs["pass_at_1_examples"] == 0:
                raise ValueError(
                    "No executable numeric validation references; pass@1 cannot be measured."
                )
        mode = "greedy" if num_beams == 1 else f"beam{num_beams}"
        self.writer.add_table(f"generation_examples_{mode}", pd.DataFrame(examples))
        return logs

    def _monitor_performance(self, logs, not_improved_count):
        config = self.config.get("generation_validation")
        if config is not None:
            for num_beams in config.get("num_beams", [1]):
                name = "greedy" if num_beams == 1 else f"beam{num_beams}"
                score = logs.get(f"val_pass_at_1_{name}")
                if score is None or score <= self.generation_best_scores.get(
                    name, -float("inf")
                ):
                    continue
                self.generation_best_scores[name] = score
                path = self.checkpoint_dir / f"model_best_{name}.pth"
                # These files are for inference; optimizer state stays in the
                # regular model_best/periodic training checkpoints.
                self._write_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "epoch": logs["epoch"],
                        "config": self.config,
                        "validation_pass_at_1": score,
                        "decoding": {
                            "num_beams": num_beams,
                            "length_penalty": config.get("length_penalty", 1.0),
                            "max_new_tokens": config.max_new_tokens,
                        },
                    },
                    path,
                )
                self.logger.info("Saving %s: pass@1=%.4f", path.name, score)
        return super()._monitor_performance(logs, not_improved_count)

    def _checkpoint_metadata(self):
        return {
            "decoding": self.selected_decoding,
            "generation_best_scores": self.generation_best_scores,
        }

    def _restore_checkpoint_metadata(self, checkpoint):
        self.selected_decoding = checkpoint.get("decoding", self.selected_decoding)
        self.generation_best_scores = checkpoint.get("generation_best_scores", {})

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
