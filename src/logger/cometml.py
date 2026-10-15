"""Comet Experiment Management adapter for the template writer interface."""

from time import perf_counter


class CometMLWriter:
    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        offline_directory="saved/comet_offline",
        **kwargs,
    ):
        import comet_ml

        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("Comet mode must be online, offline, or disabled.")

        self.exp = comet_ml.start(
            project_name=project_name,
            workspace=workspace,
            experiment_key=run_id,
            online=mode != "offline",
            experiment_config=comet_ml.ExperimentConfig(
                name=run_name,
                disabled=mode == "disabled",
                auto_metric_logging=False,
                auto_param_logging=False,
                log_graph=False,
                log_code=False,
                parse_args=False,
                offline_directory=offline_directory,
            ),
        )
        self.run_id = run_id
        self.exp.log_parameters(project_config)
        self.step = 0
        self.mode = "train"
        self.timer = perf_counter()
        if mode == "online":
            logger.info("Comet experiment: %s", self.exp.url)

    def set_step(self, step, mode="train"):
        now = perf_counter()
        previous_step, previous_mode = self.step, self.mode
        self.step, self.mode = step, mode
        if mode == previous_mode == "train" and step > previous_step:
            self.add_scalar(
                "steps_per_sec", (step - previous_step) / (now - self.timer)
            )
        self.timer = now

    def _object_name(self, name):
        return f"{name}_{self.mode}" if self.mode else name

    def add_scalar(self, scalar_name, scalar):
        self.exp.log_metric(self._object_name(scalar_name), scalar, step=self.step)

    def add_scalars(self, scalars):
        self.exp.log_metrics(
            {self._object_name(name): value for name, value in scalars.items()},
            step=self.step,
        )

    def add_table(self, table_name, table):
        self.exp.log_table(
            filename=f"{self._object_name(table_name)}_step{self.step}.csv",
            tabular_data=table,
            headers=True,
        )

    def add_checkpoint(self, checkpoint_path, save_dir):
        self.exp.log_model("checkpoints", checkpoint_path, overwrite=True)

    def finish(self):
        self.exp.end()
