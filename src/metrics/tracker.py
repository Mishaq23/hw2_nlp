import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None, transforms=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
            transforms (dict | None): functions applied after averaging,
                e.g. exp(mean NLL) for perplexity.
        """
        self.writer = writer
        self.transforms = transforms or {}
        self._data = pd.DataFrame(
            0.0, index=keys, columns=["total", "counts", "average"]
        )

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        self._data.loc[:, :] = 0.0

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        if n == 0:
            return
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        value = self._data.loc[key, "average"]
        if key in self.transforms:
            value = self.transforms[key](value)
        return float(value)

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return {key: self.avg(key) for key in self.keys()}

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
