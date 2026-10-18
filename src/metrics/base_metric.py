from abc import abstractmethod


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @property
    def names(self):
        """Tracker keys emitted by this metric."""
        return (self.name,)

    def weight(self, **batch):
        """Number of observations represented by a batch metric."""
        return 1

    def update(self, tracker, **batch):
        tracker.update(self.name, float(self(**batch)), n=self.weight(**batch))

    def compute(self, average):
        """Convert the accumulated average into the reported metric."""
        return average

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()
