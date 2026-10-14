from torch import nn


class CodeGenerationLoss(nn.Module):
    """Mean cross-entropy over target tokens, including EOS and ignoring padding."""

    def __init__(self, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )

    def forward(self, logits, labels, **batch):
        # Dataset already aligns decoder inputs and labels; no extra shift.
        return {"loss": self.loss(logits.flatten(0, 1), labels.flatten())}
