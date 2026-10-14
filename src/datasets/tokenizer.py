"""A shared, lossless byte-level BPE tokenizer for questions and Python code."""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from datasets import Dataset as ArrowDataset

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


class CodeTokenizer:
    """Keep whitespace intact; add sequence boundaries in the dataset layer."""

    def __init__(self, tokenizer: Tokenizer):
        self.backend = tokenizer
        self.backend.no_padding()
        self.backend.no_truncation()
        # Literal special-token strings in source code must also round-trip.
        self.backend.encode_special_tokens = True
        ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
        if any(token_id is None for token_id in ids):
            raise ValueError(f"Tokenizer must contain {SPECIAL_TOKENS}.")
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = ids

    @classmethod
    def train(
        cls,
        train_data: ArrowDataset,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        batch_size: int = 1000,
        show_progress: bool = True,
    ):
        """Train on the training partition only, using both question and code."""
        if not {"question", "code"}.issubset(train_data.column_names):
            raise ValueError("Training data must contain question and code columns.")
        if not len(train_data):
            raise ValueError("Cannot train a tokenizer on an empty dataset.")
        if vocab_size < 256 + len(SPECIAL_TOKENS):
            raise ValueError("Vocabulary must fit 256 byte symbols and special tokens.")
        if batch_size < 1 or min_frequency < 1:
            raise ValueError("batch_size and min_frequency must be positive.")

        tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=show_progress,
        )

        def text_batches():
            for batch in train_data.iter(batch_size=batch_size):
                yield batch["question"] + batch["code"]

        tokenizer.train_from_iterator(
            text_batches(), trainer=trainer, length=2 * len(train_data)
        )
        return cls(tokenizer)

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [
            encoding.ids
            for encoding in self.backend.encode_batch(texts, add_special_tokens=False)
        ]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)
