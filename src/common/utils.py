from transformers import AutoTokenizer
import os
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict


def setup_tokenizer(model_name_or_path: str, max_length: int) -> AutoTokenizer:
    """Set up tokenizer with proper padding and length settings.

    Args:
        model_name_or_path: Path or name of the model to load tokenizer from
        max_length: Maximum sequence length for the tokenizer

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = max_length
    return tokenizer


def load_wrapper(path, split="train"):
    if path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path)
    elif os.path.exists(path):
        dataset = load_from_disk(path)
    else:
        dataset = load_dataset(path)

    # Extract train split if it exists
    if isinstance(dataset, DatasetDict) and split in dataset:
        return dataset[split]
    return dataset
