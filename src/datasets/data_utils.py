from itertools import repeat

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.sampler import ResumableBatchSampler
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config.

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    batch_collate = instantiate(config.dataloader.collate_fn)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        if dataset_partition == "train":
            assert config.dataloader.batch_size <= len(dataset), (
                f"The batch size ({config.dataloader.batch_size}) cannot "
                f"be larger than the training dataset length ({len(dataset)})"
            )

        loader_config = OmegaConf.to_container(config.dataloader, resolve=True)
        # A separate generator keeps worker startup from consuming model RNG.
        seed = config.get("trainer", config.get("inferencer", {})).get("seed", 0)
        kwargs = dict(
            dataset=dataset,
            collate_fn=batch_collate,
            worker_init_fn=set_worker_seed,
            generator=torch.Generator().manual_seed(seed),
        )
        if dataset_partition == "train":
            batch_size = loader_config.pop("batch_size")
            loader_config.pop("shuffle", None)
            loader_config.pop("drop_last", None)
            kwargs["batch_sampler"] = ResumableBatchSampler(dataset, batch_size, seed)
        else:
            kwargs.update(drop_last=False, shuffle=False)
        partition_dataloader = instantiate(loader_config, **kwargs)
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
