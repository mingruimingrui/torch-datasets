import collections
import torch
from torch._six import string_classes


def cuda_batch(batch, device_idx=None):
    if batch is None:
        return None
    elif isinstance(batch, torch.Tensor):
        return batch.cuda(device_idx)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: cuda_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [cuda_batch(sample) for sample in batch]
    else:
        return batch
