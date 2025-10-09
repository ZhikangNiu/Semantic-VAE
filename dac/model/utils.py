import json
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from scipy.io.wavfile import write
from torch.nn.utils.parametrizations import weight_norm


def prepare_batch(batch: Union[dict, list, torch.Tensor], device: str = "cpu"):
    if isinstance(batch, dict):
        batch = flatten(batch)
        for key, val in batch.items():
            try:
                batch[key] = val.to(device)
            except:
                pass
        batch = unflatten(batch)
    elif torch.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch


def set_seed(random_seed, set_cudnn=False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_to_folder(
    model,
    folder: Union[str, Path],
    extra_data: dict | None = None,
) -> Path:
    extra_data = {} if extra_data is None else extra_data
    model_name = type(model).__name__.lower()
    target_base = Path(folder) / model_name
    target_base.mkdir(parents=True, exist_ok=True)

    # 仅保存权重（不含元数据）
    weights_path = target_base / "weights.pth"
    torch.save(model.state_dict(), weights_path)

    # 保存额外数据
    for rel_path, obj in extra_data.items():
        torch.save(obj, target_base / rel_path)

    return target_base


def read_json_file(metainfo_path):
    with open(metainfo_path, "r") as f:
        data = json.load(f)
    return data


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
