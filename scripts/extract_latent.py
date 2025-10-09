import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from audiotools import AudioSignal
from audiotools.core import util
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from train import DAC

from dac.data.datasets import find_audio
from dac.model.utils import read_json_file


class AudioPathDataset(Dataset):
    def __init__(self, input_dir):
        self.audio_root_dir = input_dir
        self.file_lines = find_audio(input_dir)

    def __len__(self):
        return len(self.file_lines)

    def __getitem__(self, idx):
        file_path = self.file_lines[idx]
        relative_path = file_path.relative_to(self.audio_root_dir)
        return str(file_path), str(relative_path)


def load_state(save_path: str, tag: str = "latest", use_ema: bool = False):
    folder = f"{save_path}/{tag}"
    print(f"Resuming from {str(Path('.').absolute())}/{folder}")
    metainfo_path = Path(".").absolute() / folder / "metainfo.json"
    metainfo = read_json_file(metainfo_path)
    if not use_ema:
        ckpt_path = Path(folder) / "dac" / "weights.pth"
        model_dict = torch.load(ckpt_path, map_location="cpu")
        filter_dict = {
            k: v
            for k, v in model_dict["state_dict"].items()
            if not k.startswith("projectors")
        }
    else:
        ckpt_path = Path(folder) / "dac" / "ema_state_dict.pth"
        model_dict = torch.load(ckpt_path, map_location="cpu")
        ckpt_dict = {k.replace("ema_model.", ""): v for k, v in model_dict.items()}
        filter_dict = {
            k: v for k, v in ckpt_dict.items() if not k.startswith("projectors")
        }
    print(f"Load from {ckpt_path}, use_ema: {use_ema}")
    generator = DAC(**metainfo["DAC"])
    del generator.projectors
    generator.load_state_dict(filter_dict, strict=False)
    generator.eval()
    return generator


@torch.no_grad()
def process(signal, generator):
    data = signal.audio_data.cuda()
    if signal.sample_rate != generator.sample_rate:
        signal.resample(generator.sample_rate)
    audio_data = generator.preprocess(data, signal.sample_rate)
    _, mu, log_var, _ = generator.encode(audio_data)
    pre_proj_latent = generator.reparameterize(mu, log_var)
    return pre_proj_latent.transpose(1, 2).cpu().numpy()


@torch.no_grad()
def get_samples(
    ckpt_dir: str,
    input_dir: str,
    output_dir: str,
    model_tag: str,
    global_seed: int,
    use_ema: bool,
):
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    generator = load_state(save_path=ckpt_dir, tag=model_tag, use_ema=use_ema).to(
        device
    )
    generator.eval()
    dataset = AudioPathDataset(input_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    if rank == 0:
        print(f"Generator SR = {generator.sample_rate}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    for file_path, relative_path in tqdm(loader):
        signal = AudioSignal(file_path[0])
        if signal.sample_rate != generator.sample_rate:
            signal = signal.resample(generator.sample_rate)
        feat = process(signal, generator)
        save_path = Path(output_dir) / relative_path[0]
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        np.save(save_path.with_suffix(".npy"), feat)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--input_dir", type=str, default="samples/input")
    parser.add_argument("--output_dir", type=str, default="samples/output")
    parser.add_argument("--model_tag", type=str, default="best")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()
    get_samples(
        ckpt_dir=args.ckpt_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
        global_seed=args.global_seed,
        use_ema=args.use_ema,
    )
