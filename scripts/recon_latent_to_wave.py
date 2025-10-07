import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from audiotools import AudioSignal
from audiotools.core import util
from extract_latent import load_state
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from train import DAC

from dac.model.utils import read_json_file


# Custom Dataset for audio file paths
class LatentPathDataset(Dataset):
    def __init__(self, input_dir):  # Added file_extension
        self.audio_root_dir = input_dir
        self.file_lines = list(Path(input_dir).rglob("*.npy"))

    def __len__(self):
        return len(self.file_lines)

    def __getitem__(self, idx):
        file_path = self.file_lines[idx]
        return str(file_path)


@torch.no_grad()
def recon_wav_from_latent(latent_data, generator):
    z_hat = torch.from_numpy(latent_data).cuda()
    final_result = generator.decode(z_hat.transpose(1, 2))
    return final_result


@torch.no_grad()
def recon_samples(
    ckpt_dir: str = "ckpt",
    input_dir: str = "samples/input",
    model_tag: str = "best",
    global_seed: int = 42,
    use_ema: bool = False,
):
    # Setup DDP:
    dist.init_process_group("nccl")
    # world_size is useful for DistributedSampler, batch size calculations etc.
    world_size = dist.get_world_size()
    # Batch size per GPU. For feature extraction, often 1.
    # args.global_batch_size would be the total batch across all GPUs.
    # Here, we assume DataLoader batch_size is per-GPU.
    # assert args.global_batch_size % world_size == 0, f"Global batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    generator = load_state(save_path=ckpt_dir, tag=model_tag, use_ema=use_ema).to(
        device
    )
    generator.eval()

    dataset = LatentPathDataset(input_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=1,  # Batch size per GPU
        shuffle=False,  # Shuffle is handled by sampler
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,  # Process all files
    )

    for file_path in tqdm(loader):
        output_audio = Path(file_path[0]).with_suffix(".wav")
        if output_audio.exists():
            continue
        latent = np.load(file_path[0])
        recon = recon_wav_from_latent(latent, generator)
        torchaudio.save(
            output_audio, recon.squeeze(0).cpu(), sample_rate=generator.sample_rate
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--input_dir", type=str, default="samples/input")
    parser.add_argument("--model_tag", type=str, default="best")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()
    recon_samples(
        ckpt_dir=args.ckpt_dir,
        input_dir=args.input_dir,
        model_tag=args.model_tag,
        global_seed=args.global_seed,
        use_ema=args.use_ema,
    )
