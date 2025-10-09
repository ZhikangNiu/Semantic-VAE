import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.data import transforms
from audiotools.ml.decorators import Tracker
from ema_pytorch import EMA
from torch.utils.tensorboard import SummaryWriter

import dac
from dac.data.datasets import AudioDataset
from dac.data.datasets import AudioLoader
from dac.data.datasets import ConcatDataset
from dac.model.utils import prepare_batch
from dac.model.utils import save_to_folder
from dac.model.utils import set_seed

warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
params = argbind.parse_args()
DAC = argbind.bind(dac.model.DAC)
metainfo = {k.replace("DAC.", ""): v for k, v in params.items() if k.startswith("DAC.")}
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC
    ema_generator: EMA
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


def save_metainfo(save_path):
    params = argbind.parse_args()
    dac_params = {
        k.replace("DAC.", ""): v for k, v in params.items() if k.startswith("DAC.")
    }
    disc_params = {
        k.replace("Discriminator.", ""): v
        for k, v in params.items()
        if k.startswith("Discriminator.")
    }
    decoder_params = read_json_file(params["DAC.bigvgan_conf"])
    metainfo = {"DAC": dac_params, "Decoder": decoder_params}
    with open(Path(save_path) / "metainfo.json", "w") as f:
        metainfo["Discriminator"] = disc_params
        json.dump(metainfo, f, ensure_ascii=False)

    with open(Path(save_path) / "config.json", "w") as f:
        metainfo["params"] = params
        metainfo["Discriminator"] = disc_params
        json.dump(params, f, indent=4, ensure_ascii=False)


def read_json_file(metainfo_path):
    with open(metainfo_path, "r") as f:
        data = json.load(f)
    return data


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator = None
    discriminator = None
    ema_generator = None

    if resume:
        ckpt_folder = Path(f"{save_path}/{tag}")
        generator_ckpt = ckpt_folder / "dac" / "weights.pth"
        generator_dict = torch.load(generator_ckpt, map_location="cpu")["state_dict"]

        discriminator_ckpt = ckpt_folder / "discriminator" / "weights.pth"
        discriminator_dict = torch.load(discriminator_ckpt, map_location="cpu")[
            "state_dict"
        ]
        metainfo_path = ckpt_folder / "metainfo.json"
        metainfo = read_json_file(metainfo_path)

        generator = DAC(**metainfo["DAC"])
        generator.load_state_dict(generator_dict, strict=True)

        discriminator = Discriminator(**metainfo["Discriminator"])
        discriminator.load_state_dict(discriminator_dict, strict=True)

        tracker.print(f"Resuming from {ckpt_folder}")

        if accel.local_rank == 0:
            ema_generator_ckpt = ckpt_folder / "dac" / "ema_state_dict.pth"
            if ema_generator_ckpt.exists():
                ema_generator_dict = torch.load(ema_generator_ckpt, map_location="cpu")
                ema_generator = EMA(generator, include_online_model=False)
                ema_generator.load_state_dict(ema_generator_dict, strict=True)
                tracker.print(f"Resume load ema_generator from {ema_generator_ckpt}")

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    if accel.local_rank == 0:
        ema_generator = (
            EMA(generator, include_online_model=False)
            if ema_generator is None
            else ema_generator
        )
        ema_generator.to(accel.device)

    tracker.print(f"[Encoder] Parameters: {count_parameters(generator.encoder):,}")
    tracker.print(f"[Decoder] Parameters: {count_parameters(generator.decoder):,}")
    tracker.print(f"[Total] Parameters: {count_parameters(generator):,}")
    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator, find_unused_parameters=True)
    discriminator = accel.prepare_model(
        discriminator, find_unused_parameters=True, broadcast_buffers=False
    )

    for name, param in generator.named_parameters():
        if not param.requires_grad:
            tracker.print(f"Unused parameter in generator: {name}")

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if resume:
        optimizer_g_path = ckpt_folder / "dac" / "optimizer.pth"
        if optimizer_g_path.exists():
            optimizer_g.load_state_dict(
                torch.load(optimizer_g_path, map_location="cpu")
            )
            tracker.print(f"Resume load optimizer_g from {optimizer_g_path}")

        scheduler_g_path = ckpt_folder / "dac" / "scheduler.pth"
        if scheduler_g_path.exists():
            scheduler_g.load_state_dict(
                torch.load(scheduler_g_path, map_location="cpu")
            )
            tracker.print(f"Resume load scheduler_g from {scheduler_g_path}")

        tracker_path = ckpt_folder / "dac" / "tracker.pth"
        tracker.load_state_dict(torch.load(tracker_path, map_location="cpu"))
        tracker.print(f"Resume load tracker from {tracker_path}")

        optimizer_d_path = ckpt_folder / "discriminator" / "optimizer.pth"
        if optimizer_d_path.exists():
            optimizer_d.load_state_dict(
                torch.load(optimizer_d_path, map_location="cpu")
            )
            tracker.print(f"Resume load optimizer_d from {optimizer_d_path}")

        scheduler_d_path = ckpt_folder / "discriminator" / "scheduler.pth"
        if scheduler_d_path.exists():
            scheduler_d.load_state_dict(
                torch.load(scheduler_d_path, map_location="cpu")
            )
            tracker.print(f"Resume load scheduler_d from {scheduler_d_path}")

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
        tracker.print(f"[Train Dataset] Total samples: {len(train_data)}")
        if hasattr(train_data, "datasets"):  # 如果是ConcatDataset
            for i, ds in enumerate(train_data.datasets):
                tracker.print(f"  Subset {i+1}: {len(ds)} samples")
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)
        tracker.print(f"[Val Dataset] Total samples: {len(val_data)}")
        if hasattr(val_data, "datasets"):
            for i, ds in enumerate(val_data.datasets):
                tracker.print(f"  Subset {i+1}: {len(ds)} samples")

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        ema_generator=ema_generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


def train_loop(state, batch, accel, lambdas, current_step, disc_warmup_step=5000):
    start_time = time.perf_counter()
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate, batch["guidance"])
        recons = AudioSignal(out["audio"], signal.sample_rate)
        kl_loss = out["vae/kl_loss"]
        proj_loss = out["vae/proj_loss"]

    if current_step >= disc_warmup_step:
        with accel.autocast():
            output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)

        state.optimizer_d.zero_grad()
        accel.backward(output["adv/disc_loss"])
        accel.scaler.unscale_(state.optimizer_d)
        output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
            state.discriminator.parameters(), 10.0
        )
        accel.step(state.optimizer_d)
        state.scheduler_d.step()
    else:
        output["adv/disc_loss"] = torch.tensor(0.0)
        output["other/grad_norm_d"] = torch.tensor(0.0)

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)

        output["vae/kl_loss"] = kl_loss
        output["vae/proj_loss"] = proj_loss
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 10  # 1e3?
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    # Update EMA weights
    if accel.local_rank == 0:
        state.ema_generator.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size
    output["time/train_loop"] = time.perf_counter() - start_time

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "ema_state_dict.pth": state.ema_generator.state_dict(),
        }

        accel.unwrap(state.generator).metadata = metainfo
        save_to_folder(
            accel.unwrap(state.generator), f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }

        save_to_folder(
            accel.unwrap(state.discriminator), f"{save_path}/{tag}", discriminator_extra
        )
        save_metainfo(f"{save_path}/{tag}")


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def kl_warmup_func(
    kl_start_weight: float,
    kl_end_weight: float,
    total_warmup_steps: int,
    current_step: int,
) -> float:
    if current_step >= total_warmup_steps:
        return kl_end_weight

    warmup_freq = int(total_warmup_steps / 4)
    stage = min(current_step // warmup_freq, 3)
    update_kl = (kl_end_weight - kl_start_weight) / 4
    current_weight = kl_start_weight + (stage + 1) * update_kl
    return min(current_weight, kl_end_weight)


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vae/kl_loss": 1.0,
    },
    use_kl_warmup: bool = False,
    kl_warmup_ratio: float = 0.25,
    kl_start_weight: float = 5e-5,
    disc_warmup_step: int = 5000,
):
    set_seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    # save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    # checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)
    state.tracker.print(f"Loss weights: {lambdas}")
    state.tracker.print(f"disc warm up: {disc_warmup_step}")
    if not args["resume"] and use_kl_warmup:
        total_warmup_steps = int(num_iters * kl_warmup_ratio)
        kl_end_weight = lambdas["vae/kl_loss"]
        state.tracker.print(f"KL start weight: {kl_start_weight}")
        state.tracker.print(f"KL end weight: {kl_end_weight}")
        state.tracker.print(f"KL warmup steps: {total_warmup_steps}")

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            if use_kl_warmup:
                lambdas["vae/kl_loss"] = kl_warmup_func(
                    kl_start_weight, kl_end_weight, total_warmup_steps, tracker.step
                )
            train_loop(
                state,
                batch,
                accel,
                lambdas,
                current_step=tracker.step,
                disc_warmup_step=disc_warmup_step,
            )

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                if accel.local_rank == 0:
                    save_samples(state, val_idx, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                if accel.local_rank == 0:
                    checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
