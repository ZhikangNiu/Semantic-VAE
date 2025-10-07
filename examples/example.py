import warnings
from pathlib import Path

import librosa
import torch
import torchaudio

from dac.model.dac import DAC
from dac.model.utils import read_json_file

warnings.filterwarnings("ignore")


def load_model(save_path: str) -> DAC:
    metainfo = read_json_file(Path(save_path) / "metainfo.json")
    ckpt = torch.load(
        Path(save_path) / "dac" / "ema_state_dict.pth", map_location="cpu"
    )

    ckpt = {k.replace("ema_model.", ""): v for k, v in ckpt.items()}
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("projectors")}

    model = DAC(**metainfo["DAC"])
    del model.projectors
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


# load model
save_path = "ckpts/semantic_vae"
model = load_model(save_path=save_path)

# load audio
audio_path = Path("examples/8461_281231_000023_000000.wav")
wav, sr = librosa.load(
    audio_path,
    sr=None,
)

# resample
if sr != model.sample_rate:
    wav = librosa.resample(y=wav, orig_sr=sr, target_sr=model.sample_rate)
    wav = torch.from_numpy(wav).unsqueeze(0)
    wav = model.preprocess(wav, model.sample_rate)  # 1, T

# encode
z_hat, _, _, _ = model.encode(wav.unsqueeze(0))  # 1, T -> 1, 1, T

# decode
x_hat = model.decode(z_hat)

out_path = Path(__file__).resolve().parent / f"{Path(audio_path).stem}_recon.wav"
torchaudio.save(out_path, x_hat.squeeze(0).cpu(), sample_rate=model.sample_rate)
