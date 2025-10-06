import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.model.utils import make_pad_mask
from .bigvgan import BigVGAN
from .regulator import InterpolateRegulator
from .attn_proj import AttnProjection
import json
import torch.nn.functional as F

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=0 if stride % 2 == 0 else 1
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResidualBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # 替换为 LayerNorm
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)   # 替换为 LayerNorm
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        vae_dim: Union[int, list] = 8,
        sample_rate: int = 44100,
        distill: bool = False,
        distill_hidden_dim: int = 1024,
        decoder_type : str = "dac", # bigvgan | dac
        attn_proj: bool = False,
        post_vae_block: bool = False,
        bigvgan_conf: str = "conf/bigvgan_conf/bigvgan_v2_24khz_100band_256x.json",
        sampling_ratios: list = [0,1],
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.sample_rate = sample_rate
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.vae_dim = vae_dim
        self.attn_proj = attn_proj
        self.post_vae_block = post_vae_block
        
        self.pre_block = AttnProjection(latent_dim,self.vae_dim,num_heads=8)

        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_var = nn.Linear(self.vae_dim, self.vae_dim)
        
        if self.attn_proj:
            self.decoder_proj = AttnProjection(self.vae_dim,latent_dim,8)
        else:
            self.decoder_proj = nn.Linear(self.vae_dim,latent_dim)
        
        self.decoder_type = decoder_type
        if self.decoder_type == "dac":
            self.decoder = Decoder(
                latent_dim,
                decoder_dim,
                decoder_rates,
            )
            self.apply(init_weights)
        elif self.decoder_type == "bigvgan":
            self.bigvgan_conf = bigvgan_conf
            with open(self.bigvgan_conf) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            self.decoder = BigVGAN(h)

        self.distill = distill
        self.distill_hidden_dim = distill_hidden_dim

        proj_dim = self.vae_dim * 2
        self.projectors = InterpolateRegulator(sampling_ratios,self.distill_hidden_dim, proj_dim, self.vae_dim)
        
        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) :
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def compute_kl_loss(self, mu, log_var):
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var.exp() + 1e-6), dim=-1)
        return kl_loss.mean() 
    
    def encode(
        self,
        audio_data: torch.Tensor
    ):
        z = self.encoder(audio_data).transpose(1,2) # torch.Size([72, 1024, 29]),[B x D x T] -> torch.Size([72, 29, 1024]),[B x T x D] ->vq torch.Size([72, 29, 8]),[B x D x T]
        z = self.pre_block(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        log_var = torch.clamp(log_var, min=-12, max=12)

        z_hat = self.reparameterize(mu,log_var)
        kl_loss = self.compute_kl_loss(mu,log_var)
        
        return z_hat, mu, log_var, kl_loss

    def decode(self, z: torch.Tensor):
        if self.decoder_type == "dac":
            z = self.decoder_proj(z).transpose(1,2) 
            recon = self.decoder(z)
        elif self.decoder_type == "bigvgan":
            recon = self.decoder(z.transpose(1,2))   
        return recon

    def forward(
        self,
        audio_data: torch.Tensor, # B, 1, T (duration)
        sample_rate: int = None,
        guidance: torch.Tensor = None # B, T, D
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, mu, log_var, kl_loss = self.encode(audio_data)
        proj_loss = 0.
        if self.distill and self.training:
            guidance_lengths = [g.shape[0] for g in guidance]
            z_lens = [zi.shape[0] for zi in z]
            target_lengths = torch.tensor(guidance_lengths, device=z.device)
            z_lengths = torch.tensor(z_lens, device = z.device)
            z_mask = make_pad_mask(z_lengths, max_len=torch.max(target_lengths)) # 16 150, padding的部分是1
            g_mask = make_pad_mask(target_lengths, max_len=torch.max(z_lengths))

            proj_g, olens = self.projectors(guidance, target_lengths, z_lengths)
            bsz,seq_len,distill_dim = proj_g.shape[0],proj_g.shape[1],proj_g.shape[2]
            for i, (pi, gi) in enumerate(zip(z,proj_g)):
                cos_sim = F.cosine_similarity(
                    pi, # 150, 1024
                    gi, # 150, 1024
                    dim = -1
                )

                proj_loss += masked_mean(-cos_sim, ~g_mask[i])
            proj_loss = proj_loss / bsz
            
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "vae/kl_loss": kl_loss,
            "vae/proj_loss": proj_loss
        }