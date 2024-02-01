import os
import numpy as np
import torch
from torch import nn
from TTS.encoder.configs.i2v_encoder_config import I2VEncoderConfig


class I2VEncoder(nn.Module):
    def __init__(self, config:I2VEncoderConfig):
        pass
    def compute_embedding(self, images: torch.Tensor)  -> torch.Tensor:
        pass
    def load_checkpoint(self, path: os.PathLike) -> None:
        pass
