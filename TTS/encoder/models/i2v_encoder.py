"""
https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/clip/modeling_clip.py
"""

import os
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from TTS.encoder.configs.i2v_encoder_config import I2VEncoderConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

class I2VEncoder(nn.Module):
    def __init__(self, config:Optional[I2VEncoderConfig]=None):
        super(I2VEncoder, self).__init__()
        if config is None:
            config = I2VEncoderConfig()
        vision_config = CLIPVisionConfig(**config.model_params)
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.projection = nn.Linear(
            config.model_params["hidden_size"],
            config.model_params["projection_dim"])
        
    def forward(self, images: torch.Tensor):
        output = self.vision_model(images)[1]
        output = self.projection(output)
        return output

    def compute_embedding(self, images: torch.Tensor)  -> torch.Tensor:
        return self.forward(images)
    
    def load_checkpoint(self, path: os.PathLike) -> None:
        self.load_state_dict(torch.load(path))
