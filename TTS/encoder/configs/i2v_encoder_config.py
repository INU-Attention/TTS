from dataclasses import asdict, dataclass, field
from typing import Dict
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from TTS.config.shared_configs import BaseTrainingConfig

@dataclass
class I2VEncoderConfig(BaseTrainingConfig):
    model: str = "i2v_encoder"
    class_name_key: str = "speaker_name"
    model_params: Dict = field(
        default_factory=lambda: {
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "hidden_act": "quick_gelu",
            "hidden_size": 1024,
            "image_size": 224,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-05,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "projection_dim": 512,
            "transformers_version": "4.35.2"
        }
    )
    
    