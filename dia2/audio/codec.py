from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import MimiModel


DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"


@dataclass(frozen=True)
class MimiConfig:
    model_id: str = DEFAULT_MIMI_MODEL_ID
    dtype: Optional[torch.dtype] = None


class MimiCodec(nn.Module):
    """Thin wrapper around transformers' MimiModel for decoding audio tokens."""

    def __init__(self, model: MimiModel, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        cfg = getattr(model, "config", None)
        self.sample_rate = getattr(cfg, "sampling_rate", 24000)
        self.frame_rate = getattr(cfg, "frame_rate", 12.5)
        self.samples_per_frame = int(round(self.sample_rate / self.frame_rate)) if self.frame_rate else 0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MIMI_MODEL_ID,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> "MimiCodec":
        model = MimiModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
        return cls(model, device)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.to(self.device)
        with torch.inference_mode():
            audio, _ = self.model.decode(codes, return_dict=False)
            return torch.clamp(audio, -1.0, 1.0)

    def decode_stateful(self, codes: torch.Tensor, past_key_values=None):
        """Stateful decode for streaming — carries decoder KV cache between calls.

        Each call processes `codes` incrementally using context from previous calls,
        eliminating the boundary discontinuities that cause clicking when decoding
        frame-by-frame without state.

        Returns:
            waveform: torch.Tensor, shape [batch, 1, N_samples], float32 in [-1, 1]
            new_past_key_values: pass into the next call to maintain continuity
        """
        codes = codes.to(self.device)
        with torch.inference_mode():
            audio, new_pkv = self.model.decode(
                codes,
                decoder_past_key_values=past_key_values,
                return_dict=False,
            )
            return torch.clamp(audio, -1.0, 1.0), new_pkv

    def encode(self, audio: torch.Tensor, *, return_dict: bool = False):
        audio = audio.to(self.device)
        with torch.inference_mode():
            return self.model.encode(audio, return_dict=return_dict)
