import torch
import torchaudio
from torch import nn

from .dpwfst.dpwfst import DPWFSTQuantizer
from .ellav.config import ELLAVModelArgs
from .ellav.model import ELLAVGARModel
from .ellav.tokenizer import ELLAVTokenizer
from .wavlm.WavLM import WavLM, WavLMConfig
from .wavtokenizer.decoder.pretrained import WavTokenizer, WavTokenizerArgs


class DPSLMPipeline(nn.Module):
    def __init__(
        self,
        wavlm: WavLM,
        dpwfst: DPWFSTQuantizer,
        ellav: ELLAVGARModel,
        wavtokenizer: WavTokenizer,
        layer_idx: int = 11,
        lmbda: int = 0,
        num_neighbors: int = 1,
    ):
        super().__init__()
        self.wavlm = wavlm
        self.dpwfst = dpwfst
        self.ellav = ellav
        self.wavtokenizer = wavtokenizer
        self.layer_idx = layer_idx
        self.lmbda = lmbda
        self.num_neighbors = num_neighbors

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_features(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.ndim == 2, "wav must be a 2D tensor, with shape (1, T)"
        features, _ = self.wavlm.extract_features(wav, output_layer=self.layer_idx)
        features = features.squeeze(0)
        return features

    @torch.no_grad()
    def encode_units(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.ndim == 2, "wav must be a 2D tensor, with shape (1, T)"
        features, _ = self.wavlm.extract_features(wav, output_layer=self.layer_idx)
        features = features.squeeze(0)
        units = self.dpwfst(
            features, lmbda=self.lmbda, num_neighbors=self.num_neighbors
        )
        return units

    @torch.no_grad()
    def generate_codec(
        self,
        units: torch.Tensor,
        top_p: float = 0.8,
        temperature: float = 1.0,
        max_phoneme_duration: float = 0.4,
        show_progress: bool = True,
    ) -> torch.Tensor:
        assert units.ndim == 1, "units must be a 1D tensor"
        units_str = [f"u{unit.item()}" for unit in units]
        codes = self.ellav.generate(
            [units_str],
            top_p=top_p,
            temperature=temperature,
            max_phoneme_duration=max_phoneme_duration,
            show_progress=show_progress,
        )
        codes = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
        return codes

    @torch.no_grad()
    def decode_codec(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.ndim == 3, "codes must be a 3D tensor, with shape (1, 1, T)"
        features = self.wavtokenizer.codes_to_features(codes)
        bandwidth_id = torch.tensor([0], device=self.device)
        wav = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        sr = 24000
        return wav, sr

    @torch.no_grad()
    def generate_audio(
        self,
        units: torch.Tensor,
        top_p: float = 0.8,
        temperature: float = 1.0,
        max_phoneme_duration: float = 0.4,
        show_progress: bool = True,
    ) -> torch.Tensor:
        codes = self.generate_codec(
            units,
            top_p=top_p,
            temperature=temperature,
            max_phoneme_duration=max_phoneme_duration,
            show_progress=show_progress,
        )
        wav, sr = self.decode_codec(codes)
        return wav, sr
