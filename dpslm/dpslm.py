import torch
import torchaudio
from torch import nn

from .dpwfst.dpwfst import DPWFSTQuantizer
from .ellav.config import ELLAVModelArgs
from .ellav.model import ELLAVGARModel
from .ellav.tokenizer import ELLAVTokenizer
from .wavlm.WavLM import WavLM, WavLMConfig
from .wavtokenizer.decoder.pretrained import WavTokenizer, WavTokenizerArgs

# available models for ELLA-V GAR
LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER = {
    11: {
        100: [0, 600, 1500, 3000, 5000, 9000],
        200: [0, 700, 1500, 3000, 5000, 7500],
        500: [0, 600, 1500, 2800, 4500, 7000],
        1000: [0, 600, 1400, 2500, 3800, 6000],
        2000: [0, 600, 1300, 2400, 3600, 5500],
    }
}
NUM_NEIGHBORS_BY_NUM_CLUSTERS_BY_LAYER = {
    11: {
        100: 5,
        200: 10,
        500: 25,
        1000: 50,
        2000: 100,
    }
}


def get_wavlm_large(
    map_location="cuda",
    progress=True,
) -> WavLM:
    checkpoint = torch.load(
        "/mnt/wsl/nvme/code/dpslm/checkpoints/wavlm-large.pt",
        weights_only=True,
    )
    model = WavLM(WavLMConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model.to(map_location)
    model.eval()
    return model


def get_dpwfst_quantizer(
    layer_idx: int,
    K: int,
    map_location="cuda",
    progress=True,
) -> DPWFSTQuantizer:
    assert (
        layer_idx in LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER
    ), "Pretrained models only available for layers: {}".format(
        LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER.keys()
    )
    assert (
        K in LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER[layer_idx]
    ), "Given layer={}, pretrained models only available for K values: {}".format(
        layer_idx, LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER[layer_idx].keys()
    )

    checkpoint = torch.load(
        f"/mnt/wsl/nvme/code/dpslm/checkpoints/codebook-layer-{layer_idx}-km-{K}.pt",
        weights_only=True,
    )
    model = DPWFSTQuantizer.from_codebook(checkpoint["codebook"])
    model.to(map_location)
    model.eval()
    return model


def get_ellav(
    layer=11,
    K=100,
    lmbda=0,
    map_location="cuda",
    progress=True,
):
    assert (
        layer in LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER
    ), "Pretrained models only available for layers: {}".format(
        LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER.keys()
    )
    assert (
        K in LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER[layer]
    ), "Given layer={}, pretrained models only available for K values: {}".format(
        layer, LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER[layer].keys()
    )
    assert (
        lmbda in LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER[layer][K]
    ), "Given layer={} and K={}, pretrained models only available for lmbda values: {}".format(
        layer, K, LMBDA_LIST_BY_NUM_CLUSTERS_BY_LAYER[layer][K]
    )

    state = torch.load(
        f"/mnt/wsl/nvme/code/dpslm/checkpoints/ellav-layer-{layer}-km-{K}-lmbda-{lmbda}.pt",
        map_location=map_location,
        weights_only=True,
    )
    model = ELLAVGARModel(
        model_args=ELLAVModelArgs.from_dict(state["model_args"]),
        tokenizer=ELLAVTokenizer.from_dict(state["tokenizer"]),
    )
    model.load_state_dict(state["model"])
    model.to(map_location)
    model.eval()
    return model


def get_wavtokenizer_small_600_24k_4096(
    map_location="cuda",
    progress: bool = True,
) -> WavTokenizer:
    checkpoint = torch.load(
        "/mnt/wsl/nvme/code/dpslm/checkpoints/wavtokenizer-small-600-24k-4096.pt",
        weights_only=True,
    )
    model = WavTokenizer(WavTokenizerArgs.from_dict(checkpoint["args"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(map_location)
    model.eval()
    return model


class DPSLMPipeline(nn.Module):
    def __init__(
        self,
        layer_idx: int = 11,
        K: int = 100,
        lmbda: int = 0,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.K = K
        self.lmbda = lmbda
        self.num_neighbors = NUM_NEIGHBORS_BY_NUM_CLUSTERS_BY_LAYER[self.layer_idx][
            self.K
        ]

        self.wavlm = get_wavlm_large()
        self.dpwfst = get_dpwfst_quantizer(self.layer_idx, self.K)
        self.ellav = get_ellav(self.layer_idx, self.K, self.lmbda)
        self.wavtokenizer = get_wavtokenizer_small_600_24k_4096()

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
