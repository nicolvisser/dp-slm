dependencies = [
    "torch",
    "torchaudio",
    "numpy",
    "k2",
    "simple-parsing",
    "xformers",
    "tqdm",
]

import torch

from dpslm.dpslm import DPSLMPipeline
from dpslm.dpwfst.dpwfst import DPWFSTQuantizer
from dpslm.ellav.config import ELLAVModelArgs
from dpslm.ellav.model import ELLAVGARModel
from dpslm.ellav.tokenizer import ELLAVTokenizer
from dpslm.wavlm.WavLM import WavLM, WavLMConfig
from dpslm.wavtokenizer.decoder.pretrained import WavTokenizer, WavTokenizerArgs

release_url = "https://github.com/nicolvisser/dp-slm/releases/download/v0.1/"

codebook_urls = {
    100: release_url + "codebook-layer-11-km-100-8b2b254e.pt",
    200: release_url + "codebook-layer-11-km-200-55b06314.pt",
    500: release_url + "codebook-layer-11-km-500-2c2dee95.pt",
    1000: release_url + "codebook-layer-11-km-1000-db31d361.pt",
    2000: release_url + "codebook-layer-11-km-2000-af7a6260.pt",
}

ellav_urls = {
    (100, 0): release_url + "ellav-layer-11-km-100-lmbda-0-866a8bd2.pt",
    (100, 600): release_url + "ellav-layer-11-km-100-lmbda-600-0660515a.pt",
    (100, 1500): release_url + "ellav-layer-11-km-100-lmbda-1500-de10c252.pt",
    (100, 3000): release_url + "ellav-layer-11-km-100-lmbda-3000-736162b6.pt",
    (100, 5000): release_url + "ellav-layer-11-km-100-lmbda-5000-761084c3.pt",
    (100, 9000): release_url + "ellav-layer-11-km-100-lmbda-9000-b94eeffe.pt",
    (200, 0): release_url + "ellav-layer-11-km-200-lmbda-0-47e52fa2.pt",
    (200, 700): release_url + "ellav-layer-11-km-200-lmbda-700-244ce4bd.pt",
    (200, 1500): release_url + "ellav-layer-11-km-200-lmbda-1500-41002b80.pt",
    (200, 3000): release_url + "ellav-layer-11-km-200-lmbda-3000-5f7bc27b.pt",
    (200, 5000): release_url + "ellav-layer-11-km-200-lmbda-5000-e20f9749.pt",
    (200, 7500): release_url + "ellav-layer-11-km-200-lmbda-7500-d1aa0f3d.pt",
    (500, 0): release_url + "ellav-layer-11-km-500-lmbda-0-7907a9f1.pt",
    (500, 600): release_url + "ellav-layer-11-km-500-lmbda-600-0f09a9f2.pt",
    (500, 1500): release_url + "ellav-layer-11-km-500-lmbda-1500-22cceeec.pt",
    (500, 2800): release_url + "ellav-layer-11-km-500-lmbda-2800-37a1ef74.pt",
    (500, 4500): release_url + "ellav-layer-11-km-500-lmbda-4500-2acda6bd.pt",
    (500, 7000): release_url + "ellav-layer-11-km-500-lmbda-7000-ee2555b7.pt",
    (1000, 0): release_url + "ellav-layer-11-km-1000-lmbda-0-d735bbe7.pt",
    (1000, 600): release_url + "ellav-layer-11-km-1000-lmbda-600-17a1ac04.pt",
    (1000, 1400): release_url + "ellav-layer-11-km-1000-lmbda-1400-d46fecc3.pt",
    (1000, 2500): release_url + "ellav-layer-11-km-1000-lmbda-2500-88804dda.pt",
    (1000, 3800): release_url + "ellav-layer-11-km-1000-lmbda-3800-3a00169b.pt",
    (1000, 6000): release_url + "ellav-layer-11-km-1000-lmbda-6000-0bee8e1b.pt",
}

num_neighbors_by_k = {
    100: 5,
    200: 10,
    500: 25,
    1000: 50,
    2000: 100,
}


def wavlm_large(
    map_location="cuda",
    progress=True,
) -> WavLM:
    checkpoint = torch.hub.load_state_dict_from_url(
        release_url + "wavlm-large-6fb4b3c3.pt",
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    model = WavLM(WavLMConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model.to(map_location)
    model.eval()
    return model


def dpwfst_quantizer(
    K: int,
    map_location="cuda",
    progress=True,
) -> DPWFSTQuantizer:
    assert (
        K in codebook_urls.keys()
    ), "Pretrained models only available for K values in: {}".format(
        list(codebook_urls.keys())
    )

    checkpoint = torch.hub.load_state_dict_from_url(
        codebook_urls[K],
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    model = DPWFSTQuantizer.from_codebook(checkpoint["codebook"])
    model.to(map_location)
    model.eval()

    return model


def ellav(
    K: int,
    lmbda: int,
    map_location="cuda",
    progress=True,
) -> ELLAVGARModel:
    assert (
        K,
        lmbda,
    ) in ellav_urls.keys(), "Pretrained models only available for (K, lmbda) values in: {}".format(
        list(ellav_urls.keys())
    )

    state = torch.hub.load_state_dict_from_url(
        ellav_urls[(K, lmbda)],
        map_location=map_location,
        progress=progress,
        check_hash=True,
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


def wavtokenizer_small_600_24k_4096(
    map_location="cuda",
    progress: bool = True,
) -> WavTokenizer:
    checkpoint = torch.hub.load_state_dict_from_url(
        release_url + "wavtokenizer-small-600-24k-4096-ed05f5d2.pt",
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    model = WavTokenizer(WavTokenizerArgs.from_dict(checkpoint["args"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(map_location)
    model.eval()
    return model


def dpslm(
    K,
    lmbda,
    map_location="cuda",
    progress=True,
) -> DPSLMPipeline:
    pipeline = DPSLMPipeline(
        wavlm=wavlm_large(map_location, progress),
        dpwfst=dpwfst_quantizer(K, map_location, progress),
        ellav=ellav(K, lmbda, map_location, progress),
        wavtokenizer=wavtokenizer_small_600_24k_4096(map_location, progress),
        layer_idx=11,
        lmbda=lmbda,
        num_neighbors=num_neighbors_by_k[K],
    )
    pipeline.to(map_location)
    pipeline.eval()
    return pipeline
