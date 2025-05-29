import torch
import torchaudio

k = 500
lmbda = 4500

# load the wavlm encoder
wavlm, extract_features = torch.hub.load(
    "nicolvisser/WavLM-codebooks", "wavlm_large", trust_repo=True
)
wavlm.to("cuda")

# load the codebook and quantizer
codebook = torch.hub.load(
    "nicolvisser/WavLM-codebooks", "codebook", layer=11, k=k, trust_repo=True
)
quantizer = torch.hub.load(
    "nicolvisser/dpdp",
    "dpdp_quantizer_from_codebook",
    codebook=codebook,
    lmbda=lmbda,
    num_neighbors=None,
    trust_repo=True,
).cuda()
quantizer.to("cuda")

ulm, ulm_tokenizer = torch.hub.load(
    "nicolvisser/Mistral-ULM",
    "ulm_wavlm_layer_11_dpdp_hours_1k_steps_10k",
    k=k,
    lmbda=lmbda,
    trust_repo=True,
    force_reload=True,
)
ulm.to("cuda")


with torch.inference_mode():
    for wav_path in ["data/manufacture.wav", "data/manufelture.wav"]:
        # load your audio
        wav, sr = torchaudio.load(wav_path)
        # extract features
        features = extract_features(wavlm, wav, sr, layer=11)
        # quantize using DPDP
        quantized_features, units_duped = quantizer(features)
        # deduplicate
        units_deduped = torch.unique_consecutive(units_duped)
        # convert to ulm tokens (adds a BOS token and shifts the units by 1)
        ulm_input = ulm_tokenizer.encode(units_deduped.tolist()).cuda()
        # compute log likelihood
        ulm_ll = ulm.loglikelihood(ulm_input)

        print(f"{wav_path}: {ulm_ll}")
