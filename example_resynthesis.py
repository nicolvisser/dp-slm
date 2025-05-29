import torch
import torchaudio

# Load waveform:

wav, sr = torchaudio.load("data/1272-128104-0000.flac")

#Pick a $K$ and $\lambda$ value from the paper:

k = 500
lmbda = 4500

# Encode features:

wavlm, extract_features = torch.hub.load(
    "nicolvisser/WavLM-codebooks",
    "wavlm_large",
    trust_repo=True,
)
wavlm.to("cuda")

with torch.inference_mode():
    features = extract_features(wavlm, wav, sr, layer=11)

# Quantize to a $K$-means codebook with DPDP:

codebook = torch.hub.load(
    "nicolvisser/WavLM-codebooks",
    "codebook",
    layer=11,
    k=k,
    trust_repo=True,
)
quantizer = torch.hub.load(
    "nicolvisser/dpdp",
    "dpdp_quantizer_from_codebook",
    codebook=codebook,
    lmbda=lmbda,
    num_neighbors=int(0.05*k),
    trust_repo=True
)
quantizer.to("cuda")

with torch.inference_mode():
    quantized_features, units_duped = quantizer(features)

# Deduplicate the units:

units_deduped = torch.unique_consecutive(units_duped)

# Compute the log-likelihood under the LM:

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
    ulm_input = ulm_tokenizer.encode(units_deduped.tolist()).cuda()
    ulm_ll = ulm.loglikelihood(ulm_input)

# Generate acoustic codes from the deduped units:

ellav = torch.hub.load(
    "nicolvisser/ELLA-V",
    "ellav_units_to_wavtokenizer",
    k=k,
    lmbda=lmbda,
    trust_repo=True
)
ellav.to("cuda")

with torch.inference_mode():
    prompt = [f"u{unit}" for unit in units_deduped.tolist()]
    codec_ids_list, finished = ellav.generate(
        prompts=[prompt] * 3, # generate 3 examples
        max_tokens=1000,
        max_codec_tokens_per_phone=10,
        temperature=1.0,
        top_p=0.8,
    )

# Vocode to waveforms:

wavtokenizer, _, vocode = torch.hub.load(
    "nicolvisser/WavTokenizer",
    "small_600_24k_4096",
    trust_repo=True,
    force_reload=True
)
wavtokenizer.to("cuda")

with torch.inference_mode():
    for i, codec_ids in enumerate(codec_ids_list):
        wav_, sr_ = vocode(wavtokenizer, codec_ids[None, None, :])
        torchaudio.save(f"resynth_{i}.wav", wav_.cpu(), sr_)