import torch
import torchaudio
from IPython.display import Audio, display

k = 500
lmbda = 4500

# load the wavlm encoder
wavlm, extract_features = torch.hub.load("nicolvisser/WavLM-codebooks", "wavlm_large", trust_repo=True)
wavlm.to("cuda")

# load the codebook and quantizer
codebook = torch.hub.load("nicolvisser/WavLM-codebooks", "codebook", layer=11, k=k, trust_repo=True )
quantizer = torch.hub.load("nicolvisser/dpdp", "dpdp_quantizer_from_codebook", codebook=codebook, lmbda=lmbda, num_neighbors=None, trust_repo=True ).cuda()
quantizer.to("cuda")

# load the ELLA-V acoustic model
ellav = torch.hub.load("nicolvisser/ELLA-V", "ellav_units_to_wavtokenizer", k=k, lmbda=lmbda, trust_repo=True)
ellav.to("cuda")

# load the WavTokenizer vocoder
wavtokenizer, _, vocode = torch.hub.load("nicolvisser/WavTokenizer", "small_600_24k_4096", trust_repo=True, force_reload=True)
wavtokenizer.to("cuda")

# load your audio
wav, sr = torchaudio.load("1272-128104-0000.flac")

# encode and resynthesize
with torch.inference_mode():
    # extract features
    features = extract_features(wavlm, wav, sr, layer=11)

    # quantize using DPDP
    quantized_features, units_duped = quantizer(features)

    # deduplicate
    units_deduped = torch.unique_consecutive(units_duped)

    # perform acoustic modeling ("add a voice")
    prompt = [f"u{unit}" for unit in units_deduped.tolist()]
    codec_ids_list, finished = ellav.generate(
        prompts=[prompt] * 3, # generate 3 examples
        max_tokens=1000,
        max_codec_tokens_per_phone=10,
        temperature=1.0,
        top_p=0.8,
        chunk_size=None,
        progress=True,
    )

    # vocode and display waveforms
    for i, codec_ids in enumerate(codec_ids_list):
        wav_, sr_ = vocode(wavtokenizer, codec_ids[None, None, :])
        display(Audio(wav_.cpu().numpy(), rate=sr_))
        torchaudio.save(f"resynth_{i}.wav", wav_.cpu(), sr_)