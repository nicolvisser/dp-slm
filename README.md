# DP-SLM

Official repository for `Spoken Language Modeling with Duration-Penalized Self-Supervised Units` (arxiv link will be added soon)

![DP-SLM pipeline](pipeline.svg)

Each component in the system can be found in one of the following repositories:

- [WavLM Encoder](https://github.com/nicolvisser/WavLM-codebooks)
- [DPDP Quantizer](https://github.com/nicolvisser/dpdp)
- [Unit Language Model](https://github.com/nicolvisser/Mistral-ULM)
- [ELLA-V Acoustic Model](https://github.com/nicolvisser/ELLA-V)
- [WavTokenizer Decoder](https://github.com/nicolvisser/WavTokenizer)

## Usage

First install the requirements:

```sh
pip install torch torchvision torchaudio xformers simple-parsing tqdm
```

Resynthesis example:

```py
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
```

Note:

- `k` is the codebook size ($K$ in the paper).
- `lmbda` controls the coarseness of the the units ($\lambda$ in the paper)
  - higher values will result in shorter sequence lengths of the resulting units

You can change the values of `k` and `lmbda` but they must be one of the following combinations (for which we have trained the downstream models):

```py
 valid_combinations = [
    # (k, lmbda)
    (100, 0),
    (100, 600),
    (100, 1500),
    (100, 3000),
    (100, 5000),
    (100, 9000),

    (200, 0),
    (200, 700),
    (200, 1500),
    (200, 3000),
    (200, 5000),
    (200, 7500),

    (500, 0),
    (500, 600),
    (500, 1500),
    (500, 2800),
    (500, 4500),
    (500, 7000),

    (1000, 0),
    (1000, 600),
    (1000, 1400),
    (1000, 2500),
    (1000, 3800),
    (1000, 6000),
]
```
You will find that there are many mispronunciations when the codebook size (`k`) is small and using coarser units (`lmbda > 0`).

However, as the codebook size increases, we can push the coarseness (`lmbda`) quite far without introducing significant mispronunciations.

![Word error rates against bitrate](wer_vs_bitrate.svg)