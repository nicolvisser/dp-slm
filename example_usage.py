import torch
import torchaudio

pipeline = torch.hub.load(
    "nicolvisser/dp-slm",
    "dpslm",
    K=100,
    lmbda=0,
)
pipeline.eval()

wav, sr = torchaudio.load("sample.flac")
wav = wav.cuda()

units = pipeline.encode_units(wav)
wav, sr = pipeline.generate_audio(
    units,
    top_p=0.8,
    temperature=1.0,
    max_phoneme_duration=0.4,
    show_progress=True,
)

torchaudio.save("resynthesized_sample.wav", wav.cpu(), sr)
