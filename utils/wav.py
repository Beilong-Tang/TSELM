import torch
import torchaudio as ta
import random
import torch.nn.functional as F
from typing import Optional


## TODO: this function not finished
def load_audio(path, sr: Optional[int] = None, flat=False):
    audio, rate = ta.load(path)
    if sr is not None:
        assert rate == sr
    if flat:
        return audio.unsqueeze(0)


## TODO: this function not finished
def squeeze(*audio):
    res = []
    for a in audio:
        res.append(a.squeeze())


def truc_wav(audio: torch.Tensor, length=64000):
    """
    audio: [T], torch
    length: the point length of audio
    ---
    random chunk the audio into the duration of 4
    ---
    """
    if audio.size(0) > length:
        offset = random.randint(0, audio.size(0) - length - 1)
        audio = audio[offset : offset + length]
    else:
        audio = F.pad(audio, (0, length - audio.size(0)), "constant")
    return audio


def split_audio(audio, length=48000, pad_last=True):
    """
    audio: [T]
    if pad_last, then the last element of the audio will be padded the same length in the array
    return a list of audio each with shape [T=length]
    """
    audio_array = []
    for start in range(0, audio.size(0), length):
        clip = audio[start : start + length]
        audio_array.append(clip)
    if pad_last and audio_array[-1].size(0) != length:
        audio_array[-1] = F.pad(
            audio_array[-1], (0, length - audio_array[-1].size(0)), "constant"
        )
    return audio_array
