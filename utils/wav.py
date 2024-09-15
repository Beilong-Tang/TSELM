import torch
import random
import torch.nn.functional as F


def truc_wav(*audio: torch.Tensor, length):
    """
    Given a list of audio with the same length as arguments, chunk the audio into a given length.
    Note that all the audios will be chunked using the same offset

    Args:
        audio: the list of audios to be chunked, should have the same length with shape [T] (1D)
        length: the length to be chunked into, if length is None, return the original audio
    Returns:
        A list of chuncked audios
    """
    audio_len = audio[0].size(0)  # [T]
    res = []
    if length == None:
        for a in audio:
            res.append(a)
        return res[0] if len(res) == 1 else res
    if audio_len > length:
        offset = random.randint(0, audio_len - length - 1)
        for a in audio:
            res.append(a[offset : offset + length])
    else:
        for a in audio:
            res.append(F.pad(a, (0, length - a.size(0)), "constant"))
    return res[0] if len(res) == 1 else res


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
