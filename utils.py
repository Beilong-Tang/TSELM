import torch
import torch.nn.functional as F
import random

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None


def truc_wav(audio: torch.Tensor, length=64000):
    """
    audio: [T], torch audio
    length: the point length of audio to be chuncked
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


def dict_to_str(dictionary):
    res = ""
    for key, value in dictionary.items():
        res += f"{key} : {value}, "
    return res


def save(path, content, max_ckpt=1):
    # if len(files_path) >= max_ckpt:
    if max_ckpt == -1:
        ##save
        torch.save(content, path)
        return
    if max_ckpt == None:
        max_ckpt = 1
    dirname = op.dirname(path)
    files = sorted(
        [f for f in os.listdir(dirname) if (f.endswith(".pth") and "best" not in f)],
        key=lambda x: int(re.search(r"[0-9]+", x).group()),
    )
    files_path = [op.join(dirname, f) for f in files]
    print(f"files path:  {files_path}")
    if len(files_path) >= max_ckpt:
        try:
            os.remove(files_path[0])
        except FileNotFoundError as e:
            print("saving error")
            print(e)
    torch.save(content, path)

def get_source_list(file_path: str, ret_name=False):
    files = []
    names = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            l = line.replace("\n", "").split(" ")
            name = l[0]
            path = l[-1]
            files.append(path)
            names.append(name)
    if ret_name:
        return names, files
    return files


