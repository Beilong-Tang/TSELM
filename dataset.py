import torch
from torch.utils.data import Dataset
import random
import torchaudio
from utils.wav import truc_wav
from utils.load_scp import get_source_list


def _activelev(*args):
    """
    need to update like matlab
    """
    res = torch.concat(list(args))
    return torch.max(torch.abs(res))


def unify_energy(*args):
    max_amp = _activelev(*args)
    mix_scale = 1.0 / max_amp
    return [x * mix_scale for x in args]


def generate_target_audio(spk1, spk2, regi, snr=5):
    """
    spk 1: T1
    spk 2: T2
    regi: T3
    """
    spk1, spk2 = unify_energy(spk1, spk2)
    snr_1 = random.random() * snr / 2
    snr_2 = -snr_1
    spk1 = spk1 * 10 ** (snr_1 / 20)
    spk2 = spk2 * 10 ** (snr_2 / 20)
    mix = spk1 + spk2
    mix, clean, regi = unify_energy(mix, spk1, regi)
    return (mix, clean, regi)


class TargetDMDataset(Dataset):
    def __init__(
        self,
        scp_path,
        rank,
        epoch_num=100000,
        mix_length=48080,
        regi_length=64080,
    ):
        """
        Initialize the Target DM Dataset.
        This class is used for dynamic mixing of target speech extraction dataset


        Args:
            scp_path: the .pt file which saves a dictionary of speker_name -> list of path to source files
            epoch_num: specifcy how many data to be considered as one epoch
            mix_length: the length of the mixing speech and clean speech
            regi_length: the length of the register speech
        """
        self.speaker_dict = torch.load(scp_path)
        self.length = epoch_num
        self.mix_length = mix_length
        self.rank = rank
        self.regi_length = regi_length
        self.num = 3
        self.ct = 0
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        keys_list = list(self.speaker_dict.keys())
        speaker_1 = random.choice(keys_list)
        speaker_2 = random.choice(keys_list)
        if self.ct < self.num:
            print(f"rank {self.rank} get spk1 {speaker_1}")
            self.ct += 1
        while speaker_2 == speaker_1:
            speaker_2 = random.choice(keys_list)
        spk1 = random.choice(self.speaker_dict[speaker_1])
        regi = random.choice(self.speaker_dict[speaker_1])
        while regi == spk1:
            regi = random.choice(self.speaker_dict[speaker_1])
        spk2 = random.choice(self.speaker_dict[speaker_2])
        spk1_audio = torchaudio.load(spk1)[0].squeeze(0)  # [T]
        spk2_audio = torchaudio.load(spk2)[0].squeeze(0)
        regi_audio = torchaudio.load(regi)[0].squeeze(0)
        if self.regi_length is not None:
            regi_audio = truc_wav(regi_audio, length=self.regi_length)
        else:
            regi_audio = truc_wav(regi_audio, length=self.mix_length)
        spk1_audio = truc_wav(spk1_audio, length=self.mix_length)
        spk2_audio = truc_wav(spk2_audio, length=self.mix_length)
        mix, clean, regi = generate_target_audio(
            spk1_audio, spk2_audio, regi_audio
        )
        return mix, clean, regi


class TargetDataset(Dataset):
    def __init__(
        self,
        mix_path: str,
        regi_path: str,
        clean_path: str,
        rank: int,
        mix_length=48080,
        regi_length=64080,
    ):
        """
        The regular dataset for target speaker extraction.
        Has to provide three .scp files that have mix_path, regi_path, clean_path aligned
        """
        self.mix_list = get_source_list(mix_path)
        self.regi_list = get_source_list(regi_path)
        self.clean_list = get_source_list(clean_path)
        self.mix_length = mix_length
        self.regi_length = regi_length
        self.rank = rank
        pass

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix_path = self.mix_list[idx]
        regi_path = self.regi_list[idx]
        clean_path = self.clean_list[idx]
        mix_audio = torchaudio.load(mix_path)[0].squeeze(0)  # [T]
        regi_audio = torchaudio.load(regi_path)[0].squeeze(0)
        clean_audio = torchaudio.load(clean_path)[0].squeeze(0)
        mix_audio, clean_audio = truc_wav(
            mix_audio, clean_audio, length=self.mix_length
        )
        regi_audio = truc_wav(regi_audio, length=self.regi_length)
        return mix_audio, clean_audio, regi_audio, mix_path, clean_path, regi_path