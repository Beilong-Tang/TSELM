import os.path as op
import os
import glob
import torch
import tqdm

BASE_PATH = op.abspath(op.dirname(__file__))


def p(*args):
    return op.join(BASE_PATH, *args)


os.makedirs(p("list"), exist_ok=True)
os.makedirs(p("list", "train"), exist_ok=True)
os.makedirs(p("list", "dev"), exist_ok=True)
os.makedirs(p("list", "test"), exist_ok=True)

train_audio = [p("librispeech", i) for i in ["train-clean-100", "train-clean-360"]]
dev_audio = "libri2mix_dev"
test_audio = "libri2mix_test"


def generate_training_pt():
    print("generate training scp")
    spk_dict = {}
    for t in train_audio:
        audio_files = glob.glob(op.join(t, "*", "*", "*.flac"))
        for a in tqdm.tqdm(audio_files):
            spk = a.split("/")[-3]
            if spk in spk_dict:
                spk_dict[spk] = spk_dict[spk] + [a]
            else:
                spk_dict[spk] = [a]
    torch.save(spk_dict, p("list", "train", "train_100_360.pt"))
    print("done!")


def generate_scp(dataset_name: str, type: str):
    print(f"generating scp for {dataset_name} of type {type}")
    files = sorted(glob.glob(p(dataset_name, type, "*.wav")))
    res = [f"{i.split('/')[-1]} {i}\n" for i in files]
    with open(p("list", dataset_name, f"{type}.scp"), "w") as f:
        for r in res:
            f.write(r)
    print("done")


# generate_training_pt()

for t in ["aux_s1", "mix_clean", "s1"]:
    for d in [dev_audio, test_audio]:
        generate_scp(d, t)
print("All scp files are generated successfully!")
