import os.path as op
import os
import glob
# import torch
import tqdm

BASE_PATH = op.abspath(op.dirname(__file__))

def p(*args):
    return op.join(BASE_PATH, *args)

os.makedirs(p("list"), exist_ok=True)

train_audio = [p(i) for i in ["train-clean-100", "train-clean-360"]]
dev_audio = "dev-clean"
test_audio = "test-clean"

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

def generate_scp(name:str):
    print(f"generating scp for {name}")
    

    print("done!")
    pass


# generate_training_pt()
generate_scp()