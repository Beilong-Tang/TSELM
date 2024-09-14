import os.path as op
import os
import glob
import torch
import tqdm
import argparse

BASE_PATH = "."


def p(*args):
    return op.join(BASE_PATH, *args)


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


def generate_scp(dataset_name: str, type: str, name: str):
    print(f"generating scp for {name} of type {type}")
    files = sorted(glob.glob(p(dataset_name, type, "*.wav")))
    res = [f"{i.split('/')[-1]} {i}\n" for i in files]
    with open(p("list", name, f"{type}.scp"), "w") as f:
        for r in res:
            f.write(r)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ls", "--librispeech", type=str, required=True)
    parser.add_argument("-lm_dev", "--libri2mix_dev", type=str, required=True)
    parser.add_argument("-lm_test", "--libri2mix_test", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    BASE_PATH = args.output

    if op.exists(p("list")):
        raise FileExistsError(
            "Please choose another folder that does not have folder 'list' as output folder."
        )
    os.makedirs(p("list"), exist_ok=True)
    os.makedirs(p("list", "train"), exist_ok=True)
    os.makedirs(p("list", "libri2mix_dev"), exist_ok=True)
    os.makedirs(p("list", "libri2mix_test"), exist_ok=True)

    train_audio = [
        op.join(args.librispeech, i) for i in ["train-clean-100", "train-clean-360"]
    ]
    dev_audio = args.libri2mix_dev
    test_audio = args.libri2mix_test

    generate_training_pt()

    for t in ["aux_s1", "mix_clean", "s1"]:
        for d, n in [(dev_audio, "libri2mix_dev"), (test_audio, "libri2mix_test")]:
            generate_scp(d, t, n)
    print("All scp files are generated successfully!")
