## inference on libri2mix test set
import argparse
import tqdm
import os.path as op
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import random_split
from hyperpyyaml import load_hyperpyyaml
from dataset import TargetDataset

SEED = 1234


def main(rank, args):
    device = args.gpus[rank % len(args.gpus)]
    world_size = args.proc
    torch.cuda.set_device(device)
    scp = args.scp_dir
    mix_scp = op.join(scp, "mix_clean.scp")
    s1_scp = op.join(scp, "s1.scp")
    aux_s1_scp = op.join(scp, "aux_s1.scp")
    dataset = TargetDataset(
        mix_scp, aux_s1_scp, s1_scp, -1, mix_length=None, regi_length=None
    )
    if world_size != 1:
        generator = torch.Generator().manual_seed(SEED)
        num_samples = len(dataset)
        split_size = num_samples // world_size
        remainder = num_samples % world_size
        split_sizes = [split_size] * world_size
        for i in range(remainder):
            split_sizes[i] += 1
        splits = random_split(dataset, split_sizes, generator=generator)
        dataset = splits[rank]
        print(f"rank {rank} get dataset of length {len(dataset)} on device {device}")
    with open(args.config_path, "r") as f:
        config = load_hyperpyyaml(f)
    model: nn.Module = config.get("model")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.cuda(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    with torch.no_grad():
        for mix, _, regi, mix_path, _, _ in tqdm.tqdm(dataset):
            mix, regi = mix.to(device), regi.cuda(device)
            mix, regi = mix.unsqueeze(0), regi.unsqueeze(0)  # [1, T]
            output, _ = model.inference(mix, regi)  # [1,T]
            output = output.cpu()
            name = mix_path.split("/")[-1]
            torchaudio.save(op.join(args.output, name), output, 16000)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-scp", "--scp_dir", type=str, required=True)
    parser.add_argument("-config", "--config_path", type=str, required=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, required=True)
    parser.add_argument("-output", "--output", type=str, required=True)
    parser.add_argument(
        "-gpus",
        "--gpus",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="The gpus to run the ddp.",
    )
    parser.add_argument(
        "-proc",
        "--proc",
        type=int,
        default=8,
    )
    args = parser.parse_args()
    if args.proc != 1:
        mp.spawn(main, args=(args,), nprocs=args.proc, join=True)
    else:
        main(0, args)
