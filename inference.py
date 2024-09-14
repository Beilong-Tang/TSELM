import sys
import os

base_dir = os.getcwd()[: os.getcwd().find("exp")]
sys.path.append(base_dir)
sys.path.append(os.getcwd())
from utils.io import make_path
from utils.load_model import load_ckpt
import tqdm
import random
from torch.utils.data import Dataset
from torch.utils.data import random_split
from hyperpyyaml import load_hyperpyyaml
import argparse
import os.path as op
from utils.env import AttrDict
import torch.multiprocessing as mp
import torch

SEED = 1234


def main(rank, args):
    config = args.config
    with open(config, "r") as f:
        config = load_hyperpyyaml(f)
    h = AttrDict(**config)
    dataset = h.dataset
    ## split dataset
    world_size = args.num_proc
    device = 0
    if world_size:
        generator = torch.Generator().manual_seed(SEED)
        num_samples = len(dataset)
        split_size = num_samples // world_size
        remainder = num_samples % world_size
        split_sizes = [split_size] * world_size
        for i in range(remainder):
            split_sizes[i] += 1
        splits = random_split(dataset, split_sizes, generator=generator)
        h.dataset = splits[rank]
        print(f"rank {rank} get dataset of length {len(h.dataset)}")
        device = rank % args.num_gpu
    torch.cuda.set_device(device)
    if args.name is None:
        if h.name is not None:
            args.name = h.name
    h.output_dir = (
        op.join(h.output_dir, args.name) if h.name is not None else h.output_dir
    )
    make_path(h.output_dir)
    inference = h.inference(h)
    inference.inference()
    pass


class AbsInference:
    def __init__(self, h):
        self.h = h
        self.dataset: Dataset = h.dataset
        self.out_num: int = h.output_num
        self.model = h.model
        self.model_config = h.model_config
        if self.model_config is not None:
            with open(self.model_config, "r") as f:
                config = load_hyperpyyaml(f)
            self.model = config.get("model")
        self.output_dir = h.output_dir
        load_ckpt(
            self.model,
            h.ckpt_path,
            strict=True if h.strict is None else h.strict,
        )

    def infer(self, data, idx):
        raise NotImplementedError()

    def inference(self):
        if self.out_num is None:
            for idx, data in enumerate(tqdm.tqdm(self.dataset)):
                self.infer(data, idx)
        else:
            index_list = []
            for _ in range(0, self.out_num):
                index = random.randint(0, len(self.dataset) - 1)
                while index in index_list:
                    index = random.randint(0, len(self.dataset) - 1)
                index_list.append(index)
            for i, idx in enumerate(tqdm.tqdm(index_list)):
                data = self.dataset.__getitem__(idx)
                self.infer(data, i)
        print("done")


if __name__ == "__main__":
    random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num_gpu", type=int, default=4)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    if args.num_proc is None and args.num_gpu:
        args.num_proc = args.num_gpu
    if args.num_proc != 1:
        mp.spawn(main, args=(args,), nprocs=args.num_proc, join=True)
    else:
        main(0, args)

    pass
