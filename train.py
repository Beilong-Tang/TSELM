import argparse
import torch
import random
import numpy as np
import sys
import os
import yaml

sys.path.append(os.getcwd())

### set up logging
import logging
import datetime

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

###
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from utils.env import AttrDict

### add the path to the funcodec library


## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    with open(args.config_path, "r") as f:
        config_base = yaml.load(f, Loader=yaml.BaseLoader)
    config_base = AttrDict(**config_base)
    config_base.world_size = len(config_base.gpus)
    SEED = int(config_base.seed)
    random.seed(SEED + rank)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"rank {rank} of world_size {config_base.world_size} started...")
    ## set up logger before the config
    setup(rank, config_base.world_size, args.dist_backend, port=int(config_base.port))
    ## logger
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = args.log
    print(f"logging dir: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/{now}.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    logger.info("logger initialized")
    for f in config_base.sys_path:
        sys.path.append(f)
    with open(args.config_path, "r") as f:
        config = load_hyperpyyaml(f)
    config = AttrDict(**config)
    config.world_size = len(config.gpus)
    ###
    # Set the CUDA device based on local_rank
    config.gpu = rank
    config.rank = rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    ### prepare model
    model = config.model.cuda(config.gpu)
    model.to(config.gpu)

    model = DDP(
        model, device_ids=[config.gpu], find_unused_parameters=config.find_unused
    )
    tr_dataset = config.tr_dataset(rank=config.rank)
    tr_data = DataLoader(
        tr_dataset,
        batch_size=config.batch_size // config.world_size,
        shuffle=False,
        sampler=DistributedSampler(dataset=tr_dataset, seed=config.sampler_seed),
        num_workers=config.num_workers,
        collate_fn=config.collate_fn,
    )
    cv_dataset = config.cv_dataset(rank=config.rank)
    cv_data = DataLoader(
        cv_dataset,
        batch_size=(
            config.batch_size_eval // config.world_size
            if config.batch_size_eval
            else config.batch_size // config.world_size
        ),
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=config.collate_fn,
    )

    optim = config.optim(params=filter(lambda p: p.requires_grad, model.parameters()))
    ### start training loop

    trainer_class = config.trainer
    trainer = trainer_class(
        model,
        tr_data,
        cv_data,
        optim,
        config,
        args.ckpt_path,
        config.gpu,
        config.rank,
        logger,
    )
    print("start training model")
    trainer.train()
    cleanup()
    print("training done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config_base = yaml.load(f, Loader=yaml.BaseLoader)
    config_base = AttrDict(**config_base)
    config_base.world_size = len(config_base.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config_base.gpus])
    mp.spawn(main, args=(args,), nprocs=config_base.world_size, join=True)

    pass
