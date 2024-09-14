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

from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from utils.env import AttrDict

def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_random_seed(worker_seed)

## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def setup_logger(args):
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
    return logger

# def setup_seed(seed, rank):
#     SEED = int(seed)
#     random.seed(SEED + rank)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     pass


def main(rank, args):
    with open(args.config_path, "r") as f:
        config_base = AttrDict(**yaml.load(f, Loader=yaml.BaseLoader))
        config_base.world_size = len(config_base.gpus)
    print(f"rank {rank} of world_size {config_base.world_size} started...")
    # setup_seed(config_base.seed, rank)
    setup(rank, config_base.world_size, args.dist_backend, port=int(config_base.port))
    ## logger
    logger = setup_logger(args)
    with open(args.config_path, "r") as f:
        config = AttrDict(**load_hyperpyyaml(f))
        config.world_size = len(config.gpus)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    ### prepare model
    model = config.model.cuda(rank)
    model.to(rank)

    model = DDP(
        model, device_ids=[rank], find_unused_parameters=config.find_unused
    )
    tr_dataset = config.tr_dataset(rank=rank)
    tr_data = DataLoader(
        tr_dataset,
        batch_size=config.batch_size // config.world_size,
        shuffle=False,
        sampler=DistributedSampler(dataset=tr_dataset, seed=config.sampler_seed + rank),
        num_workers=config.num_workers,
        collate_fn=config.collate_fn,
        worker_init_fn=seed_worker,
    )
    cv_dataset = config.cv_dataset(rank=rank)
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
        worker_init_fn=seed_worker,
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
        rank,
        rank,
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
    os.makedirs(args.ckpt_path, exist_ok=True)
    with open(args.config_path, "r") as f:
        config_base = AttrDict(**yaml.load(f, Loader=yaml.BaseLoader))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config_base.gpus])
    mp.spawn(main, args=(args,), nprocs=len(config_base.gpus), join=True)

    pass
