### The base class for abstract trainer class
import torch
import os
import time
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader
from .helper import dict_to_str, save, load_ckpt
import random


def gather_tensors(tensor):
    """
    Gather tensors from all GPUs.
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def get_avg_result(res: dict):
    new_res = {}
    for k, v in res.items():
        tensors = gather_tensors(v)
        value = sum(t.item() for t in tensors) / len(tensors)
        new_res[k] = value
    return new_res


class AbsTrainer:
    def __init__(
        self,
        model,
        tr_data,
        cv_data,
        optim,
        config,
        ckpt_path,
        device,
        rank,
        logger,
    ):
        self.model = model
        self.tr_data = tr_data
        self.cv_data = cv_data
        self.config = config
        self.ckpt_dir = ckpt_path
        self.epoch_start = 0
        self.step = 0
        self.cv_log = {}
        self.optim = optim
        self.device = device
        self.rank = rank
        print(f"device is {self.device} for rank {rank}")
        self.log_interval = config.log_interval
        self.logger = logger
        self.max_ckpt = config.max_ckpt
        self.best_field = config.best_field
        self.best_value = None
        self.best_save_type = config.best_save_type
        self.ckpt_path = load_ckpt(self.ckpt_dir)
        ckpt_path = self.ckpt_path
        self.scheduler = config.scheduler
        random.seed(config.seed + rank)
        if self.scheduler is not None:
            self.scheduler = self.scheduler(optimizer=self.optim)
        self.new_bob = config.new_bob
        if ckpt_path is not None:
            ## loading ckpt
            self._log(f"loading model from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            torch.cuda.empty_cache()
            self.model.module.load_state_dict(ckpt["model_state_dict"])
            self.optim.load_state_dict(ckpt["optim"])
            self.epoch_start = ckpt["epoch"] + 1
            self.step = ckpt["step"]
            self.cv_log = ckpt["cv_log"]
            self.best_value = ckpt[self.best_field]
            self.optim.load_state_dict(ckpt["optim"])
            self.scheduler = ckpt["scheduler"]
            self.new_bob = ckpt["new_bob"]

    def _log(self, msg):
        if self.rank == 0:
            self.logger.info(msg)
        pass

    def _apply_scheduler(self, result):
        if self.new_bob is not None:
            if self.scheduler is not None:
                print(
                    "WARNING!, ONLY APPLY NEW BOB instead of the scheduler provided since you provided both "
                )
            _, lr = self.new_bob(result)
            ## adjust the learning rate
            for p in self.optim.param_groups:
                p["lr"] = lr
        elif self.scheduler is not None:
            self.scheduler.step()

    def _save(self, model, cv_log, epoch, optim, path, step, save_best: bool):
        if self.rank == 0:
            self._log(f"saving model... for epoch {epoch}")
            content = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optim": optim.state_dict(),
                "cv_log": cv_log,
                "scheduler": self.scheduler,
                "new_bob": self.new_bob,
                self.best_field: self.best_value,
            }
            save(
                path,
                content,
                self.max_ckpt,
            )
            if save_best:
                self._log(f"saving the best model of epoch {epoch}")
                torch.save(content, path.replace(f"epoch{epoch}.pth", f"best.pth"))
        pass

    def _train_one_batch(self, batch, data, optim, if_log) -> dict:
        raise NotImplementedError("not implemented")

    def _eval_one_batch(self, data) -> dict:
        raise NotImplementedError("not implemented")

    def _train(self, optim, tr_data, epoch):
        self.model.train()
        total = len(tr_data) * tr_data.batch_size
        start_time = time.time()
        for batch, data in enumerate(tr_data):
            if_log = batch % self.log_interval == 0
            res = self._train_one_batch(batch, data, optim, if_log)
            if if_log:
                current = (batch + 1) * tr_data.batch_size
                res["epoch"] = epoch
                res["step"] = self.step
                res["p"] = f"[{current:>5d}/{total:>5d}]"
                res["time/batch"] = (
                    f"{(time.time() - start_time)*1000 / self.log_interval :.2f}ms"
                )
                start_time = time.time()
                self._log(f"tr, {dict_to_str(res)}")
            self.step += 1

    def _eval(self, cv_data, epoch):
        self.model.eval()
        result = None
        if self.rank == 0:
            print(f"evaluating on cv_data of len {len(cv_data)* cv_data.batch_size}")
        with torch.no_grad():
            for data in cv_data:
                res = self._eval_one_batch(data)
                if result == None:
                    result = res
                else:
                    for key in result.keys():
                        result[key] += res[key]
        for key in result.keys():
            result[key] = result[key] / len(cv_data)
        ## gather all tensors onto the same device
        result = get_avg_result(result)
        self._log(f"eval epoch {epoch} {dict_to_str(result)}")
        if epoch != -1:
            self.cv_log[epoch] = result
        return result[self.best_field]

    def train(self):
        if self.config.get("pre_eval"):
            self._log(f"tesing eval --- ")
            ## test eval first
            self._eval(
                DataLoader(
                    Subset(
                        self.cv_data.dataset, list(range(2 * self.cv_data.batch_size))
                    ),
                    batch_size=self.cv_data.batch_size,
                    collate_fn=self.cv_data.collate_fn,
                ),
                -1,
            )
        for epoch in range(self.epoch_start, self.config["epoch"]):
            self._log(f"...epoch {epoch}...")
            self.tr_data.sampler.set_epoch(epoch)
            ### training
            self._train(self.optim, self.tr_data, epoch)
            #### evaluation
            result = self._eval(self.cv_data, epoch)
            if self.best_value is None:
                save_best = True
                self.best_value = result
            else:
                save_best = (
                    result > self.best_value
                    if self.best_save_type == "ascend"
                    else result < self.best_value
                )
                if save_best:
                    self.best_value = result
            ### save model
            self._save(
                self.model,
                self.cv_log,
                epoch,
                self.optim,
                os.path.join(self.ckpt_dir, f"epoch{epoch}.pth"),
                self.step,
                save_best,
            )
            dist.barrier()
            self._apply_scheduler(result)
