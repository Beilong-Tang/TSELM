from trainer.abs_trainer import AbsTrainer
import os.path as op


class Trainer(AbsTrainer):
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
        super().__init__(
            model,
            tr_data,
            cv_data,
            optim,
            config,
            ckpt_path,
            device,
            rank,
            logger,
        )
        print(f"using trainer at {op.abspath(__file__)}")

    def get_res(self, loss, error):
        res = {}
        res["loss"] = loss
        res["error"] = error
        return res

    def _train_one_batch(self, batch, data, optim, if_log) -> dict:
        mix, clean, regi = data
        mix, clean, regi = (
            mix.to(self.device),
            clean.to(self.device),
            regi.to(self.device),
        )
        loss, _, _, error = self.model(mix, clean, regi, inference=False)
        loss.backward()

        optim.step()
        optim.zero_grad()
        if if_log:
            return self.get_res(loss, error)
        return None

    def _eval_one_batch(self, data) -> dict:
        mix, clean, regi, _, _, _ = data
        mix, clean, regi = (
            mix.to(self.device),
            clean.to(self.device),
            regi.to(self.device),
        )
        loss, _, _, error = self.model(mix, clean, regi, inference=False)
        res = self.get_res(loss, error)
        return res
