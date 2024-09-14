## inference on libri2mix test set
import argparse
import tqdm
import os.path as op
import torch
import torch.nn as nn
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from dataset import TargetDataset


def main(args):
    scp = args.scp_dir
    mix_scp = op.join(scp, "mix_clean.scp")
    s1_scp = op.join(scp, "s1.scp")
    aux_s1_scp = op.join(scp, "aux_s1.scp")
    dataset = TargetDataset(
        mix_scp, aux_s1_scp, s1_scp, -1, mix_length=None, regi_length=None
    )
    with open(args.config_path, "r") as f:
        config = load_hyperpyyaml(f)
    model:nn.Module = config.get("model")
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.cuda(args.device)
    model.load_state_dict(ckpt['model_state_dict'], strict = False)
    with torch.no_grad():
        for mix, _, regi, mix_path, _, _ in tqdm.tqdm(dataset):
            mix, regi = mix.to(args.device), regi.cuda(args.device)
            mix, regi = mix.unsqueeze(0), regi.unsqueeze(0) # [1, T]
            print(regi.shape)
            output = model.inference(mix, regi) #[1,T]
            output = output.cpu()
            name = mix_path.split("/")[-1]
            torchaudio.save(op.join(args.output,name), output, 16000)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-scp", "--scp_dir", type=str, required=True)
    parser.add_argument("-config", "--config_path", type=str, required=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    main(args)

    pass
