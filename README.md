# TSELM: Target Speaker Extraction using Discrete Tokens and Language Models

<div style="display:flex">
<div >
    <a href="https://arxiv.org/abs/2409.07841" class="no-link-style me-3" style="margin-right:5px">
      <button type="button" class="btn btn-dark pr-4">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
          fill="currentColor" class="bi bi-file-pdf"
          viewBox="0 0 500 500">
          <path fill="currentColor"
            d="M181.9 256.1c-5-16-4.9-46.9-2-46.9 8.4 0 7.6 36.9 2 46.9zm-1.7 47.2c-7.7 20.2-17.3 43.3-28.4 62.7 18.3-7 39-17.2 62.9-21.9-12.7-9.6-24.9-23.4-34.5-40.8zM86.1 428.1c0 .8 13.2-5.4 34.9-40.2-6.7 6.3-29.1 24.5-34.9 40.2zM248 160h136v328c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V24C0 10.7 10.7 0 24 0h200v136c0 13.2 10.8 24 24 24zm-8 171.8c-20-12.2-33.3-29-42.7-53.8 4.5-18.5 11.6-46.6 6.2-64.2-4.7-29.4-42.4-26.5-47.8-6.8-5 18.3-.4 44.1 8.1 77-11.6 27.6-28.7 64.6-40.8 85.8-.1 0-.1.1-.2.1-27.1 13.9-73.6 44.5-54.5 68 5.6 6.9 16 10 21.5 10 17.9 0 35.7-18 61.1-61.8 25.8-8.5 54.1-19.1 79-23.2 21.7 11.8 47.1 19.5 64 19.5 29.2 0 31.2-32 19.7-43.4-13.9-13.6-54.3-9.7-73.6-7.2zM377 105L279 7c-4.5-4.5-10.6-7-17-7h-6v128h128v-6.1c0-6.3-2.5-12.4-7-16.9zm-74.1 255.3c4.1-2.7-2.5-11.9-42.8-9 37.1 15.8 42.8 9 42.8 9z">
          </path>
        </svg>
        Paper
      </button>
    </a>
  </div>

<div>
          <a href="https://github.com/Beilong-Tang/TSELM" class="no-link-style">
            <button type="button" class="btn btn-dark">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                fill="currentColor" class="bi bi-github"
                viewBox="0 0 16 16">
                <path
                  d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8">
                </path>
              </svg>
              Code
            </button>
          </a>
        </div>
    </div>

Official Implementation of TSELM: Target Speaker Extraction using Discrete Tokens and Language Models. 

## Abstract
We propose TSELM, a novel target speaker extraction network that leverages discrete tokens and language models.
TSELM utilizes multiple discretized layers from WavLM as input
tokens and incorporates cross-attention mechanisms to integrate
target speaker information. Language models are employed to
capture the sequence dependencies, while a scalable HiFi-GAN
is used to reconstruct the audio from the tokens. By applying a
cross-entropy loss, TSELM models the probability distribution of
output tokens, thus converting the complex regression problem of
audio generation into a classification task. Experimental results
show that TSELM achieves excellent results in speech quality
and comparable results in speech intelligibility.


## Pre-requisites

1. Install [SpeechBrain](https://github.com/speechbrain/speechbrain/tree/main) (We use the Conformer model from it). 
2. Install all dependencies from `requirements.txt`
3. Download the encoder(WavLM Large and Kmeans) and decoder(Scalable HiFiGAN) checkpoint. Details can be found in [Model](#pretrained-model) session.
4. Download the data and run the scripts following [data/README.md](./data/README.md).

### Pretrained Model
Befor running experiments, we need to download the following frozen pretrained models.

| Name         | Link                                                        | Result                       |
|--------------|-------------------------------------------------------------|----------------------------|
| WavLM Large  | https://huggingface.co/microsoft/wavlm-large/tree/main      | wavlm-large |
| Kmeans       | [Download Kmeans Checkpoint](https://huggingface.co/Beilong/TSELM/resolve/main/kmeans_ckpt/kmeans_wavlm_ckpt.tar.gz?download=true)  | kmeans_ckpt      |
| Scalable HiFiGAN | [Download HiFiGAN Checkpoint](https://huggingface.co/Beilong/TSELM/resolve/main/backend_ckpt/hifigan-wavlm-l1-3-7-18-23-k1000-LibriTTS.tar.gz?download=true) | hifigan-wavlm-l1-3-7-18-23-k1000-LibriTTS |

Note that for the output of WavLM Large, it is recommended to clone the whole repository or download the whole directory. For Kmeans and Scalable HiFiGAN, we need to extract them after downloading. 


## Training

The training config is specified using `hyperpyyaml` package, which is basically a reflection. 

The config for training `TSELM-L` can be found in [config/tselm_l.yaml](./config/tselm_l.yaml). Before training, you need to specify the config for the frozen pretrained models and other training details. Details can be found in [config/tselm_l.yaml](./config/tselm_l.yaml) and [config/README.md](./config/README.md). 

After configuration, you can run 
```shell
## Train the model using the config 
python train.py --config_path ./config/tselm_l.yaml --log ./log --ckpt_path ./ckpt/tselm_l 
```
- `--config_path` specifies the path to the config file.
- `--log` specifies the log output directory. All logs will be put here.
- `--ckpt_path` specifies the checkpoint directory. Training can be resumed using the same checkpoint path. 

After training, the best model will be at `<ckpt_path>/best.pth`. 


## Inference
To infer our model on libri2mix testset, for example, you can run

```shell
## Generate output audio on libri2mix testset
python inference.py -scp <path_to_libri2mix_test_scp_folder> \
  -config ./config/tselm_l.yaml \
  -ckpt <path_to_ckpt> \
  --output <path_to_output_folder> \
  -gpus cuda:0 cuda:1 cuda:2 cuda:3 \
  -proc 8
```

- `-scp` specifies the the path to the libri2mix testset folder containing `aux_s1.scp`, `s1.scp`, and `mix_clean.scp`. 
- `-config` specifies the config. This config needs to have the `model` field. 
- `-ckpt` specifies the model checkpoint.
- `--output` specifies the output directory. 
The output audio will be output to this folder. Their names will be the same as those in .scp files. 
- `-gpus` specifies the available gpus to run inference.
- `-proc` specifies the total number of processes to run the inference in parallel. It will 
use the provided gpus and divide the processes equally on each device. Data will be split equally to each process.


## Model Checkpoint

Our TSELM-L checkpoint can be downloaded [here](https://huggingface.co/Beilong/TSELM/resolve/main/model_ckpt/tselm_l.pth?download=true).

You can infer on the libri2mix testset by substituting the `-ckpt` with path to the checkpoint. 

Note that you still need to download the [pretrained models](#pretrained-model) and add the corresponding checkpoint folder to [config/tselm_l.yaml](./config/tselm_l.yaml).

