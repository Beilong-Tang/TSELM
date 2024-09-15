# Under Processing 

# TSELM: Target Speaker Extraction using Discrete Tokens and Language Models
Official Implementation of TSELM: Target Speaker Extraction using Discrete Tokens and Language Models

 [**Paper**](https://arxiv.org/abs/2409.07841)
| [**Demo**](https://beilong-tang.github.io/TSELM.demo/)

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
3. Download the encoder(WavLM+Kmeans) and decoder(Unit HiFiGAN) checkpoint. Details can be found in [Model](#model) session.

## Usage 

### Data Preparation

See the `README.md` in `data` foler.

### Model
Our model is comprised of three components: `encoder`, `modeling`, and `decoder`.
Our `encoder` and `decoder` is fronzen. To successfully run experiments, we need to download the checkpoint of them. All the checkpoints can be found [here](https://huggingface.co/Beilong/TSELM/tree/main).

#### Encoder

__WavLM Large__  It can be downloaded [here](https://huggingface.co/microsoft/wavlm-large/tree/main). We recommend to download the whole folder. 

__Kmeans__. Our 6 kmeans models of hidden layer 1, 3, 7, 12, 18, 23 from WavLM Large can be downloaded [here](https://huggingface.co/Beilong/TSELM/tree/main/kmeans_ckpt). Extract it afterwards.

#### Decoder

__ 



