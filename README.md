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
3. Download the encoder(WavLM Large and Kmeans) and decoder(Unit HiFiGAN) checkpoint. Details can be found in [Model](#model) session.
4. Download the data following `README.md` in `data` folder.

### Model
To run experiments, we need to download the following pretrained models.


<div style="overflow-x:auto; width:100%;">
  <table style="table-layout:auto; width:100%; border-collapse: collapse;">
    <thead>
      <tr>
        <th>Name</th>
        <th>Link</th>
        <th>Note</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>WavLM Large</td>
        <td style="word-break: break-all; white-space: normal;">
          <a href="https://huggingface.co/microsoft/wavlm-large/tree/main">https://huggingface.co/microsoft/wavlm-large/tree/main</a>
        </td>
        <td>Download the whole folder.</td>
      </tr>
      <tr>
        <td>Kmeans</td>
        <td style="word-break: break-all; white-space: normal;">
          <a href="https://huggingface.co/Beilong/TSELM/resolve/main/kmeans_ckpt/kmeans_wavlm_ckpt.tar.gz?download=true">https://huggingface.co/Beilong/TSELM/resolve/main/kmeans_ckpt/kmeans_wavlm_ckpt.tar.gz?download=true</a>
        </td>
        <td>Download and extract.</td>
      </tr>
      <tr>
        <td>Unit HiFiGAN</td>
        <td style="word-break: break-all; white-space: normal;">
          <a href="https://huggingface.co/Beilong/TSELM/resolve/main/backend_ckpt/hifigan-wavlm-l1-3-7-18-23-k1000-LibriTTS.tar.gz?download=true">https://huggingface.co/Beilong/TSELM/resolve/main/backend_ckpt/hifigan-wavlm-l1-3-7-18-23-k1000-LibriTTS.tar.gz?download=true</a>
        </td>
        <td>Download and extract.</td>
      </tr>
    </tbody>
  </table>
</div>

## Pretrained Model

Our pretrained TSELM-L can be downloaded [here](https://huggingface.co/Beilong/TSELM/resolve/main/model_ckpt/tselm_l.pth?download=true).

## Training






## Evaluation
