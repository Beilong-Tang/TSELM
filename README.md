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


