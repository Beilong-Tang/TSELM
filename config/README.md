# Config 

The template configuration for TSELM-L is given at [tselm_l.yaml](tselm_l.yaml). 

__Before processing__, make sure you 
1. Have all the data ready by following the scripts in `data` folder
2. Download the frozen pretrained models.  

To train the model, you need to at least change the following in the config

## Data and Pretrained Models
```yaml
# DATA 
# The path to the train_100_360.pt containing the training data 
# e.g. .../list/train/train_100_360.pt
tr_data_scp_path: <path_to_train_100_360.pt> 

# development set mixture path 
# e.g. .../list/libri2mix_dev/mix_clean.scp
cv_mix_path: <path_to_mix_clean.scp>

# development set reference path 
# e.g. .../list/libri2mix_dev/aux_s1.scp
cv_ref_path: <path_to_aux_s1.scp>

# development set target path 
# e.g. .../list/libri2mix_dev/s1.scp
cv_clean_path: <path_to_s1.scp>
```
```yaml
# PRETRAINED MODEL
# The path to the hifigan folder
# e.g. .../hifigan-wavlm-l1-3-7-18-23-k1000-LibriTTS
hifi_gan_path: <path_to_hifi_gan_ckpt_folder>

# The path to the wavlm-large folder
# e.g. .../wavlm-large
wavlm_path: <path_to_wavlm_ckpt_folder>

# The path to the kmeans model folder
# e.g. .../kmeans_ckpt
kmeans_path: <path_to_kmeans_ckpt_folder> 
```

Note that the path for the config for the __pretrained__ model is the path to the __directory__ instead of single files!

---

To specify 