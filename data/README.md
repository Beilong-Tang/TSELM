# Data Preparation

## Training data:
We use `train-clean-100`, and `train-clean-360` of LibriSpeech for training. The data is available at [https://www.openslr.org/12](https://www.openslr.org/12). 

#### Download
You can download manualy using the [link](https://www.openslr.org/12), or use the script 
provided in the file `download_librispeech.sh`.

To download the data into the current folder, you can run
```
bash download_librispeech.sh .
```

## Evaluation and Test data

For evaluation and testing, our reference speech is randomly selected. 
To enhance reproducibility and lower the complexity to generate the data, we directly upload
our data to huggingface:

`libri2mix_dev`: https://huggingface.co/datasets/Beilong/libri2mix_clean_target/resolve/main/libri2mix_dev.tar.gz?download=true


`libri2mix_test`: https://huggingface.co/datasets/Beilong/libri2mix_clean_target/resolve/main/libri2mix_test.tar.gz?download=true

After downloading and extracting, you get `libri2mix_dev` and `libri2mix_test` where each 
folder has:
- s1: the clean speech for target speaker
- aux_s1: reference speech for s1
- mix_clean: the mixture. 

## SCP generation

Make sure you have all the data following the [Data Preparation](#data-preparation) step. 

Create an output folder which does not have subfoler `list` in it.

Run
```
python generate_list.py --librispeech_train_100 <path_to_train-clean-100> \
 --librispeech_train_360 <path_to_train-clean-360> \
 --libri2mix_dev <path_to_libri2mix_dev> \
 --libri2mix_test <path_to_libri2mix_test>
 --output <path_to_output_folder>
```

Scp files will be generated under the `list` folder of your output path.

You list folder will look like:
```
.
├── libri2mix_dev
│   ├── aux_s1.scp 
│   ├── mix_clean.scp
│   └── s1.scp
├── libri2mix_test
│   ├── aux_s1.scp
│   ├── mix_clean.scp
│   └── s1.scp
└── train
    └── train_100_360.pt # Dict[str, list[str]] mapping a speaker to all its utterances
```

`train_100_360.pt` is nothing but a `Dict[str, list[str]]` which maps a 
speaker to all its utterances. 

For example, you can validate that
```python
spk_dict = torch.load("train_100_360.pt")
for k, v in spk_dict.items()[:1]:
    print(k) # the speaker name
    print(v) # the speech utterances from this speaker
```