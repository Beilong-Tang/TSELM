# Librispeech Data

Our model is trained on LibriSpeech with dynamic mixing. The data is available at [https://www.openslr.org/12](https://www.openslr.org/12). 


To use our training flow, we need the following data in this folder:

- train-clean-100
- train-clean-360
- dev-clean (for evaluation)
- test-clean (for testing)

You should have the following folder structure before proceeding:
```
.
├── dev-clean
├── generate_list.py
├── README.md
├── test-clean
├── train-clean-100
└── train-clean-360
```

