python inference.py -scp /public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/dev -config ./config/tselm_l.yaml -ckpt /public/home/qinxy/bltang/ml_framework_slurm/exp/tsslm_final/discrete_k_1000/ckpt/conf_l/best.pth -d cuda:0 -o /public/home/qinxy/bltang/TSELM/output/tselm_l -gpus cuda:0 cuda:1 cuda:2 cuda:3 -proc 8 

rank 1 get spk1 5163
rank 0 get spk1 1316
rank 2 get spk1 5092
rank 1 get spk1 770
rank 3 get spk1 1263
rank 0 get spk1 7240
rank 2 get spk1 3866
rank 3 get spk1 6272
rank 1 get spk1 7938
rank 3 get spk1 731
rank 0 get spk1 7188
rank 1 get spk1 362
rank 2 get spk1 492
rank 2 get spk1 6098
rank 1 get spk1 2512
rank 0 get spk1 1624
rank 2 get spk1 5126
rank 0 get spk1 7011
rank 3 get spk1 2061
rank 2 get spk1 6509
rank 3 get spk1 3187
rank 1 get spk1 1624
rank 3 get spk1 2960
rank 0 get spk1 543