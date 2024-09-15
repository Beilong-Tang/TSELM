python inference.py -scp /public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/dev -config ./config/tselm_l.yaml -ckpt /public/home/qinxy/bltang/ml_framework_slurm/exp/tsslm_final/discrete_k_1000/ckpt/conf_l/best.pth -d cuda:0 -o /public/home/qinxy/bltang/TSELM/output/tselm_l -gpus cuda:0 cuda:1 cuda:2 cuda:3 -proc 8 


rank 3 get spk1 1413
rank 0 get spk1 1316
rank 1 get spk1 770
rank 2 get spk1 8183
rank 2 get spk1 511
rank 0 get spk1 7240
rank 3 get spk1 3289
rank 1 get spk1 8498
rank 3 get spk1 1961
rank 1 get spk1 7938
rank 2 get spk1 8534
rank 0 get spk1 7188
rank 1 get spk1 6574
rank 1 get spk1 2512
rank 3 get spk1 5022
rank 0 get spk1 7011
rank 2 get spk1 984
rank 1 get spk1 196
rank 2 get spk1 5514
rank 3 get spk1 597
rank 0 get spk1 543
rank 0 get spk1 1624
rank 3 get spk1 6538
rank 2 get spk1 1098