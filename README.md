# CubiAI

## Generate dataset


## Run experiments locally
```
python experiment_binary.py config/local/pretrain.json D:/cn_perf_local/pretrain --temp_folder D:/cn_perf_temp/pretrain --epochs 2
```


## Run experiemts on Orion
```
sbatch slurm_pretrain_binary.sh config/pretrain/b0_normal_level2.json b0_normal_level2 2
sbatch slurm_pretrain_multiclass.sh config/pretrain/b0_normal_level1_level2.json b0_normal_level1_level2 2

sbatch slurm_scratch_binary.sh config/scratch/b0_normal_level2.json b0_normal_level2 2
sbatch slurm_scratch_multiclass.sh config/scratch/b0_normal_level1_level2.json b0_normal_level1_level2 2
```
