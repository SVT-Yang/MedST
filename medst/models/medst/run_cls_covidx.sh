CUDA_VISIBLE_DEVICES=1 python medst_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01  --path path/to/medst --batch_size 96  
CUDA_VISIBLE_DEVICES=1 python medst_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1  --path path/to/medst --batch_size 96 
CUDA_VISIBLE_DEVICES=1 python medst_finetuner.py --gpus 1 --dataset covidx --data_pct 1  --path path/to/medst --batch_size 96 