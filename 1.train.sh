export CUDA_VISIBLE_DEVICES=0
python train.py --output_dir=./output --model_type=gpt2 --model_name_or_path=./scgpt --do_train --do_eval --eval_data_file=./data/TMB_music/train.txt --per_gpu_train_batch_size 1 --num_train_epochs 1 --learning_rate 5e-5 --overwrite_cache --use_tokenize --train_data_file=./data/TMB_music/train.txt --overwrite_output_dir