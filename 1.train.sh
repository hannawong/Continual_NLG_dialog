CUDA_VISIBLE_DEVICES="3" \
python train.py --output_dir=./output_mixup_1.2_bnm --model_type=gpt2 \
--model_name_or_path=gpt2 --do_train  \
--do_eval \
--eval_data_file=sgd_alarm,sgd_banks,sgd_calendar \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train \
--overwrite_output_dir \
--split T \
--block_size=80 \
--mode=adapter --gradient_accumulation_step=8 --num_train_epochs 1 --per_gpu_train_batch_size 10 \
--EWC F --aug_method=mixup --BNM T --replay T


# sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train
CUDA_VISIBLE_DEVICES="3" \
python test.py \
--model_type=gpt2 --model_name_or_path=./output_mixup_1.2_bnm --num_samples 5 \
--input_file sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup_1.2_bnm