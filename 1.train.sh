CUDA_VISIBLE_DEVICES="6" \
python train.py --output_dir=./output_backtrans --model_type=gpt2 \
--model_name_or_path=gpt2 --do_train  \
--do_eval \
--eval_data_file=sgd_alarm,sgd_banks,sgd_calendar \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train \
--overwrite_output_dir \
--split T \
--block_size=80 \
--mode=adapter --gradient_accumulation_step=8 --num_train_epochs 5 --per_gpu_train_batch_size 10 \
--EWC T --aug_method back_trans --BNM F

