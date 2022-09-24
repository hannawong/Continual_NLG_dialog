CUDA_VISIBLE_DEVICES="3" \
python train.py --output_dir=./output_ctr --model_type=gpt2 \
--model_name_or_path=./gpt --do_train  \
--do_eval \
--eval_data_file=sgd_weather \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=sgd_weather,sgd_alarm,sgd_trains \
--overwrite_output_dir \
--split \
--block_size=80 \
--mode=ctr --gradient_accumulation_step=8 --num_train_epochs 10 --per_gpu_train_batch_size 10





##sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_ridesharing,sgd_media,sgd_events,sgd_music,sgd_movies,sgd_flights,sgd_rentalcars,sgd_buses,sgd_hotels,sgd_services,sgd_homes,sgd_banks,sgd_alarm