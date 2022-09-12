CUDA_VISIBLE_DEVICES="0" \
python train.py --output_dir=./output --model_type=gpt2 \
--model_name_or_path=./scgpt --do_train \
--do_eval \
--eval_data_file=sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_ridesharing,sgd_media,sgd_events,sgd_music,sgd_music,sgd_movies,sgd_flights,sgd_rentalcars,sgd_buses,sgd_hotels,sgd_services,sgd_homes,sgd_banks,sgd_alarm \
--per_gpu_train_batch_size 16 --num_train_epochs 2 --learning_rate 5e-6 \
--overwrite_cache --use_tokenize \
--train_data_file=sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_ridesharing,sgd_media,sgd_events,sgd_music,sgd_music,sgd_movies,sgd_flights,sgd_rentalcars,sgd_buses,sgd_hotels,sgd_services,sgd_homes,sgd_banks,sgd_alarm \
--overwrite_output_dir 