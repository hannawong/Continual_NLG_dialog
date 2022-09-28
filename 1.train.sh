CUDA_VISIBLE_DEVICES="0" \
python train.py --output_dir=./output --model_type=gpt2 \
--model_name_or_path=./gpt --do_train  \
--do_eval \
--eval_data_file=MWOZ_attraction,MWOZ_hotel,MWOZ_restaurant,MWOZ_taxi,MWOZ_train \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=MWOZ_attraction,MWOZ_hotel,MWOZ_restaurant,MWOZ_taxi,MWOZ_train \
--overwrite_output_dir \
--split \
--block_size=80 \
--mode=adapter --gradient_accumulation_step=8 --num_train_epochs 10 --per_gpu_train_batch_size 10

CUDA_VISIBLE_DEVICES="0" \
python test.py \
--model_type=gpt2 --model_name_or_path=./output --num_samples 5 \
--input_file MWOZ_attraction,MWOZ_hotel,MWOZ_restaurant,MWOZ_taxi,MWOZ_train \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter

##lamol 10: 18.63, 0.1709
##no replay: 14.16, 0.1624
##replay 10: 19.098, 0.0844

##sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_ridesharing,sgd_media,sgd_events,sgd_music,sgd_movies,sgd_flights,sgd_rentalcars,sgd_buses,sgd_hotels,sgd_services,sgd_homes,sgd_banks,sgd_alarm