CUDA_VISIBLE_DEVICES="0" \
python train.py --output_dir=./outputs/cl37_agem --model_type=gpt2 \
--model_name_or_path=gpt2 --do_train  \
--do_eval \
--eval_data_file=sgd_travel,sgd_payment \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=sgd_travel,sgd_payment,TMA_restaurant,TMB_music,sgd_ridesharing,TMA_auto,sgd_music,sgd_buses,TMB_restaurant,MWOZ_attraction,TMB_sport,sgd_movies,sgd_homes,TMA_coffee,sgd_restaurants,sgd_hotels,sgd_weather,sgd_trains,MWOZ_train,sgd_flights,sgd_media,MWOZ_taxi,sgd_alarm,TMA_movie,sgd_banks,TMA_pizza,TMB_flight,sgd_rentalcars,TMB_movie,sgd_events,MWOZ_restaurant,sgd_services,sgd_calendar,TMB_food-ordering,MWOZ_hotel,TMA_uber,TMB_hotel \
--overwrite_output_dir \
--split T \
--block_size=80 \
--BNM_ratio 0.5 --layer 11 \
--seed 0 \
--alpha 0.6 \
--mode=adapter --gradient_accumulation_step=1 --num_train_epochs 1 --per_gpu_train_batch_size 1 \
--EWC F --aug_method=none --BNM F --replay T --lamol F --AGEM T --only F --dataset cl37 \
--train T --test F
###
'''
CUDA_VISIBLE_DEVICES="4" \
python test.py \
--model_type=gpt2 --model_name_or_path=outputs/cl37_swapseed0 --num_samples 5 \
--input_file sgd_travel,sgd_payment,TMA_restaurant,TMB_music,sgd_ridesharing,TMA_auto,sgd_music,sgd_buses,TMB_restaurant,MWOZ_attraction,TMB_sport,sgd_movies,sgd_homes,TMA_coffee,sgd_restaurants,sgd_hotels,sgd_weather,sgd_trains,MWOZ_train,sgd_flights,sgd_media,MWOZ_taxi,sgd_alarm,TMA_movie,sgd_banks,TMA_pizza,TMB_flight,sgd_rentalcars,TMB_movie,sgd_events,MWOZ_restaurant,sgd_services,sgd_calendar,TMB_food-ordering,MWOZ_hotel,TMA_uber,TMB_hotel \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup


##sgd_travel,sgd_payment,TMA_restaurant,TMB_music,sgd_ridesharing,TMA_auto,sgd_music,sgd_buses,TMB_restaurant,MWOZ_attraction,TMB_sport,sgd_movies,sgd_homes,TMA_coffee,sgd_restaurants,sgd_hotels,sgd_weather,sgd_trains,MWOZ_train,sgd_flights,sgd_media,MWOZ_taxi,sgd_alarm,TMA_movie,sgd_banks,TMA_pizza,TMB_flight,sgd_rentalcars,TMB_movie,sgd_events,MWOZ_restaurant,sgd_services,sgd_calendar,TMB_food-ordering,MWOZ_hotel,TMA_uber,TMB_hotel


