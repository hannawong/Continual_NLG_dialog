CUDA_VISIBLE_DEVICES="2" \
python test.py \
--model_type=gpt2 --model_name_or_path=outputs/output_mixup1_alpha0.6_bnm0.5_layer6_seed0 --num_samples 5 \
--input_file sgd_travel,sgd_payment,TMA_restaurant,TMB_music,sgd_ridesharing,TMA_auto,sgd_music,sgd_buses,TMB_restaurant,MWOZ_attraction,TMB_sport,sgd_movies,sgd_homes,TMA_coffee,sgd_restaurants,sgd_hotels,sgd_weather,sgd_trains,MWOZ_train,sgd_flights,sgd_media,MWOZ_taxi,sgd_alarm,TMA_movie,sgd_banks,TMA_pizza,TMB_flight,sgd_rentalcars,TMB_movie,sgd_events,MWOZ_restaurant,sgd_services,sgd_calendar,TMB_food-ordering,MWOZ_hotel,TMA_uber,TMB_hotel \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup
