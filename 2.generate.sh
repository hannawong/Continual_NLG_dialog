CUDA_VISIBLE_DEVICES="0" \
python test.py \
--model_type=gpt2 --model_name_or_path=outputs/dailydialog_mixup0.6_bnmpool1.0 --num_samples 5 \
--input_file TMB_hotel,sgd_payment,TMB_movie,TMA_movie,TMB_restaurant,TMB_music,sgd_media,sgd_rentalcars,sgd_ridesharing,TMB_food-ordering,sgd_music,MWOZ_train,sgd_movies,sgd_hotels,MWOZ_attraction,TMA_uber,TMA_pizza,sgd_events,sgd_homes,sgd_services,sgd_calendar,sgd_weather,TMA_restaurant,TMB_flight,sgd_banks \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup
