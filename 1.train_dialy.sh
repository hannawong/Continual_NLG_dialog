CUDA_VISIBLE_DEVICES="6" \
python train.py --output_dir=./outputs/dailydialog_ewc --model_type=gpt2 \
--model_name_or_path=gpt2 --do_train  \
--do_eval \
--eval_data_file=Ordinary_Life,School_Life,Culture_and_Education,Attitude_and_Emotion,Relationship,Tourism,Health,Work,Politics,Finance \
--learning_rate 0.0625 --use_tokenize \
--overwrite_cache \
--train_data_file=sgd_travel,sgd_payment,TMA_restaurant,TMB_music,sgd_ridesharing,TMA_auto,sgd_music,sgd_buses,TMB_restaurant,MWOZ_attraction,TMB_sport,sgd_movies,sgd_homes,TMA_coffee,sgd_restaurants,sgd_hotels,sgd_weather,sgd_trains,MWOZ_train,sgd_flights,sgd_media,MWOZ_taxi,sgd_alarm,TMA_movie,sgd_banks,TMA_pizza,TMB_flight,sgd_rentalcars,TMB_movie,sgd_events,MWOZ_restaurant,sgd_services,sgd_calendar,TMB_food-ordering,MWOZ_hotel,TMA_uber,TMB_hotel \
--overwrite_output_dir \
--split T \
--block_size=80 \
--max_seq=100 \
--BNM_ratio 0.6 --layer 11 \
--seed 0 \
--alpha 0.6 \
--mode=adapter --gradient_accumulation_step=1 --num_train_epochs 5 --per_gpu_train_batch_size 100 \
--EWC T --aug_method=none --BNM F --replay F --lamol F --AGEM F --only F --dataset dailydialog \
--train T --test T
###
'''
CUDA_VISIBLE_DEVICES="5" \
python test.py \
--model_type=gpt2 --model_name_or_path=./outputs/dailydialog_multi_seed0 --num_samples 1 \
--input_file Health \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup