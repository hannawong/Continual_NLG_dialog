CUDA_VISIBLE_DEVICES="1" \
python train.py --output_dir=./output_mixup --model_type=gpt2 \
--model_name_or_path=gpt2 --do_train  \
--do_eval \
--eval_data_file=sgd_alarm,sgd_banks,sgd_rentalcars \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train \
--overwrite_output_dir \
--split T \
--block_size=80 \
--mode=adapter --gradient_accumulation_step=8 --num_train_epochs 5 --per_gpu_train_batch_size 10 \
--EWC F

'''
CUDA_VISIBLE_DEVICES="5"
python test.py \
--model_type=gpt2 --model_name_or_path=./output --num_samples 1 \
--input_file sgd_alarm,sgd_banks \
--top_k 1 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix "baseline"

##lamol 10: 18.63, 0.1709
##no replay: 14.16, 0.1624
##replay 10: 19.098, 0.0844
'''
##sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_ridesharing,sgd_media,sgd_events,sgd_music,sgd_movies,sgd_flights,sgd_rentalcars,sgd_buses,sgd_hotels,sgd_services,sgd_homes,sgd_banks,sgd_alarm