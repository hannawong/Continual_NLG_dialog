CUDA_VISIBLE_DEVICES="1" \
python test.py \
--model_type=gpt2 --model_name_or_path=./output_mixup --num_samples 5 \
--input_file sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup_bnm
