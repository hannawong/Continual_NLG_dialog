python generate.py \
--model_type=gpt2 --model_name_or_path=./output --num_samples 1 \
--input_file sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_ridesharing,sgd_media,sgd_events \
--top_k 1 --length 80 \
--device cuda:1
