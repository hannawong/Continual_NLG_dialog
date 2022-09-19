CUDA_VISIBLE_DEVICES="0" \
python generate.py \
--model_type=gpt2 --model_name_or_path=./output --num_samples 5 \
--input_file sgd_rentalcars \
--top_k 1 --top_p 0.9 --length 100 \
--device cuda \
--mode adapter


##,sgd_media,sgd_events,sgd_travel,sgd_payment,sgd_weather,sgd_trains,sgd_calendar,sgd_music,sgd_movies,sgd_flights,sgd_rentalcars
