python generate.py \
--model_type=gpt2 --model_name_or_path=./output --num_samples 1 \
--input_file sgd_buses,sgd_hotels,sgd_services,sgd_homes,sgd_banks,sgd_alarm \
--top_k 1 --length 80 \
--device cuda:2


