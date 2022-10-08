CUDA_VISIBLE_DEVICES="1" \
python 3.scorer.py \
--domain sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train --mode=adapter --suffix "mixup_bnm"


##baseline: 19.8525 0.05369839165486486
##mixup:21.80333333333333 0.03850840325673669
