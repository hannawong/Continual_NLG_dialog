# Towards Robustness and Diversity: Continual Learning in Dialogue Generation with Text-Mixup and Batch Nuclear-Norm Maximization

This repository contains the dataset, source code for paper: *Towards Robustness and Diversity: Continual Learning in Dialogue Generation with Text-Mixup and Batch Nuclear-Norm Maximization*

This repository is based on hugginface transformer package and OpenAI GPT-2, containing model training code. 

## Dataset 
The preprocessed dataset is stored under directory `data/`

**Data files includes** 

<code>{domain}/train.txt</code>: linearized training set for GPT-2 models.

<code>{domain}/test.txt</code>: linearized testing set for GPT-2 models.

<code>{domain}/val.txt</code>: linearized validation set for GPT-2 models.

**Data format**
TOD37:
```text
flight_search ( type = "round trip" ; origin = "LAX" ; destination = "SLC") & A round trip flight from LAX to SLC?
```

DailyDialog:
```text
Look out ! __eou__ What's it ? __eou__ You must have rolled over something just now . __sou__ What you said gives me the creeps !
```


## Pipeline
**Training**
```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --output_dir=MODEL_SAVE_PATH --model_type=gpt2 --model_name_or_path=PRE_TRINED_MODEL_PATH --do_train --do_eval --eval_data_file=data/restaurant/train.txt --per_gpu_train_batch_size 1 --num_train_epochs EPOCH --learning_rate LR --overwrite_cache --use_tokenize --train_data_file=data/restaurant/train.txt --overwrite_output_dir
```
<code>MODEL_SAVE_PATH </code>: Path of the saving model .

<code>PRE_TRAINED_MODEL_PATH </code>: Initial checkpoint; Could start from gpt2, gpt2-meidum or our provided scgpt folder.

<code>EPOCH </code>: Number of training epochs;  5 is enough for a reasonable performance

<code>LR </code>: Learning rate; 5e-5, 1e-5, or 1e-4

**Decoding**
```bash
export CUDA_VISIBLE_DEVICES=0
python generate.py --model_type=gpt2 --model_name_or_path=MODEL_SAVE_PATH --num_samples 5 --input_file=data/restaurant/test.txt --top_k 5 --output_file=results.json --length 80
```

**Evaluate**
```bash
python evaluator.py --domain restaurant results.json
```

