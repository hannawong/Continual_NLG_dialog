# Towards Robustness and Diversity: Continual Learning in Dialogue Generation with Text-Mixup and Batch Nuclear-Norm Maximization

This repository contains the dataset, source code for paper: *Towards Robustness and Diversity: Continual Learning in Dialogue Generation with Text-Mixup and Batch Nuclear-Norm Maximization*

This repository is based on hugginface transformer package and OpenAI GPT-2, containing model training code. 

## Dataset 
The preprocessed dataset is stored under directory `data/`

**Data files includes** 

<code>{domain}/train.txt</code>: linearized training set for GPT-2 models.

<code>{domain}/test.txt</code>: linearized testing set for GPT-2 models.

<code>{domain}/val.txt</code>: linearized validation set for GPT-2 models.

**Data Example**

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
CUDA_VISIBLE_DEVICES="0" \
python train.py --output_dir=OUTPUT_DIR --model_type=gpt2 \
--model_name_or_path=gpt2 --do_train  \
--do_eval \
--eval_data_file=EVAL_DATA_FILES \
--learning_rate LR --use_tokenize \
--overwrite_cache \
--train_data_file=TRAIN_DATA_FILES \
--overwrite_output_dir --split T --block_size=80 \
--BNM_ratio KAPPA --seed 0 \
--alpha ALPHA \
--mode=adapter --gradient_accumulation_step=1 --num_train_epochs EPOCH --per_gpu_train_batch_size BATCHSIZE \
--EWC F --aug_method=none --BNM F --replay T --lamol F --AGEM T --only F --dataset DATASET \
```
<code>OUTPUT_DIR </code>: Path of the saving model .

<code>EVAL_DATA_FILES </code>: domains to perform evaluation, separated by ","

<code>TRAIN_DATA_FILES </code>: domains to train, separated by ","
<code> KAPPA </code>: combination ratio kappa of BNNM.
<code> ALPHA </code>: hyperparameter alpha for text-mixup.
<code>EPOCH </code>: Number of training epochs;  5 is enough for a reasonable performance
<code>BATCHSIZE </code>: batchsize; 32 or 64.
<code>LR </code>: Learning rate; 6.25E-05 or 6.25E-04

**Decoding**
```bash
CUDA_VISIBLE_DEVICES="0" \
python test.py \
--model_type=gpt2 --model_name_or_path=OUTPUT_DIR --num_samples 5 \
--input_file EVAL_FILE_NAMES \
--top_k 5 --top_p 1.0 --length 80 \
--device cuda \
--mode adapter --suffix mixup
```

<code>OUTPUT_DIR </code>: Path of the saving model .
<code>EVAL_FILE_NAMES </code>: Domains to perform decoding, separated by ",".

**Evaluate**
```bash
CUDA_VISIBLE_DEVICES="7" \
python 3.scorer.py --domain DOMAIN_NAMES --mode=adapter --suffix SUFFIX
```

<code> DOMAIN_NAMES </code>: Domains to perform evaluating, separated by ",".


