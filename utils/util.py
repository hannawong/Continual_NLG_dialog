from symbol import with_item
import torch
import argparse
import torch,json
import torch.nn.functional as F
import numpy as np
from model.Seq2SeqToD import Seq2SeqToD
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.multiprocessing as mp # 这里不需要使用pytorch的multiprocessing
from utils.data_augmentation import *


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 
            logits: shape: [5,50527].
            top_k: keep only top k tokens with highest probability (top-k filtering).
            top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                 (http://arxiv.org/abs/1904.09751)
    """

    # top-k filtering: Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value ##大部分都是-inf,只有topk个不是-inf。 shape = [5,50527]
    ## top-p filtering:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True) ##sorted_logits = [-2.0444, -4.1561, -4.8751,  ...,    -inf,    -inf,    -inf]重复5次；sorted_indices = [ 1222,   425,   532,  ..., 16760, 16761, 16762]重复5次
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) ###[0.7994, 0.8962, 0.9433,  ..., 1.0000, 1.0000, 1.0000]重复5次
        # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p  ###[False, False,  True,  ...,  True,  True,  True]重复5次
        # 保证第一个一定是False，即不过滤掉。Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False ##第一个一定是False

        # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(args,model, length, context, num_samples=1, temperature=1.1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu',task_id = -1):
    assert task_id != -1
    num_samples = 5
    ##length:80, num_samples:5, temperature = 1.0,top_k = 5, top_p = 0.9,repetition_penalty = 1.0
    context = torch.tensor(context, dtype=torch.long, device=device)  ##shape:([25])
    context = context.unsqueeze(0).repeat(num_samples, 1) ##shape:[5,25]
    
    generated = context
    
    with torch.no_grad():
        for _ in range(length): ##length = 80
            if args.mode == "GPT2":
                inputs = {'input_ids': generated} ###shape:[5,25]
            elif args.mode == "adapter":
                inputs = {'input_ids': generated, 'labels':None, 'task_id':task_id}
            if args.mode == "GPT2":
                outputs = model(**inputs) ###outputs[0].shape: [5,25,50527] 
                next_token_logits = outputs[0][:, -1, :] #/ temperature ###[5,50527]
            elif args.mode == "adapter":
                outputs = model(generated,labels = None,task_id = task_id,s = 400)
                next_token_logits = outputs[:,-1, :]

            #next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  ###[5,50527], 只是大部分都变成了-inf
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1) ##[1222]重复5次
            generated = torch.cat((generated, next_token), dim=1) ###[5,26]
    return generated

from torch.utils.data import DataLoader, Dataset
prompt = {"sgd_hotels":"find hotel and ratings","sgd_services":"book doctor appointments","sgd_alarm":"set up alarm time",\
            "sgd_events":"show when and where event is","sgd_buses":"confirm leaving time, passengers and fare of buses","sgd_weather":"show temperature and wind","sgd_rentalcars":"rent cars"\
                ,"sgd_calendar":"begin and end time","sgd_travel":"travel","sgd_ridesharing":"share a ride","sgd_media":"media",\
                    "sgd_music":"music","sgd_movies":"find movie and theater","sgd_payment":"make payment to people","sgd_trains":"journey time and money","sgd_banks":"transfer and balance in checking/saving account"}


class TextSeqDataset(Dataset):
    def __init__(self, tokenizer, args, raw_texts, max_seq=80):
        self.examples = []
        self.labels = []
        self.masks = []
  
        for line in raw_texts:
            self.masks.append([1] *  max_seq)
            line = line.strip()   
            raw_str = line.lower() ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced
            if len(raw_str.split()) > max_seq -1: ##截断
                raw_str = ' '.join(raw_str.split()[:max_seq -1])
            raw_str += ' ' + tokenizer.eos_token ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced <|endoftext|>
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))

            if len(tokenized_text) < max_seq: ##pad
                tokenized_text = [0] * (max_seq - len(tokenized_text)) + tokenized_text   ###pad
            else:
                tokenized_text = tokenized_text[:max_seq]
            
            self.examples.append(tokenized_text)

        self.labels = self.examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.masks[item]), torch.tensor(self.labels[item])



def generate_replay(args,task_id,domain_names,tokenizer,model,mode = "REPLAY",sample_frac = 0.01):
    file =  open("data/replay/train.txt","a")
    if mode == "LAMOL":
            print("Generate LAMOL!!")
            prev_task_id = task_id-1

            if domain_names[prev_task_id] not in prompt:
                prompt[domain_names[prev_task_id]] = domain_names[prev_task_id].split("_")[1]
            replay_buffer = []
            cnt = 0
            while True:
                context_tokens = tokenizer.encode("["+domain_names[prev_task_id][4:]+"]", add_special_tokens=False)
                print("["+domain_names[prev_task_id][4:]+"]")
                out = sample_sequence(
                        args,
                        model=model,
                        context=context_tokens,
                        num_samples=1,
                        length=80,
                        temperature=1.0,
                        top_k=5,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        is_xlnet=False,
                        is_xlm_mlm=False,
                        xlm_mask_token=False,
                        xlm_lang=False,
                        device=args.device,
                        task_id = task_id
                    )
                out = out[:, len(context_tokens):].tolist() ###只取生成的后面
                examples = []
                for o in out:
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    text = text[: text.find('<|endoftext|>')] ##只取到<endoftext>之前
                    examples.append(text)
                for example in examples:
                    cnt += 1
                    if cnt > 10 * sample_size:
                        replay_buffer.append(example.split(prompt[domain_names[prev_task_id]])[-1])

                    elif len(example) > 100 and prompt[domain_names[prev_task_id]] in example: ##domain_names[prev_task_id][4:] in example and
                        #print("find",example)
                        replay_buffer.append(example.split(prompt[domain_names[prev_task_id]])[-1])
                if len(replay_buffer) > sample_size:
                    #print(replay_buffer)
                    break
            for item in range(sample_size):
                file.write(replay_buffer[item]+"\n")
    elif mode == "REPLAY":
        from random import sample
        prev_task_id = task_id - 1
        replay_buffer = []
        training = open("data/"+domain_names[prev_task_id]+"/train.txt").read().split("\n")
        sample_size = int(len(training) * sample_frac)
        choose = sample(training,min(len(training),sample_size))
        print(len(choose))
        augs = []
        for sentence in choose:   
            for i in range(2):    
                if args.aug_method == "replace":
                    aug = synonym_replacement(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                    aug = " ".join(aug)
                    augs.append(aug)
                if args.aug_method == "del":
                    aug = random_deletion(sentence.split(),0.1)
                    aug = " ".join(aug)
                    augs.append(aug)
                if args.aug_method == "insert":
                    aug = random_insertion(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                    aug = " ".join(aug)
                    augs.append(aug)
                if args.aug_method == "swap":
                    aug = random_swap(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                    aug = " ".join(aug)
                    augs.append(aug)
                if args.aug_method == "simcse":
                    aug = sentence
                    augs.append(aug)
                
            with open("./data/replay/train.txt",'a') as file:
                if args.aug_method in ["del","replace","insert","swap","back_trans","simcse"]:
                    for aug in augs:
                        file.write(aug+"\n")
                file.write(sentence+"\n")

import random

def shuffle_replay():
    training = open("data/replay/train.txt").read().split("\n")
    random.shuffle(training)
    with open("data/replay/train.txt","w") as file:
        for item in training:
            file.write(item + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=300, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--text_chunk', action='store_true', help="")
    parser.add_argument('--use_reverse', action='store_true', help="")
    parser.add_argument('--with_code_loss', type=bool, default=True, help="")
    parser.add_argument('--use_tokenize', action='store_true', help="")
    parser.add_argument("--max_seq", default=80, type=int,help="")
    parser.add_argument("--split", dest='split', default=False, action='store_false')
    parser.add_argument('--mode', type=str, default=None, required = True,help="model type")
    args = parser.parse_args()
    return args