from __future__ import absolute_import, division, print_function
import argparse
import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                                  T5Config,T5ForConditionalGeneration,T5Tokenizer)
                                

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    't5':(T5Config,T5ForConditionalGeneration,T5Tokenizer)
}

class TextSeqDataset(Dataset):
    def __init__(self, tokenizer, args, file_paths,max_seq=80,mode = "train"):
        self.examples = []
        self.labels = []
        self.masks = []
        for filepath in file_paths:
            with open("./data/"+filepath+"/"+mode+".txt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()      
                    raw_str = line.lower() ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced
                    if len(raw_str.split()) > max_seq -1: ##截断
                        raw_str = ' '.join(raw_str.split()[:max_seq -1])
                    raw_str += ' ' + tokenizer.eos_token ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced <|endoftext|>
                    if args.use_tokenize: ###True
                        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                    else:
                        tokenized_text = tokenizer.convert_tokens_to_ids(raw_str.split())

                    label = [-1] *  max_seq
                    label[:len(tokenized_text)] = tokenized_text ##raw_str
                    mask = [1] *  max_seq


                    if len(tokenized_text) < max_seq: ##pad
                        mask[-(max_seq - len(tokenized_text)):] = [0] * (max_seq - len(tokenized_text))
                        tokenized_text = tokenized_text + [0] * (max_seq - len(tokenized_text))  ###补零
                    else:
                        tokenized_text = tokenized_text[:max_seq]
                    
                    self.examples.append(tokenized_text)
                    self.masks.append(mask)
                    self.labels.append(label)

        self.labels = self.examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.masks[item]), torch.tensor(self.labels[item])


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer):  ### Train the model
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) ##1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    model.resize_token_embeddings(len(tokenizer))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  
    for epoch in train_iterator:
        for step, batch in enumerate(train_dataloader):
            print(f"  PROGRESS: {float(global_step)/t_total*100}%")
            inputs, masks, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    print(f"  EVALERR:  {(tr_loss - logging_loss)/float(args.logging_steps)}")
                    logging_loss = tr_loss

                if global_step % args.save_steps == 0:  ###5000
                    checkpoint_prefix = 'checkpoint' # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)  
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    print("Saving model checkpoint to", output_dir)

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset):

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, masks, labels = batch
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {
            "perplexity": perplexity
        }
        print(result)
        return result


def main():
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
    parser.add_argument("--block_size", default=80, type=int,
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
    parser.add_argument('--seed', type=int, default=42,
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
    args = parser.parse_args()


    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] ##<class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'> <class 'transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer'>
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, ###False
                                                cache_dir=args.cache_dir if args.cache_dir else None) ##None
    

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence) ##min(80,1024)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)
    print("args:", args)
  
    ################################# Training ########################################
    domains = args.train_data_file.split(",")
    if args.split: ###分开训练
        for domain in domains:
            print("======================= domain:",domain,"===========================")
            train_dataset = TextSeqDataset(tokenizer, args, file_paths= [domain], max_seq=args.max_seq)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            print(" global_step = ",global_step," average loss = ", tr_loss)

            print("Saving model checkpoint to", args.output_dir)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) ##save arg

            # Load a trained model and vocabulary that you have fine-tuned
            print("Loading Model checkpoint.....")
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            model.to(args.device)
    else: ##合在一起
        train_dataset = TextSeqDataset(tokenizer, args, file_paths= domains, max_seq=args.max_seq)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        print(" global_step = ",global_step," average loss = ", tr_loss)

        print("Saving model checkpoint to", args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) ##save arg

            # Load a trained model and vocabulary that you have fine-tuned
        print("Loading Model checkpoint.....")
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    #############################  Evaluation  ###################################
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    print("Evaluate the following checkpoints:", checkpoints)

    for checkpoint in checkpoints:
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        domains = args.eval_data_file.split(",")
        for domain in domains:
            eval_dataset = TextSeqDataset(tokenizer, args, file_paths=[domain], max_seq=args.max_seq,mode = "test")
            evaluate(args, model, eval_dataset)    

if __name__ == "__main__":
    main()
