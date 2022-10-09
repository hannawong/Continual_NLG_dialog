from __future__ import absolute_import, division, print_function
import argparse
import glob
import os
import random
from model.Seq2SeqToD import Seq2SeqToD
import numpy as np
import torch
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from utils.util import *
import torch.nn as nn
import copy
from utils.data_augmentation import *

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

prompt = {"sgd_hotels":"find hotel and ratings","sgd_services":"book doctor appointments","sgd_alarm":"set up alarm time",\
            "sgd_events":"show when and where event is","sgd_buses":"confirm leaving time, passengers and fare of buses","sgd_weather":"show temperature and wind","sgd_rentalcars":"rent cars"\
                ,"sgd_calendar":"begin and end time","sgd_travel":"travel","sgd_ridesharing":"share a ride","sgd_media":"media",\
                    "sgd_music":"music","sgd_movies":"find movie and theater","sgd_payment":"make payment to people","sgd_trains":"journey time and money","sgd_banks":"transfer and balance in checking/saving account"}

class TextSeqDataset(Dataset):
    def __init__(self, tokenizer, args, file_paths,max_seq=80,mode = "train",with_lamol = False,with_replay = False, task_id = -1,task_name = "", ):
        self.examples = []
        self.labels = []
        self.masks = []
        if with_replay:
            file_paths.append("replay")
        for filepath in file_paths:
            with open("./data/"+filepath+"/"+mode+".txt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()      
                    raw_str = line.lower() ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced
                    if with_lamol:
                      raw_str_lamol = "["+task_name.split("_")[1]+"] "+ prompt[task_name]+" "+line.lower() if task_name in prompt else  "["+task_name.split("_")[1]+"] "+ task_name.split("_")[1]+" "+line.lower()
                    if len(raw_str.split()) > max_seq -1: ##截断
                        raw_str = ' '.join(raw_str.split()[:max_seq -1])
                    raw_str += ' ' + tokenizer.eos_token ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced <|endoftext|>
                    if with_lamol:
                      if len(raw_str_lamol.split()) > max_seq -1: ##截断
                          raw_str_lamol = ' '.join(raw_str_lamol.split()[:max_seq -1])
                      raw_str_lamol += ' ' + tokenizer.eos_token ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced <|endoftext|>
                    
                    if with_lamol:
                      tokenized_text_lamol = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str_lamol))
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))

                    label = [-1] *  max_seq
                    label[:len(tokenized_text)] = tokenized_text ##raw_str
                    mask = [1] *  max_seq


                    if len(tokenized_text) < max_seq: 
                        mask[-(max_seq - len(tokenized_text)):] = [0] * (max_seq - len(tokenized_text))
                        tokenized_text = tokenized_text + [0] * (max_seq - len(tokenized_text))  ###补零
                    else:
                        tokenized_text = tokenized_text[:max_seq]
                    
                    if with_lamol:
                      if len(tokenized_text_lamol) < max_seq: 
                          tokenized_text_lamol = tokenized_text_lamol + [0] * (max_seq - len(tokenized_text_lamol))  ###补零
                      else:
                          tokenized_text_lamol = tokenized_text_lamol[:max_seq]

                    self.examples.append(tokenized_text)
                    self.masks.append(mask)
                    self.labels.append(label)
                    if mode == "train" and with_lamol:
                        self.examples.append(tokenized_text_lamol)
                        self.masks.append(mask)
                        self.labels.append(mask)

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

def prepare_fisher_for_this_task(args,model,train_dataloader,t_total):
    print("="*10+"Preparing EWC for this task"+"="*10)
    if args.EWC:
      for n, p in model.model.named_parameters():
          model.optpar[n] = torch.Tensor(p.data.cpu()).cuda()
          model.fisher[n] = torch.zeros(p.size()).cuda() #torch.Tensor(p.cpu().data).zero_()

      for _, batch in enumerate(train_dataloader):
          model.model.zero_grad()
          inputs, masks, labels = batch
          inputs = inputs.to(args.device)
          labels = labels.to(args.device)
          loss = model(inputs, labels=labels,task_id = 0)
          loss.backward()
          for n, p in model.model.named_parameters():
              if p.grad is not None:
                  model.fisher[n].data += p.grad.data ** 2

      for name_f,_ in model.fisher.items():
          model.fisher[name_f] /= t_total
      model.model.zero_grad()
      return model

def prepare_optimizer(args,model,tokenizer):
    if args.mode == "GPT2":
        no_decay = ['bias', 'LayerNorm.weight']
    
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate * 0.01, eps=args.adam_epsilon)
    elif args.mode == "adapter":
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if "adapter" in str(n).lower() ], 'weight_decay': args.weight_decay, 'lr':args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if "adapter" not in str(n).lower() ], 'weight_decay': 0.0,'lr':args.learning_rate * 0.01}
    ]
        parameters_to_update = [p for n, p in model.named_parameters() ]#if "adapter" in str(n) or "ln" in str(n) or "lm" in str(n)]
        optimizer = AdamW(optimizer_grouped_parameters,  eps=args.adam_epsilon)
    
    if args.mode == "adapter":
        model.model.resize_token_embeddings(len(tokenizer))
    elif args.mode == "GPT2":
        model.resize_token_embeddings(len(tokenizer))
    return optimizer,model,tokenizer

def train_meta(args,train_dataset,model,tokenizer,task_id,domain):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) ##1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    train_len = len(train_dataloader) ##有几个batch
    
    meta_train_dataset = TextSeqDataset(tokenizer, args, file_paths= [domain], max_seq=args.max_seq,task_id=task_id,task_name=domain,with_lamol=True,with_replay = True)
    meta_train_sampler = RandomSampler(meta_train_dataset)
    meta_train_dataloader = DataLoader(meta_train_dataset, sampler=meta_train_sampler, batch_size= args.train_batch_size)

    print("meta length",len(meta_train_dataloader)) ##12
    print("train_length",len(train_dataloader)) ##11

    meta_trainingset = []
    for step, batch in enumerate(meta_train_dataloader):
        meta_trainingset.append(batch)
   
    global_step = 0; tr_loss = 0.0
    model.model.zero_grad()
    model_copy = Seq2SeqToD(args)
    model_copy = copy.deepcopy(model)
    model_copy.model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  
    for epoch in train_iterator: ##EPOCH
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                print(f"  PROGRESS: {float(global_step)/t_total*100}%")
            model.model.zero_grad()
            model_copy.model.zero_grad()
            inputs, masks, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model = copy.deepcopy(model_copy)
            model.train()
            model_copy.train()
            TIMES = 2
            
            for i in range(TIMES):
                len_input = inputs.shape[0] // TIMES
                loss = model(inputs, labels ,task_id = task_id)  ###inputs:[32,80], labels:[32,80]
                loss.backward()
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if "adapter" in str(n).lower() ], 'weight_decay': 0.0, 'lr':args.learning_rate*0.01},
                    {'params': [p for n, p in model.named_parameters() if "adapter" not in str(n).lower() ], 'weight_decay': 0.0,'lr':args.learning_rate * 0.0001}
                    ]

                optimizer = AdamW(optimizer_grouped_parameters,  eps=args.adam_epsilon)
                parameters_to_update = [p for n, p in model.named_parameters() ]
                torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm)
                optimizer.step() ###fast update 更新参数
                model.model.zero_grad()
            
            metaloss = model(meta_trainingset[step][0].cuda(), labels=meta_trainingset[step][2].cuda(),task_id = task_id)
            metaloss.backward()
            parameters_to_update = [p for n, p in model.named_parameters() ] 
            torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm) ##meta 梯度存储在model中

            if step % 50 == 0:
                print(metaloss)
            
            model_grads_dict = {} ##把目前model中的梯度放到model_copy中去
            for n,f in model.named_parameters():
                model_grads_dict[n] = f.grad
            for n, f in model_copy.named_parameters():
                f.grad = model_grads_dict[n]
            
            optimizer_grouped_parameters_meta = [
            {'params': [p for n, p in model_copy.named_parameters() if "adapter" in str(n).lower() ], 'weight_decay': args.weight_decay, 'lr':args.learning_rate},
            {'params': [p for n, p in model_copy.named_parameters() if "adapter" not in str(n).lower() ], 'weight_decay': 0.0,'lr':args.learning_rate * 0.01}]
            optimizer_meta = AdamW(optimizer_grouped_parameters_meta,  eps=args.adam_epsilon)
            optimizer_meta.step() ##更新model_copy参数
            model_copy.model.zero_grad()

        for s in range(step+1,len(meta_trainingset)):
            inputs, masks, labels = meta_trainingset[s]
            loss = model_copy(inputs.cuda(), labels=labels.cuda(),task_id = task_id)
            loss.backward()
            parameters_to_update = [p for n, p in model_copy.named_parameters() ] 
            torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm)
            optimizer_meta.step()
            model_copy.model.zero_grad()

    for task_id,domain in enumerate(args.eval_data_file.split(",")[:task_id+1]):
            eval_dataset = TextSeqDataset(tokenizer, args, file_paths=[domain], max_seq=args.max_seq,mode = "test",task_id=task_id,task_name=domain)
            evaluate(args, model_copy, eval_dataset,task_id,tokenizer,domain)
    return global_step, tr_loss, model_copy


def generate_text_ans(inputs,tokenizer,model):
      text = ""
      for _ in inputs:
          _ = _.item()
          text += tokenizer.convert_ids_to_tokens(_)
      text = text.replace('Ġ',' ')
      raw_text = text[:-1].split(' & ')[0]# + ' & '
      tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
      generated = torch.LongTensor(tokenized_text).unsqueeze(0).cuda()
      for step in range(60):
          outputs = model(generated,labels = None,task_id = 0)
          next_token_logits = outputs[:,-1, :]
          next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
          generated = torch.cat((generated, next_tokens), dim=1)
      out = generated.tolist() ###只取生成的后面
      for o in out:
          text = tokenizer.decode(o, clean_up_tokenization_spaces=True) ##只取到<endoftext>之前
      return text


def train(args, train_dataset, model, tokenizer,task_id):  ### Train the model
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) ##1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer,model,tokenizer = prepare_optimizer(args,model,tokenizer)
    
    replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True)
    replay_sampler = RandomSampler(replay_dataset)
    replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)

    global_step = 0; tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  
    for epoch in train_iterator: ##EPOCH
        if task_id >= 1:
            mix_step = list(np.random.randint(0,len(train_dataloader)-1,len(replay_dataloader)))
            bnm_step = list(np.random.randint(0,len(train_dataloader)-1,1))

        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                print(f"  PROGRESS: {float(global_step)/t_total*100}%")
            inputs, masks, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()

            loss = model(inputs, labels=labels,task_id = task_id)  ###inputs:[32,80], labels:[32,80]
            ###################### EWC ########################
            if args.EWC and task_id >= 1:  ####add EWC loss to original loss
                ewc_loss = 0
                for n,p in model.model.named_parameters():
                  l = 0.01 * model.fisher[n].cuda() * (p - model.optpar[n].cuda()).pow(2)
                  ewc_loss += l.sum()
                loss = loss + ewc_loss

            ###################### EWC ends ########################
            loss.backward()
            if step % 50 == 0:
                print(loss)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                parameters_to_update = [p for n, p in model.named_parameters() ]
                if args.mode == "GPT2":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                elif args.mode == "adapter" :
                    torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm)
                optimizer.step()
                model.model.zero_grad()
                global_step += 1
            ###2.8376,1.9717 -> 3.2810, 2.4692, 1.9476 -> 3.6967, 2.6558, 2.3392, 1.8834 -> 3.9895, 3.0885
            #####mixup
            if task_id >= 1 and step in mix_step and args.aug_method in ["replace","insert","del","swap"]:
                with open("./data/replay/train.txt") as file:
                    sentences = file.readlines()
                    augs = []
                    for sentence in sentences:
                        print(sentence)
                        if args.aug_method == "replace":
                            aug = synonym_replacement(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                        if args.aug_method == "del":
                            aug = random_deletion(sentence.split(),0.1)
                        if args.aug_method == "insert":
                            aug = random_insertion(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                        if args.aug_method == "swap":
                            aug = random_swap(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                        if args.aug_method == "back_trans":
                            from googletrans import Translator
                            translator = Translator()
                            trans_text = translator.translate(sentence, dest='fr').text
                            aug = translator.translate(trans_text, dest='en').text
                        aug = " ".join(aug)
                        augs.append(aug)
                with open("./data/replay/train.txt",'a') as file:
                    for aug in augs:
                        file.write(aug+"\n")
            if task_id >= 1 and step in mix_step and args.aug_method == "mixup":
                replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True)
                replay_sampler = RandomSampler(replay_dataset)
                replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)
                for _, batch in enumerate(replay_dataloader):
                    if _ >= 1:
                        break
                    inputs_B, masks_B, labels_B = batch
                    inputs_B = inputs_B.to(args.device)
                    labels_B = labels_B.to(args.device)
                    model.train()
                    mix_layer = np.random.choice([4,8,11],1)[0]
                    replay_loss = model(input_ids = inputs, labels=labels, input_ids_prev = inputs_B,labels_prev = labels_B,mix_layer = mix_layer)
                    print("replay_loss",replay_loss)
                    replay_loss *= 0.5
                    replay_loss.backward()
                parameters_to_update = [p for n, p in model.named_parameters() ]

                not_gradient = ["wte","wpe"]
                for i in range(mix_layer):
                    not_gradient.append("transformer.h."+str(i))
                    not_gradient.append("adapter_blocks."+str(i))

                optimizer_grouped_parameters_mix = [
                    {'params': [p for n, p in model.named_parameters() if n not in not_gradient and "adapter" in str(n).lower() ], 'weight_decay': args.weight_decay, 'lr':args.learning_rate},
                    {'params': [p for n, p in model.named_parameters() if n not in not_gradient and "adapter" not in str(n).lower() ], 'weight_decay': 0.0,'lr':args.learning_rate * 0.01}]
                
                optimizer_mix = AdamW(optimizer_grouped_parameters_mix,  eps=args.adam_epsilon)

                torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm)
                optimizer_mix.step()
                model.model.zero_grad()
            ######### mixup finish


            #########BNM
            if task_id >= 1 and step in bnm_step and args.BNM == True:
                replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True)
                replay_sampler = RandomSampler(replay_dataset)
                replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=10000)
                for _, batch in enumerate(replay_dataloader):
                    inputs_B, masks_B, labels_B = batch
                    inputs_B = inputs_B.to(args.device)
                    labels_B = labels_B.to(args.device)
                    model.train()
                    bnm_loss = model(input_ids = inputs, labels=labels, input_ids_prev = inputs_B,labels_prev = labels_B) * 0.5
                    print("bnm_loss",bnm_loss)
                    bnm_loss.backward()
                    parameters_to_update = [p for n, p in model.named_parameters() ]
                    torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm)
                    optimizer.step()
                    model.model.zero_grad()
            

    prepare_fisher_for_this_task(args,model,train_dataloader,t_total)

    for task_id,domain in enumerate(args.eval_data_file.split(",")[:task_id+1]):
        eval_dataset = TextSeqDataset(tokenizer, args, file_paths=[domain], max_seq=args.max_seq,mode = "test",task_id=task_id,task_name=domain)
        evaluate(args, model, eval_dataset,task_id,tokenizer,domain)
    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset,task_id,tokenizer,domain):
        output_tests = []
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        cnt = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, masks, labels = batch
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                if args.mode == "adapter" :
                    outputs =  model(inputs, labels=labels,task_id = task_id,s = 400)
                    lm_loss = outputs
                        
                elif args.mode == "GPT2":
                    outputs =  model(inputs, labels=labels)
                    lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
                nb_eval_steps += 1
                ###### generate answer #######
                if cnt < 5:
                    cnt += 1
                    text = ""
                    for _ in inputs[0]:
                        _ = _.item()
                        text += tokenizer.convert_ids_to_tokens(_)
                    text = text.replace('Ġ',' ')
                    raw_text = text[:-1].split(' & ')[0]# + ' & '
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
                    generated = torch.LongTensor(tokenized_text).unsqueeze(0).cuda()
                    for step in range(60):
                        outputs = model(generated,labels = None,task_id = task_id)
                        next_token_logits = outputs[:,-1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                        generated = torch.cat((generated, next_tokens), dim=1)
                    out = generated.tolist() ###只取生成的后面
                    examples = []
                    for o in out:
                        text = tokenizer.decode(o, clean_up_tokenization_spaces=True) ##只取到<endoftext>之前
                        examples.append(text)
                    print(examples)
                    output_tests.append(examples)


        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {
            "perplexity": perplexity
        }
        print(result)
        return result

def parse_arg():
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
    parser.add_argument("--split",  type=str, default='T')
    parser.add_argument('--mode', type=str, default=None, required = True,help="model type")
    parser.add_argument("--EWC", type=str, default='F')
    parser.add_argument("--aug_method", type=str, default='None')
    parser.add_argument("--BNM", type=str, default='F')
    args = parser.parse_args()
    args.split = args.split == "T"
    args.EWC = args.EWC == "T"
    args.BNM = args.BNM == "T"
    return args

def prepare_for_main():
    args = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.train_data_file == "all":
        perm1 = {0:'sgd_travel',1:'sgd_payment',2:"TMA_restaurant",3:"TMB_music",4:"sgd_ridesharing",5:"TMA_auto",6:"sgd_music",7:"sgd_buses",8:"TMB_restaurant",9:"MWOZ_attraction",10:"TMB_sport",11:"sgd_movies",12:"sgd_homes",13:"TMA_coffee",14:"sgd_restaurants",15:"sgd_hotels",16:"sgd_weather",17:"sgd_trains",18:"MWOZ_train",19:"sgd_flights",20:"sgd_media",21:"MWOZ_taxi",22:"sgd_alarm",23:"TMA_movie",24:"sgd_banks",25:"TMA_pizza",26:"TMB_flight",27:"sgd_rentalcars",28:"TMB_movie",29:"sgd_events",30:"MWOZ_restaurant",31:"sgd_services",32:"sgd_calendar",33:"TMB_food-ordering",34:"MWOZ_hotel",35:"TMA_uber",36:"TMB_hotel"}
        domains = list(perm1.values())
    else:
        domains = args.train_data_file.split(",")
    print(domains)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] ##<class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'> <class 'transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer'>
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, ###False
                                                cache_dir=args.cache_dir if args.cache_dir else None) ##None
    

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence) ##min(80,1024)
    if args.mode == "GPT2":
        model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    elif args.mode == "adapter" :
        model = Seq2SeqToD(args,1)
    model.to(args.device)
    args.split = True
    return args,device,domains,model_class,tokenizer_class,model,tokenizer,config,config_class

def save_model_and_tokenizer(args,model,tokenizer):
  print("Saving model checkpoint to", args.output_dir)
  if args.mode == "GPT2":
      model.save_pretrained(args.output_dir)
  elif args.mode == "adapter":
      torch.save(model, args.output_dir + "/adapter.ckpt")
      tokenizer.save_pretrained(args.output_dir)
      torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) ##save arg

def load_model_and_tokenizer(args,model_class,tokenizer_class):
    print("Loading Model checkpoint.....")
    if args.mode == "GPT2":
        model = model_class.from_pretrained(args.output_dir)
    elif args.mode == "adapter" :
        model = Seq2SeqToD(args)
        model = torch.load(args.output_dir+"/adapter.ckpt")
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)
    return model,tokenizer

def main():
    args,device,domains,model_class,tokenizer_class,model,tokenizer,config,config_class = prepare_for_main()
    ################################# Training ########################################
    if args.split: ###分开训练
        for task_id,domain in enumerate(domains):
            print("======================= domain:",domain,"===========================")
            train_dataset = TextSeqDataset(tokenizer, args, file_paths= [domain], max_seq=args.max_seq,task_id=task_id,task_name=domain,with_lamol=False,with_replay=True)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer,task_id)
            #global_step, tr_loss,model = train_meta(args, train_dataset, model, tokenizer,task_id,domain)
            ########################### 此domain训练完成 ######################################
            print(" global_step = ", global_step," average loss = ", tr_loss)
            generate_replay(args,task_id=task_id+1,domain_names=domains,tokenizer=tokenizer,model = model,mode = "REPLAY",sample_frac=0.01)

        save_model_and_tokenizer(args,model,tokenizer)
    else: ##合在一起
        train_dataset = TextSeqDataset(tokenizer, args, file_paths= domains, max_seq=args.max_seq)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        print(" global_step = ",global_step," average loss = ", tr_loss)
        save_model_and_tokenizer(args,model,tokenizer)
        model,tokenizer = load_model_and_tokenizer(args,model_class,tokenizer_class)


if __name__ == "__main__":
    main()