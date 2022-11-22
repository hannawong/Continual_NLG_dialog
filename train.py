from __future__ import absolute_import, division, print_function
import argparse
import os
import random,json
from model.Seq2SeqToD import Seq2SeqToD
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from utils.util import *
import torch.nn as nn
import copy
import pandas as pd
from utils.data_augmentation import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

topic_dic = {1: "Ordinary_Life", 2: "School_Life", 3: "Culture_and_Education",
                        4: "Attitude_and_Emotion", 5: "Relationship", 6: "Tourism" , 7: "Health", 8: "Work", 9: "Politics", 10: "Finance"}
prompt = {}

class TextSeqDataset(Dataset):
    def __init__(self, tokenizer, args, file_paths,max_seq=80,mode = "train",with_lamol = False,with_replay = False, task_id = -1,task_name = "", examples = []):
        self.examples = []
        self.labels = []
        self.masks = []
        self.length = []
        self.is_replay = []
        if len(examples) > 0:
            for line in examples:
                    line = line.strip()   
                    self.is_replay.append(0)   
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
                    self.length.append(len(tokenized_text) if len(tokenized_text) <= 80 else 80)


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
                    if not with_lamol:
                        self.examples.append(tokenized_text)
                        self.masks.append(mask)
                        self.labels.append(label)
                    if with_lamol:
                        self.examples.append(tokenized_text_lamol)
                        self.masks.append(mask)
                        self.labels.append(mask)

            self.labels = self.examples
            return 

        if with_replay:
            file_paths.append("replay")
        for filepath in file_paths:
            if filepath == "replay": mode = "train_"+args.output_dir.split("/")[-1]
            with open("./data/"+filepath+"/"+mode+".txt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()   
                    if filepath == "replay":
                        self.is_replay.append(1)
                    else:
                        self.is_replay.append(0)   
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
                    self.length.append(len(tokenized_text) if len(tokenized_text) <= 80 else 80)


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
                    if not with_lamol:
                        self.examples.append(tokenized_text)
                        self.masks.append(mask)
                        self.labels.append(label)
                    if with_lamol:
                        self.examples.append(tokenized_text_lamol)
                        self.masks.append(mask)
                        self.labels.append(mask)

        self.labels = self.examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.masks[item]), torch.tensor(self.labels[item]), torch.tensor(self.length[item]), torch.tensor(self.is_replay[item])

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def prepare_fisher_for_this_task(args,model,train_dataloader,t_total):
    print("="*10+"Preparing EWC for this task"+"="*1)
    if args.EWC:
      for n, p in model.model.named_parameters():
          model.optpar[n] = torch.Tensor(p.data.cpu()).cuda()
          model.fisher[n] = torch.zeros(p.size()).cuda() #torch.Tensor(p.cpu().data).zero_()

      for _, batch in enumerate(train_dataloader):
          model.model.zero_grad()
          inputs, masks, labels,length,is_replay = batch
          inputs = inputs.to(args.device)
          labels = labels.to(args.device)
          _,loss = model(inputs, labels=labels,task_id = 0)
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
          _,outputs = model(generated,labels = None,task_id = 0)
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
    if task_id >= 1 and args.replay:
        replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True )
        replay_sampler = RandomSampler(replay_dataset)
        replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)

    global_step = 0; tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args) 
    if args.AGEM and task_id >= 1:
        model.zero_grad()
        replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True)
        replay_sampler = RandomSampler(replay_dataset)
        replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)
        for _, batch in enumerate(replay_dataloader):
                if _ >= 1:
                    break
                inputs_B, masks_B, labels_B,length,is_replay = batch
                inputs_B = inputs_B.to(args.device)
                labels_B = labels_B.to(args.device)
                length = length.to(args.device)
                model.train()
                _,replay_loss = model(inputs_B, labels_B)
        replay_loss.backward()
        grad_ref = []
        for n,p in model.model.named_parameters():
            if p.requires_grad and p.grad!= None:
                grad_ref.append(p.grad.view(-1))
        grad_ref = torch.cat(grad_ref) ## from eq. 10 of AGEM Paper
        model.model.zero_grad()
    for epoch in train_iterator: ##EPOCH
        if task_id >= 1 and args.replay:
            if len(train_dataloader) == 1:
                mix_step = [0]
            else:
                mix_step = list(np.random.randint(0,len(train_dataloader)-1,int(len(replay_dataloader)*1)))

        for step, batch in enumerate(tqdm(train_dataloader)):
            if step % 100 == 0:
                print(f"  PROGRESS: {float(global_step)/t_total*100}%")
            inputs, masks, labels,length,is_replay = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            length = length.to(args.device)
            is_replay = is_replay.to(args.device)

            model.train()

            _,loss = model(inputs, labels=labels,task_id = task_id,BNM = args.BNM and task_id >= 1,length = length,is_replay = is_replay)  ###inputs:[32,80], labels:[32,80]
            #if args.BNM: print("matrix rank is:", _)
            ###################### EWC ########################
            if args.EWC and task_id >= 1:  ####add EWC loss to original loss
                ewc_loss = 0
                for n,p in model.model.named_parameters():
                  l = 0.01 * model.fisher[n].cuda() * (p - model.optpar[n].cuda()).pow(2)
                  ewc_loss += l.sum()
                loss = loss + ewc_loss
            loss.backward()
            ###################### EWC ends ########################
            if args.AGEM and task_id >= 1:
                ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
                grad_cur = []
                for p in model.model.parameters():
                    if p.requires_grad and p.grad != None:
                        grad_cur.append(p.grad.view(-1))
                grad_cur = torch.cat(grad_cur)
                # -check inequality constrain
                angle = (grad_cur*grad_ref).sum()

                if angle < 0:
                    # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                    length_rep = (grad_ref*grad_ref).sum()
                    grad_proj = grad_cur-(angle/length_rep)*grad_ref
                    # -...and replace all the gradients within the model with this projected gradient
                    index = 0
                    for p in model.model.parameters():
                        if p.requires_grad and p.grad != None:
                            n_param = p.numel()  # number of parameters in [p]
                            p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                            index += n_param
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
            '''
            if args.aug_method in ["swap","del","replace","simcse"]  and task_id >= 1 and step in mix_step:
                for time in range(mix_step.count(step)):
                    replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True)
                    replay_sampler = RandomSampler(replay_dataset)
                    replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)
                    for _, batch in enumerate(replay_dataloader):
                        if _ >= 1:
                            break
                        inputs, masks, labels,length,is_replay = batch
                        inputs = inputs.to(args.device)
                        labels = labels.to(args.device)
                        model.train()
                        aug_sentences = []
                        for i in range(inputs.shape[0]):
                            text = ""
                            for _ in inputs[i]:
                                _ = _.item()
                                text += tokenizer.convert_ids_to_tokens(_)
                            sentence = text.replace('Ġ',' ')
                            aug = random_swap(sentence.split(),min(1,int(len(sentence.split())*0.1)))
                            aug = " ".join(aug)
                            aug_sentences.append(aug)
                        replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True,examples = aug_sentences)
                        replay_sampler = RandomSampler(replay_dataset)
                        replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)
                        for step, batch in enumerate(train_dataloader):
                            inputs, masks, labels,length,is_replay = batch
                            inputs = inputs.to(args.device)
                            labels = labels.to(args.device)
                            length = length.to(args.device)
                            is_replay = is_replay.to(args.device)
                            model.train()
                            _,loss = model(inputs, labels=labels,task_id = task_id,BNM = args.BNM and task_id >= 1,length = length,is_replay = is_replay)  ###inputs:[32,80], labels:[32,80]
                            loss.backward()
                            parameters_to_update = [p for n, p in model.named_parameters() ]
                            not_gradient = ["wte","wpe"]

                            optimizer_grouped_parameters_mix = [
                                {'params': [p for n, p in model.named_parameters() if n not in not_gradient and "adapter" in str(n).lower() ], 'weight_decay': args.weight_decay, 'lr':args.learning_rate},
                                {'params': [p for n, p in model.named_parameters() if n not in not_gradient and "adapter" not in str(n).lower() ], 'weight_decay': 0.0,'lr':args.learning_rate * 0.01}]
                            
                            optimizer = AdamW(optimizer_grouped_parameters_mix,  eps=args.adam_epsilon)

                            torch.nn.utils.clip_grad_norm_(parameters_to_update, args.max_grad_norm)
                            optimizer.step()
                            model.model.zero_grad()
                '''     
            if args.aug_method == "mixup" and task_id >= 1 and step in mix_step:
                for time in range(mix_step.count(step)):
                    replay_dataset = TextSeqDataset(tokenizer, args, file_paths= [], max_seq=args.max_seq,task_id=task_id,task_name="",with_lamol=False,with_replay=True)
                    replay_sampler = RandomSampler(replay_dataset)
                    replay_dataloader = DataLoader(replay_dataset, sampler=replay_sampler, batch_size=args.train_batch_size)
                    for _, batch in enumerate(replay_dataloader):
                        if _ >= 1:
                            break
                        inputs_B, masks_B, labels_B,length_B,is_replay = batch
                        inputs_B = inputs_B.to(args.device)
                        labels_B = labels_B.to(args.device)
                        model.train()
                        mix_layer = np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11],1)[0]
                        replay_loss = model(input_ids = inputs, labels=labels, input_ids_prev = inputs_B,labels_prev = labels_B,mix_layer = mix_layer,BNM =args.BNM,length = length)
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
        #evaluate_dailydialog(args,model,tokenizer,["Health"])    
            ######### mixup finish
            
    prepare_fisher_for_this_task(args,model,train_dataloader,t_total)

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
            inputs, masks, labels,length,is_replay = batch
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device)
            length = length.to(args.device)
            with torch.no_grad():
                if args.mode == "adapter" :
                    _,outputs =  model(inputs, labels=labels,task_id = task_id,s = 400)
                    lm_loss = outputs
                        
                elif args.mode == "GPT2":
                    _,outputs =  model(inputs, labels=labels)
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
                        _, outputs = model(generated,labels = None,task_id = task_id)
                        next_token_logits = outputs[:,-1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                        generated = torch.cat((generated, next_tokens), dim=1)
                    out = out[:, len(tokenized_text):].tolist() ###只取生成的后面
                    examples = []
                    for o in out:
                        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                        text = text[: text.find('<|endoftext|>')] ##只取到<endoftext>之前
                        examples.append(text)
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
    parser.add_argument("--layer", default=11, type=int)
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
    parser.add_argument("--dataset",default="CL37", type=str)
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--BNM_ratio", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--alpha", default=0.75, type=float,
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
    parser.add_argument('--seed', type=int, default=0,
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
    parser.add_argument("--replay", type=str, default='F')
    parser.add_argument("--lamol", type=str, default='F')
    parser.add_argument("--AGEM", type=str, default='F')
    parser.add_argument("--only", type=str, default='F')
    parser.add_argument("--train", type=str, default='F')
    parser.add_argument("--test", type=str, default='F')
    args = parser.parse_args()
    args.split = args.split == "T"
    args.EWC = args.EWC == "T"
    args.BNM = args.BNM == "T"
    args.replay = args.replay == "T"
    args.lamol = args.lamol == "T"
    args.AGEM = args.AGEM == "T"
    args.only = args.only == "T"
    args.train = args.train == "T"
    args.test = args.test == "T"
    return args

def prepare_for_main():
    args = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #domain = sgd_travel,sgd_payment,TMA_restaurant,TMB_music,sgd_ridesharing,TMA_auto,sgd_music,sgd_buses,TMB_restaurant,MWOZ_attraction,TMB_sport,sgd_movies,sgd_homes,TMA_coffee,sgd_restaurants,sgd_hotels,sgd_weather,sgd_trains,MWOZ_train,sgd_flights,sgd_media,MWOZ_taxi,sgd_alarm,TMA_movie,sgd_banks,TMA_pizza,TMB_flight,sgd_rentalcars,TMB_movie,sgd_events,MWOZ_restaurant,sgd_services,sgd_calendar,TMB_food-ordering,MWOZ_hotel,TMA_uber,TMB_hotel"}

    '''
    domains = args.train_data_file.split(",")
    shuffle(domains)
    print(domains)
    with open("seed1_domain.txt","w") as f:
        for domain in domains:
            f.write(domain+",")
    '''
    domains = open("seed1_domain.txt").read().split(",")
    if args.dataset == "dailydialog":
        domains = list(topic_dic.values())

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] ##<class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'> <class 'transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer'>
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, ###False
                                                cache_dir=args.cache_dir if args.cache_dir else None) ##None
    special_tokens_dict = {'additional_special_tokens': ['__eou__']}
    tokenizer.add_special_tokens(special_tokens_dict)

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence) ##min(80,1024)
    if args.mode == "GPT2":
        model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    elif args.mode == "adapter" :
        model = Seq2SeqToD(args,1)
    model.to(args.device)
    return args,device,domains,model_class,tokenizer_class,model,tokenizer,config,config_class
res = None
res_after = None
def get_token_representations(model,eval_dataset,args,task_id,tokenizer): ###for each sentences in training set, get all token representations
        global res
        global res_after
        args.eval_batch_size = 32
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        model.eval()
        hidden_list = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, masks, labels,length,is_replay = batch
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device)
            length = length.to(args.device)
            with torch.no_grad():
                hidden,outputs =  model(inputs, labels=labels,task_id = 0,s = 400)
                hidden_ = [hidden[i,1:length[i].long(),:] for i in range(hidden.shape[0])]
                hidden = torch.cat(hidden_,axis = 0)
                hidden = hidden.view(-1,768)
            print(hidden.shape)
            hidden_list.append(hidden)
        if task_id == 0:
            res = np.array(torch.cat(hidden_list,0).cpu().detach())
        else:
            res_after = np.array(torch.cat(hidden_list,0).cpu().detach())
            tot_res =np.concatenate([res,res_after],axis = 0)
            print(tot_res.shape)
            df = pd.DataFrame({'hue':["curr"]*res.shape[0]+["replay"]*res_after.shape[0]})
            X_embedded = TSNE(n_components=2).fit_transform(tot_res)
            print(X_embedded.shape)
            df["tsne_one"] = list(X_embedded[:,0])
            df["tsne_two"] = list(X_embedded[:,1]) 
            sns.scatterplot(x="tsne_one", y="tsne_two",hue="hue",palette=sns.color_palette("hls", 2),data=df,legend="full",alpha=0.3)
            plt.savefig("last_task"+str(args.BNM_ratio)+".png")
def evaluate_dailydialog(args, model, tokenizer, domains):
        for domain in domains:
            print(domain)
            eval_dataset = TextSeqDataset(tokenizer, args,file_paths=[domain],mode = "test")
            output_tests = []
            args.eval_batch_size = args.per_gpu_eval_batch_size
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # Eval!
            eval_loss = 0.0
            nb_eval_steps = 0
            model.eval()
            cnt = 0
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                inputs, masks, labels,length,is_replay = batch
                inputs = inputs.to(args.device)
                masks = masks.to(args.device)
                labels = labels.to(args.device)
                length = length.to(args.device)
                with torch.no_grad():
                    _,lm_loss =  model(inputs, labels=labels,task_id = 0,s = 400)
                            
                    eval_loss += lm_loss.mean().item()
                    nb_eval_steps += 1
                    ###### generate answer #######
                    
                    if cnt < 50000:
                        cnt += 1
                        text = ""
                        for _ in inputs[0]:
                            _ = _.item()
                            text += tokenizer.convert_ids_to_tokens(_)
                        text = text.replace('Ġ',' ').replace("âĢĻ","")
                        raw_text = text[:-1].split('__sou__')[0] + "__sou__"
                        #print("raw_test:",raw_text)
                        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
                        generated = torch.LongTensor(tokenized_text).unsqueeze(0).cuda()
                        for step in range(60):
                            _, outputs = model(generated,labels = None,task_id = 1)
                            next_token_logits = outputs[:,-1, :]
                            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                            generated = torch.cat((generated, next_tokens), dim=1)
                        generated = generated[:, len(tokenized_text):].tolist() ###只取生成的后面
                        examples = []
                        for o in generated:
                            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                            text = text[: text.find('<|endoftext|>')] ##只取到<endoftext>之前
                            examples.append(text)
                            #print(text)
                        output_tests.append(examples)
                        
                    if not os.path.exists("/data1/jiayu_xiao/project/wzh/Continual_NLG_dialog/data/"+domain+"/result/"):
                        os.makedirs("/data1/jiayu_xiao/project/wzh/Continual_NLG_dialog/data/"+domain+"/result/")
                    json.dump(output_tests, open("/data1/jiayu_xiao/project/wzh/Continual_NLG_dialog/data/"+domain+"/result/"+args.output_dir.split("/")[-1]+".txt",'w'), indent=2)
                    

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {
            "perplexity": perplexity
        }
        print(result)
        return result
        
def save_model_and_tokenizer(args,model,tokenizer,task_id = ""):
  print("Saving model checkpoint to", args.output_dir+task_id)
  if not os.path.exists(args.output_dir+task_id):
        os.makedirs(args.output_dir+task_id)
  if args.mode == "GPT2":
      model.save_pretrained(args.output_dir)
  elif args.mode == "adapter":
      torch.save(model, args.output_dir + task_id + "/adapter.ckpt")
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
    file =  open("data/replay/train_"+args.output_dir.split("/")[-1]+".txt","w")
    model.set_number_of_tasks(len(domains))
    if args.AGEM:
        model.set_up_gem() 
    ################################# Training ########################################
    if args.split: 
        if args.train:
            for task_id,domain in enumerate(domains):
                print("======================= domain:",domain,"===========================")
                train_dataset = TextSeqDataset(tokenizer, args, file_paths= [domain], max_seq=args.max_seq,task_id=task_id,task_name=domain,with_lamol=args.lamol,with_replay=args.replay,mode = "train")
                global_step, tr_loss = train(args, train_dataset, model, tokenizer,task_id)
                print(" global_step = ", global_step," average loss = ", tr_loss)
                if args.replay:
                    generate_replay(args,task_id=task_id+1,domain_names=domains,tokenizer=tokenizer,model = model,mode = "REPLAY" if not args.lamol else "LAMOL",sample_size = 5)
            
            for task_id,domain in enumerate(args.eval_data_file.split(",")):
                eval_dataset = TextSeqDataset(tokenizer, args, file_paths=[domain], max_seq=args.max_seq,mode = "test",task_id=task_id,task_name=domain)
                evaluate(args, model, eval_dataset,task_id,tokenizer,domain)
            save_model_and_tokenizer(args,model,tokenizer)
        if args.test:
            model,tokenizer = load_model_and_tokenizer(args,model_class,tokenizer_class)
            if args.dataset == "dailydialog":
                evaluate_dailydialog(args,model,tokenizer,args.eval_data_file.split(","))
    else:
        if args.train:
            train_dataset = TextSeqDataset(tokenizer, args, file_paths= domains, max_seq=args.max_seq,mode = "train")
            global_step, tr_loss = train(args, train_dataset, model, tokenizer,0)
            print(" global_step = ",global_step," average loss = ", tr_loss)
            save_model_and_tokenizer(args,model,tokenizer)
        if args.test:
            model,tokenizer = load_model_and_tokenizer(args,model_class,tokenizer_class)
            if args.dataset == "dailydialog":
                evaluate_dailydialog(args,model,tokenizer,args.eval_data_file.split(","))


if __name__ == "__main__":
    main()