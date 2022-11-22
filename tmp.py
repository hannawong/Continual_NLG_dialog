'''import os,re
g = os.walk(r"data")  

for path,dir_list,file_list in g:  
    for file_name in file_list: 
        if file_name.endswith("txt") and path.split("/")[-1] in "sgd_alarm,sgd_banks,sgd_calendar,sgd_payment,MWOZ_attraction,sgd_media,sgd_movies,sgd_rentalcars,MWOZ_taxi,sgd_ridesharing,sgd_weather,MWOZ_train": 
            print(os.path.join(path, file_name))
            sentences = open(os.path.join(path, file_name)).read().split("\n")
            outputs = []
            for sentence in sentences:
                if sentence.count("(") > 1 and sentence.count("&") > 1 and "Holiday Inn Express Nashville-I-40&I-24(Spence Ln)" not in sentence:
                    print(sentence)
                    sentence = sentence.split(" & ")[:-1]
                    secend_act = re.split("[.|\?]",sentence[-1])[-1]
                    sentence[1] = " ".join(re.split("[.|\?]",sentence[-1])[:-1])
                    sentence[0] += " @ "+secend_act
                    outputs.append(" & ".join(sentence))
                else:
                    outputs.append(sentence)

            #with open(os.path.join(path, file_name),"w") as file:
            #    for out in outputs:
            #        file.write(out + "\n")

import json
import string

from utils.eval_metric import moses_multi_bleu
from argparse import ArgumentParser
from collections import defaultdict

all_bleurt = 0
def get(gen_text_gem,ref_text):
            gen = []
            for gen_ in gen_text_gem:
                gen_str_list = []
                for i in range(len(gen_)):
                    cl_idx = gen_[i].find('<|endoftext|>')
                    gen_str = gen_[i][:cl_idx].strip().lower()
                    gen_str = gen_str.replace('\xa0','')
                    gen_str_list.append(gen_str)
                gen.append(gen_str_list)
            ############################ SLOT ERR ##############################
            tot = 0
            cnt_bad_tot = 0; cnt_superflous = 0
            bad = []
            my_result = []
            for i in range(len(gen_text_gem)):
                gen_ = gen[i]
                min_bad_cnt = 10000
                ans = ""
                res = []
                for k in range(len(gen_)):
                    cnt_bad = 0
                    cnt_tot = 0
                    line = ref_text[i]
                    line = line.split("&")[0]
                    cl_idx = gen_[k].find('<|endoftext|>')
                    gen_str = gen_[k].strip().lower()[1:].strip()
                    gen_str = gen_str.split("&")[-1].strip()
                    line = line.split(")")
                    for item in line:
                        if "=" not in item:continue
                        item = item.split("(")[1].split(";")
                        for item1 in item:
                            v = item1.split("=")[1].replace("\"","").strip().lower()
                            #for s in string.punctuation:
                            #    v = v.replace(s,'')
                            if(v not in ["true", "false", "yes", "no","none"]):
                                if(v.lower() not in gen_str.lower()):
                                    cnt_bad += 1
                                cnt_tot += 1
                    res.append([cnt_bad,k,cnt_tot])

                res.sort()
                #print("===============================================")
                #print(res)
                #print("gen:",gen_)
                #print("ref:",line)
                ans = gen_[res[0][-2]].strip().lower().strip()#.split("&")[0]
                for s in string.punctuation:
                    ans = ans.replace(s,'')
                ans = ans.strip()
                #ddd
                #print(ans,"***********",ref[i])
                #print(min_bad_cnt)
                my_result.append(ans)
                bad.append(res[0][0])
                #print("===============================================")
            return my_result,bad
def score_folder():
            domain = "MWOZ_attraction"
            print(domain)
            ref_text = open("./data/"+domain+"/test.txt").read().split("\n")
            ref = []
            gen = []
            for line in ref_text:
                if len(line) == 0: continue
                ref_line = line.split("&")[1].strip()
                for s in string.punctuation:
                    ref_line = ref_line.replace(s,'')
                ref.append(ref_line.lower())

            gen_text_gem = json.load(open("./data/"+domain+"/resultagem"+".json"))
            my_result_gem,bad_gem = get(gen_text_gem,ref_text)
            gen_text_bnm = json.load(open("./data/"+domain+"/resultmixupbnmonlyB2"+".json"))
            my_result_bnm,bad_bnm = get(gen_text_bnm,ref_text)
            gen_text_mixup = json.load(open("./data/"+domain+"/resultmixup_bnm_0.0"+".json"))
            my_result_mix,bad_mix = get(gen_text_mixup,ref_text)
                #print("===============================================")
            for i in range(len(gen_text_gem)):
                BLEU_gem = moses_multi_bleu(my_result_gem[i],ref[i])
                BLEU_bnm = moses_multi_bleu(my_result_bnm[i],ref[i])
                BLEU_mix = moses_multi_bleu(my_result_mix[i],ref[i])
                if BLEU_bnm > BLEU_gem and bad_gem[i] >= bad_bnm[i] and bad_bnm[i] <= bad_mix[i] and BLEU_bnm > BLEU_mix:
                    print(my_result_bnm[i],"===",my_result_mix[i],"====",my_result_gem[i],"===",ref[i],"====")
                    print()
            exit()
            ERR = (cnt_bad_tot+cnt_superflous)/float(tot)
            print(len(ref),len(my_result))
            try:
                BLEU = moses_multi_bleu(my_result,ref)
            except:
                BLEU = 0.0
        
        
score_folder()
'''
import os,re
import numpy as np
import time
import torch
from transformers import AdamW
from model.Seq2SeqToD import Seq2SeqToD
import argparse
import torch,json,copy
import torch.nn.functional as F
import numpy as np
from model.Seq2SeqToD import Seq2SeqToD
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}

g = os.walk(r"data")  
domains = ["sgd_alarm","sgd_banks","sgd_calendar","sgd_payment","MWOZ_attraction","sgd_media","sgd_movies","sgd_rentalcars","MWOZ_taxi","sgd_ridesharing","sgd_weather","MWOZ_train"]
da_dic = {}
for domain in domains:
    train = open("data/"+domain+"/train.txt").read().split("\n")
    test = open("data/"+domain+"/test.txt").read().split("\n")
    dev = open("data/"+domain+"/dev.txt").read().split("\n")
    #act_num = np.mean([string.count("@")+1 for string in train] + [string.count("@")+1 for string in test] + [string.count("@")+1 for string in dev])
    #act_num = np.mean([len(string.split()) for string in train] + [len(string.split()) for string in test] + [len(string.split()) for string in dev])
    #print(domain,act_num)
    for item in train+test+dev:
        act = item.split("(")[0].strip().lower()
        if act not in da_dic:
            da_dic[act] = 1
        else:
            da_dic[act] += 1
        if "@" in item:
            act = item.split("@")[-1].split("(")[0].strip().lower()
            if act not in da_dic:
                da_dic[act] = 1
            else:
                da_dic[act] += 1
for key in da_dic.keys():
    if da_dic[key] > 1:
        print(key,da_dic[key],"-----")



def compute_distance():
    device = "cuda:4"
    model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
    tokenizer = tokenizer_class.from_pretrained("./outputs/output_mixup_bnm_1.5")
    args = None
    model = Seq2SeqToD(args)
    model= torch.load("./outputs/output_mixup_bnm_0.0"+"/adapter.ckpt") #0.36579889595618953 0.41428914391089855
    ###2: 0.7880025246270684, 0.8225349200240124
    ###0.0:0.5706878233351527,0.5827821893423017
    model.to(device)
    model.model.eval()
    dic = {} ###{domain: []}
    inter_class_dist = 0
    inter_class_pair_num = 0
    intra_class_dist = 0
    intra_class_pair_num = 0

    for domain in domains:
        dic[domain] = []
        fin = open("./data/"+domain+"/test.txt") #'data/restaurant/test.txt'
        inputs = [i.strip() for i in fin]
        for idx in range(0, len(inputs), 1):
            if idx % 10 == 0:
                print(f"PROGRESS: {int(idx/len(inputs)*100)}%")
            lines = inputs[idx]
            raw_text = lines.split(' & ')[0]  ##inform ( name = arabian nights restaurant ; food = arabian ; goodformeal = dinner ) &
            raw_text = raw_text.lower()
            if len(raw_text) == 0:
                continue
            context = tokenizer.encode(raw_text, add_special_tokens=False)
            context = torch.tensor(context, dtype=torch.long, device=device)  ##shape:([25])
            context = context.unsqueeze(0).repeat(1, 1) ##shape:[5,25]
        
            generated = context
            hidden_state, outputs = model(generated,labels = None,task_id = 0,s = 400) 
            #print(hidden_state.shape)
            #dic[domain].append(hidden_state.detach().cpu().numpy())
            for kk in range(hidden_state.shape[1]):
                #kk = -1
                t = hidden_state[0,kk,:].squeeze() / torch.norm(hidden_state[0,kk,:],p = 2)
                dic[domain].append(t.detach().cpu().numpy())
        for i in range(len(dic[domain])-1):
            for j in range(i+1,len(dic[domain])):
                inter_class_dist += np.dot(dic[domain][i],dic[domain][j])
                inter_class_pair_num += 1
    print(1- inter_class_dist / inter_class_pair_num) ##

    for domain_i in range(len(domains)-1):
        for domain_j in range(domain_i,len(domains)):
            for i in range(len(dic[domains[domain_i]])):
                for j in range(len(dic[domains[domain_j]])):
                    intra_class_dist += np.dot(dic[domains[domain_i]][i],dic[domains[domain_j]][j])
                    intra_class_pair_num += 1
    print(1-intra_class_dist / intra_class_pair_num) ##

    
compute_distance()
    
    