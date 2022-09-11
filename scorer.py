import re
import json
from utils.eval_metric import moses_multi_bleu
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
from tabulate import tabulate
import glob
import os.path
from tqdm import tqdm
from dictdiffer import diff


def parse_API(text):
    API = defaultdict(lambda:defaultdict(str))
    for function in text.split(") "):
        if(function!=""):
            if("(" in function and len(function.split("("))==2):
                intent, parameters = function.split("(")
                parameters = sum([s.split(';') for s in parameters.split("=")],[])
                if len(parameters)>1:
                    if len(parameters) % 2 != 0:
                        parameters = parameters[:-1]

                    for i in range(0,len(parameters),2):
                        API[intent][parameters[i]] = parameters[i+1].replace('"',"")

                if(len(API)==0): API[intent]["none"] = "none"
    return API




def evaluate_EER(args,results_dict,entities_json,path, names):
    ERR = []
    cnt_bad = 0
    cnt_superflous = 0
    tot = 0
    
    for d in results_dict:
        if(d["spk"]=="SYSTEM"):
            ent = set()
            ent_corr = []
            if args.task_type == "E2E":
                d['hist'] = d['hist'].split("API-OUT: ")[1]
                if(d['hist']==""):
                    continue

            for speech_act, slot_value_dict in parse_API(d['hist']+" ").items():
                tot += len(slot_value_dict.keys())
                for s,v in slot_value_dict.items():
                    v = v.strip().lower()
                    print(v)
                    if(v not in ["True", "False", "yes", "no", "?","none"]):
                        if(v.lower() not in d["genr"].lower()):
                            cnt_bad += 1
                        else:
                            ent_corr.append(v.lower())
                        ent.add(v.lower())
                

    return (cnt_bad+cnt_superflous)/float(tot)


def evaluate(args,path,names,ent={}):
    results_json = json.load(open(path))
    entities_json = ent
    acc = 0
    if("ADAPTER" in path):
        acc = []
        for r in results_json:

            ts_id_gold = names.index(eval(r['task_id'])[0])
            if(ts_id_gold == r['pred_task_id']):
                acc.append(1)
            else:
                acc.append(0)
        acc = np.mean(acc)
        # print("ACC:",np.mean(acc))



    domain_BLEU = defaultdict(lambda: defaultdict(list))
    domain_API = defaultdict(lambda: defaultdict(list))
    domain_NLG = defaultdict(list)
    for r in results_json:
        if(r['spk']=='SYSTEM'):
            domain_BLEU[r['task_id']]["pred"].append(r['genr'].strip())
            domain_BLEU[r['task_id']]["gold"].append(r['gold'].replace("[eos]","").strip())
            domain_NLG[r['task_id']].append(r)
        elif(r['spk']=='API'):
            domain_API[r['task_id']]["pred"].append(r['genr'])
            domain_API[r['task_id']]["gold"].append(r['gold'])

    T_BLEU = {}
    T_NLG = {}
    if args.task_type =="NLG" or args.task_type =="E2E":
        for k, sample_NLG in domain_NLG.items():
            T_NLG[k] = evaluate_EER(args,sample_NLG,entities_json, path, names)
        for k,v in domain_BLEU.items():
            T_BLEU[k] = moses_multi_bleu(v["pred"],v["gold"])
    return {"BLEU":T_BLEU, "EER":T_NLG, "ACC":acc}


perm1 = {0:"['sgd_travel']",1:"['sgd_payment']",2:"['TMA_restaurant']",3:"['TMB_music']",4:"['sgd_ridesharing']",5:"['TMA_auto']",6:"['sgd_music']",7:"['sgd_buses']",8:"['TMB_restaurant']",9:"['MWOZ_attraction']",10:"['TMB_sport']",11:"['sgd_movies']",12:"['sgd_homes']",13:"['TMA_coffee']",14:"['sgd_restaurants']",15:"['sgd_hotels']",16:"['sgd_weather']",17:"['sgd_trains']",18:"['MWOZ_train']",19:"['sgd_flights']",20:"['sgd_media']",21:"['MWOZ_taxi']",22:"['sgd_alarm']",23:"['TMA_movie']",24:"['sgd_banks']",25:"['TMA_pizza']",26:"['TMB_flight']",27:"['sgd_rentalcars']",28:"['TMB_movie']",29:"['sgd_events']",30:"['MWOZ_restaurant']",31:"['sgd_services']",32:"['sgd_calendar']",33:"['TMB_food-ordering']",34:"['MWOZ_hotel']",35:"['TMA_uber']",36:"['TMB_hotel']"}

def score_folder():
    ref_text = open("/data/jiayu_xiao/project/wzh/SC-GPT/data/TMB_music/test.txt").read().split("\n")
    ref = []
    gen = []
    for line in ref_text:
        if len(line) == 0: continue
        ref_line = line.split("&")[1].strip()
        ref.append(ref_line.lower())
    gen_text = json.load(open("/data/jiayu_xiao/project/wzh/SC-GPT/data/TMB_music/results.json"))
    for gen_ in gen_text:
        
        cl_idx = gen_[0].find('<|endoftext|>')
        gen_str = gen_[0][:cl_idx].strip().lower()
        gen_str = gen_str.replace('\xa0','')
        gen.append(gen_str)
    print(len(ref),len(gen))
    print(ref[0],"===",gen[0])

    BLEU = moses_multi_bleu(gen,ref)
    print(BLEU)
    ############################ SLOT ERR ##############################
    tot = 0
    cnt_bad = 0; cnt_superflous = 0
    print(len(gen_text),len(ref_text))
    for i in range(len(gen_text)):
        line = ref_text[i]
        gen_ = gen_text[i]
        cl_idx = gen_[0].find('<|endoftext|>')
        gen_str = gen_[0][:cl_idx].strip().lower()
        for speech_act, slot_value_dict in parse_API(line.split("&")[0]+" ").items():
            tot += len(slot_value_dict.keys())
            for s,v in slot_value_dict.items():
                v = v.strip().lower()
                if(v not in ["true", "false", "yes", "no", "?","none"]):
                    if(v.lower() not in gen_str.lower()):
                        cnt_bad += 1
    ERR = (cnt_bad+cnt_superflous)/float(tot)
    print(ERR)
    exit()

    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path to the folder with the results")
    parser.add_argument("--task_type", type=str, default="E2E", help="Path to the folder with the results")
    
    args = parser.parse_args()
    folders = glob.glob(f"{args.model_checkpoint}/*")

    names = list(perm1.keys())
    RESULT = []
    for folder in folders:
        if "png" in folder or "TOO_HIGH_LR" in folder or "TEMP" in folder:
            continue
        res = evaluate(args,f'{folder}/FINAL/generated_responses.json',names)#, ent=entities_json)
        if(args.task_type == "INTENT"):
            INTENT = np.mean([v["intent_accuracy"] for k,v in res["API"].items()])
            RESULT.append({"Name":folder.split("/")[-1].split("_")[0],"INTENT":INTENT})
        elif(args.task_type == "DST"):
            JGA = np.mean([v["turn_level_joint_acc"] for k,v in res["API"].items()])
            RESULT.append({"Name":folder.split("/")[-1].split("_")[0],"JGA":JGA})
        elif(args.task_type == "NLG"):
            BLEU = np.mean([v for k,v in res["BLEU"].items()])
            EER = np.mean([v for k,v in res["EER"].items()])
            RESULT.append({"Name":folder.split("/")[-1].split("_")[0],"BLEU":BLEU,"EER":EER})
        elif(args.task_type == "E2E"):
            INTENT = np.mean([v["intent_accuracy"] for k,v in res["API"].items()])
            JGA = np.mean([v["turn_level_joint_acc"] for k,v in res["API"].items()])
            BLEU = np.mean([v for k,v in res["BLEU"].items()])
            EER = np.mean([v for k,v in res["EER"].items()])
            RESULT.append({"Name":folder.split("/")[-1].split("_")[0],"INTENT":INTENT,"JGA":JGA,"BLEU":BLEU,"EER":EER})

    print(tabulate(RESULT, headers="keys",tablefmt="github"))


score_folder()
