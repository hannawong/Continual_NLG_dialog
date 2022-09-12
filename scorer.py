import json
from py_compile import _get_default_invalidation_mode
from utils.eval_metric import moses_multi_bleu
from argparse import ArgumentParser
from collections import defaultdict

perm1 = {0:"['sgd_travel']",1:"['sgd_payment']",2:"['TMA_restaurant']",3:"['TMB_music']",4:"['sgd_ridesharing']",5:"['TMA_auto']",6:"['sgd_music']",7:"['sgd_buses']",8:"['TMB_restaurant']",9:"['MWOZ_attraction']",10:"['TMB_sport']",11:"['sgd_movies']",12:"['sgd_homes']",13:"['TMA_coffee']",14:"['sgd_restaurants']",15:"['sgd_hotels']",16:"['sgd_weather']",17:"['sgd_trains']",18:"['MWOZ_train']",19:"['sgd_flights']",20:"['sgd_media']",21:"['MWOZ_taxi']",22:"['sgd_alarm']",23:"['TMA_movie']",24:"['sgd_banks']",25:"['TMA_pizza']",26:"['TMB_flight']",27:"['sgd_rentalcars']",28:"['TMB_movie']",29:"['sgd_events']",30:"['MWOZ_restaurant']",31:"['sgd_services']",32:"['sgd_calendar']",33:"['TMB_food-ordering']",34:"['MWOZ_hotel']",35:"['TMA_uber']",36:"['TMB_hotel']"}

def parse_API(text):
    API = defaultdict(lambda:defaultdict(str))
    for function in text.split(") "):
        if(function!=""):
            if("(" in function and len(function.split("("))==2):
                intent, parameters = function.split("(")
                parameters = sum([s.split('",') for s in parameters.split("=")],[])
                if len(parameters)>1:
                    if len(parameters) % 2 != 0:
                        parameters = parameters[:-1]

                    for i in range(0,len(parameters),2):
                        API[intent][parameters[i]] = parameters[i+1].replace('"',"")

                if(len(API)==0): API[intent]["none"] = "none"
    return API

def score_folder():
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, default="", help="domain")
    args = parser.parse_args()
    domains = args.domain.split(",")
    for domain in domains:
        print(domain)
        ref_text = open("./data/"+domain+"/test.txt").read().split("\n")
        ref = []
        gen = []
        for line in ref_text:
            if len(line) == 0: continue
            ref_line = line.split("&")[1].strip()
            ref.append(ref_line.lower())
        gen_text = json.load(open("./data/"+domain+"/result.json"))
        for gen_ in gen_text:
            cl_idx = gen_[0].find('<|endoftext|>')
            gen_str = gen_[0][:cl_idx].strip().lower()
            gen_str = gen_str.replace('\xa0','')
            gen.append(gen_str)

        BLEU = moses_multi_bleu(gen,ref)
        print("BLEU",BLEU)
        ############################ SLOT ERR ##############################
        tot = 0
        cnt_bad = 0; cnt_superflous = 0
        for i in range(len(gen_text)):
            line = ref_text[i]
            line = line.split("&")[0]
            gen_ = gen_text[i]
            cl_idx = gen_[0].find('<|endoftext|>')
            gen_str = gen_[0][:cl_idx].strip().lower()[1:].strip()
            line = line.split(")")
            for item in line:
                if "=" not in item:continue
                item = item.split("(")[1].split(";")
                for item1 in item:
                    v = item1.split("=")[1].replace("\"","").strip().lower()
                    if(v not in ["true", "false", "yes", "no", "?","none"]):
                        if(v.lower() not in gen_str.lower()):
                            cnt_bad += 1
                        else:
                            tot += 1
        ERR = (cnt_bad+cnt_superflous)/float(tot)
        print("ERR",ERR)

score_folder()
