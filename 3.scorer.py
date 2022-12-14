

import json
import string

from utils.eval_metric import moses_multi_bleu
from argparse import ArgumentParser
from collections import defaultdict

perm1 = {0:"['sgd_travel']",1:"['sgd_payment']",2:"['TMA_restaurant']",3:"['TMB_music']",4:"['sgd_ridesharing']",5:"['TMA_auto']",6:"['sgd_music']",7:"['sgd_buses']",8:"['TMB_restaurant']",9:"['MWOZ_attraction']",10:"['TMB_sport']",11:"['sgd_movies']",12:"['sgd_homes']",13:"['TMA_coffee']",14:"['sgd_restaurants']",15:"['sgd_hotels']",16:"['sgd_weather']",17:"['sgd_trains']",18:"['MWOZ_train']",19:"['sgd_flights']",20:"['sgd_media']",21:"['MWOZ_taxi']",22:"['sgd_alarm']",23:"['TMA_movie']",24:"['sgd_banks']",25:"['TMA_pizza']",26:"['TMB_flight']",27:"['sgd_rentalcars']",28:"['TMB_movie']",29:"['sgd_events']",30:"['MWOZ_restaurant']",31:"['sgd_services']",32:"['sgd_calendar']",33:"['TMB_food-ordering']",34:"['MWOZ_hotel']",35:"['TMA_uber']",36:"['TMB_hotel']"}
BLEU_list = []
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
all_blue = 0
all_ter = 0
all_error = 0
all_meteor = 0
all_moverscore = 0
all_bertscore = 0
all = 0

all_bleurt = 0
def score_folder():
    global all_blue
    global all_moverscore
    global all_error
    global all_bertscore
    global all_meteor
    global all_ter
    global all
    global all_bleurt
    out = open("result.txt","w")
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, default="", help="domain")
    parser.add_argument("--mode", type=str, default="adapter")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    if args.domain == 'all':
        perm1 = {0:'sgd_travel',1:'sgd_payment',2:"TMA_restaurant",3:"TMB_music",4:"sgd_ridesharing",5:"TMA_auto",6:"sgd_music",7:"sgd_buses",8:"TMB_restaurant",9:"MWOZ_attraction",10:"TMB_sport",11:"sgd_movies",12:"sgd_homes",13:"TMA_coffee",14:"sgd_restaurants",15:"sgd_hotels",16:"sgd_weather",17:"sgd_trains",18:"MWOZ_train",19:"sgd_flights",20:"sgd_media",21:"MWOZ_taxi",22:"sgd_alarm",23:"TMA_movie",24:"sgd_banks",25:"TMA_pizza",26:"TMB_flight",27:"sgd_rentalcars",28:"TMB_movie",29:"sgd_events",30:"MWOZ_restaurant",31:"sgd_services",32:"sgd_calendar",33:"TMB_food-ordering",34:"MWOZ_hotel",35:"TMA_uber",36:"TMB_hotel"}
        domains = list(perm1.values())
    else:
        domains = args.domain.split(",")
    for domain in domains:
        print(domain)
        out.write(domain+"\n")
        ref_text = open("./data/"+domain+"/test.txt").read().split("\n")
        ref = []
        gen = []
        for line in ref_text:
            if len(line) == 0: continue
            ref_line = line.split("&")[1].strip()
            for s in string.punctuation:
                ref_line = ref_line.replace(s,'')
            ref.append(ref_line.lower())

        gen_text = json.load(open("./data/"+domain+"/result/"+args.suffix+".json"))

        for gen_ in gen_text:
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
        my_result = []
        for i in range(len(gen_text)):
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
                        if(v not in ["true", "false", "yes", "no", "?","none"]):
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
            cnt_bad_tot += res[0][0]
            tot += res[0][-1]
            #print("===============================================")
        ERR = (cnt_bad_tot+cnt_superflous)/float(tot)
        print("ERR",ERR)
        out.write("ERR"+str(ERR)+"\n")
        print(len(ref),len(my_result))
        try:
            BLEU = moses_multi_bleu(my_result,ref)[0]
        except:
            BLEU = 0.0
        print("BLEU",BLEU)
        BLEU_list.append(BLEU)
        all_blue += BLEU
        all += 1
        all_error += ERR
        out.write("BLUE"+str(BLEU)+"\n")
        
        with open("data/"+domain+"/my.txt","w") as file:
                for __ in my_result:
                    file.write(__ + "\n")
        with open("data/"+domain+"/ref.txt","w") as file:
            for __ in ref:
                file.write(__ + "\n")
        import os
        os.system("python prepare_files.py "+"data/"+domain+"/my.txt "+ "data/"+domain+"/ref.txt")
        os.system("java -Xmx2G -jar utils/meteor-1.5/meteor-1.5.jar "+"data/"+domain+"/my.txt"+ " all-notdelex-refs-meteor.txt -l en -norm -r 8 > ../meteor.txt")
        with open('../meteor.txt') as f:
            meteor = float(f.readlines()[-1].strip().split()[-1])
            all_meteor += meteor
            print("METEOR: {:.2f}".format(meteor))
        os.system("java -jar utils/tercom-0.7.25/tercom.7.25.jar -h relexicalised_predictions-ter.txt -r all-notdelex-refs-ter.txt > ../ter.txt")
        os.system("bert-score -r "+" data/"+domain+"/ref.txt -c "+"data/"+domain+"/my.txt --lang en > ../bertscore.txt")
        
        #os.system("python utils/bleurt/score.py -candidate_file="+"/data/jiayu_xiao/project/wzh/CL/Continual_NLG_dialog/data/"+domain+"/ref.txt -reference_file="+"/data/jiayu_xiao/project/wzh/CL/Continual_NLG_dialog/data/"+domain+"/my.txt -bleurt_checkpoint=utils/bleurt/bleurt/test_checkpoint -scores_file=../bleurt.txt")
        with open('../ter.txt') as f:
            ter = float(f.readlines()[-4].strip().split()[2])
            print("TER: {:.2f}".format(ter))
            all_ter += ter
        with open('../bertscore.txt') as f:
            bertscore = float(f.read().strip().split()[-1])
            print("BERTScore F1: {:.2f}".format(bertscore))
            all_bertscore += bertscore
        
        
score_folder()
print("BLEU",all_blue / all, "ERR",all_error / all,"TER",all_ter/all,"BERT",all_bertscore / all,"METEOR",all_meteor/all,"BLEURT",all_bleurt/all)
print(BLEU_list)
