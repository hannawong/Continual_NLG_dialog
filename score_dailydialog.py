from utils.eval_metric import moses_multi_bleu
import json
suffix = "dailydialog_multi"
topic_dic = {1: "Ordinary_Life", 2: "School_Life", 3: "Culture_and_Education",
                        4: "Attitude_and_Emotion", 5: "Relationship", 6: "Tourism" , 7: "Health", 8: "Work", 9: "Politics", 10: "Finance"}
all_blue1 = 0.0; all_blue2 = 0.0; all_blue3 = 0.0; all_blue4 = 0.0 
all_blue = 0.0
for domain in topic_dic.values():
    print(domain)
    my_result = []
    ref = []
    gen_text = json.load(open("/data1/jiayu_xiao/project/wzh/Continual_NLG_dialog/data/"+domain+"/result/"+suffix+".txt"))
    text = open("/data1/jiayu_xiao/project/wzh/Continual_NLG_dialog/data/"+domain+"/test.txt").read().split("\n")[:-1]
    print(len(gen_text),len(text))
    for i in range(len(text)):
        ref.append(text[i].split("__sou__")[1])
        my_result.append(gen_text[i][0])
    BLEU,all = moses_multi_bleu(my_result,ref)
    all_blue1 += all[0];  all_blue2 += all[1]; all_blue3 += all[2]; all_blue4 += all[3]
    all_blue += BLEU

print("BLEU:",all_blue / 10, "BLEU-1:",all_blue1 / 10, "BLEU-2:", all_blue2 / 10, "BLEU-3:", all_blue3 / 10, "BLEU-4:", all_blue4 / 10 )

#### MULTI: ,BLEU-1: 14.37 BLEU-2: 5.17 BLEU-3: 2.92 BLEU-4: 1.85
#### FINETUNE: PPL = 1.8968, BLEU-1: 9.419999999999998 BLEU-2: 1.44 BLEU-3: 0.43999999999999995 BLEU-4: 0.22999999999999998
#### EWC: PPL = 1.9124, BLEU-1: 9.38 BLEU-2: 1.5 BLEU-3: 0.6 BLEU-4: 0.35
#### replay: BLEU: 3.0909999999999997 BLEU-1: 9.55 BLEU-2: 1.72 BLEU-3: 0.71 BLEU-4: 0.39
#### agem: BLEU: 3.116 BLEU-1: 9.870000000000001 BLEU-2: 1.7200000000000002 BLEU-3: 0.61 BLEU-4: 0.29000000000000004

