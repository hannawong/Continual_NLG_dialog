from utils.eval_metric import moses_multi_bleu
import json
suffix = "dailydialog_mixup1.0_1.5"
print("suffix",suffix)
topic_dic = {1: "Ordinary_Life", 2: "School_Life", 3: "Culture_and_Education",
                        4: "Attitude_and_Emotion", 5: "Relationship", 6: "Tourism" , 7: "Health", 8: "Work", 9: "Politics", 10: "Finance"}
all_blue1 = 0.0; all_blue2 = 0.0; all_blue3 = 0.0; all_blue4 = 0.0 
all_blue = 0.0
all_meteor = 0.0; all_ter = 0.0; all_bertscore = 0.0
blues = []
for domain in list(topic_dic.values()):
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
    blues.append(BLEU)
    
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
    
print("BLEU",all_blue / 10,"TER",all_ter/10,"BERT",all_bertscore / 10,"METEOR",all_meteor/10)
print(blues)
exit()
print("BLEU:",all_blue / 10, "BLEU-1:",all_blue1 / 10, "BLEU-2:", all_blue2 / 10, "BLEU-3:", all_blue3 / 10, "BLEU-4:", all_blue4 / 10 )
###multi:BLEU: 7.228999999999999 BLEU-1: 14.280000000000001 BLEU-2: 6.309999999999999 BLEU-3: 4.659999999999999 BLEU-4: 3.679999999999999
###replay: BLEU 2.258 TER 1.1980430241138422 BERT 0.8528544 METEOR 0.06309071981161109
###simcse:BLEU: 2.193 BLEU-1: 7.38 BLEU-2: 0.93 BLEU-3: 0.30999999999999994 BLEU-4: 0.16
##mixup0.1:BLEU: 2.0599999999999996 BLEU-1: 7.0200000000000005 BLEU-2: 0.78 BLEU-3: 0.27999999999999997 BLEU-4: 0.12999999999999998
###mixup0.2: BLEU: 2.1499999999999995 BLEU-1: 7.279999999999999 BLEU-2: 0.86 BLEU-3: 0.32999999999999996 BLEU-4: 0.17
###mixup0.3:BLEU: 2.346 BLEU-1: 7.769999999999999 BLEU-2: 1.05 BLEU-3: 0.4 BLEU-4: 0.19
###mixup0.4: BLEU: 2.191 BLEU-1: 7.4 BLEU-2: 0.9 BLEU-3: 0.30999999999999994 BLEU-4: 0.16999999999999998
###mixup0.5:BLEU: 2.2800000000000002 BLEU-1: 7.4 BLEU-2: 1.0 BLEU-3: 0.45 BLEU-4: 0.26
###**mixup 0.6**:BLEU: 2.342 BLEU-1: 7.739999999999999 BLEU-2: 1.0799999999999998 BLEU-3: 0.38 BLEU-4: 0.18
###mixup0.7: BLEU: 2.1260000000000003 BLEU-1: 7.25 BLEU-2: 0.82 BLEU-3: 0.28 BLEU-4: 0.13999999999999999
###mixup0.8: BLEU: 2.2800000000000002 BLEU-1: 7.4 BLEU-2: 1.0 BLEU-3: 0.45 BLEU-4: 0.26
##mixup0.9:BLEU: 2.1049999999999995 BLEU-1: 7.260000000000001 BLEU-2: 0.82 BLEU-3: 0.25 BLEU-4: 0.09
##mixup1.0:BLEU: 2.109 BLEU-1: 7.219999999999999 BLEU-2: 0.8300000000000001 BLEU-3: 0.24 BLEU-4: 0.13999999999999999

###bnmpool0.2: BLEU: 2.3 BLEU-1: 7.619999999999999 BLEU-2: 0.97 BLEU-3: 0.41 BLEU-4: 0.22999999999999998
### bnmpool0.4: BLEU: 2.323 BLEU-1: 7.609999999999999 BLEU-2: 1.0900000000000003 BLEU-3: 0.39 BLEU-4: 0.2
### bnmpool0.6: BLEU: 2.5159999999999996 BLEU-1: 8.190000000000001 BLEU-2: 1.2 BLEU-3: 0.4800000000000001 BLEU-4: 0.22999999999999998
### bnmpool0.7: BLEU: 2.3080000000000003 BLEU-1: 7.69 BLEU-2: 1.0 BLEU-3: 0.37 BLEU-4: 0.16999999999999998
### lamol: BLEU 2.258 TER 1.1980430241138422 BERT 0.8528544 METEOR 0.06309071981161109
### agem: BLEU: 2.1870000000000003 BLEU-1: 7.38 BLEU-2: 0.9199999999999999 BLEU-3: 0.29000000000000004 BLEU-4: 0.12000000000000002