

from regex import E


domains = ["sgd_alarm","sgd_banks","sgd_calendar","sgd_payment","MWOZ_attraction","sgd_media","sgd_movies","sgd_rentalcars","MWOZ_taxi","sgd_ridesharing","sgd_weather","MWOZ_train"]
one_gram_dic = {} ###1-gram: times
two_gram_dic = {}
three_gram_dic = {}
four_gram_dic = {}
for domain in domains:
    lines = open("/data1/jiayu_xiao/project/wzh/Continual_NLG_dialog/data/"+domain+"/test.txt").read().split("\n")
    Set = set()
    Set_2 = set()
    Set_3 = set()
    Set_4 = set()
    for line in lines:
        ref = line.split("&")[-1].lower().replace("?","").replace(".","").replace(",","").split()
        ###1-gram
        for i in range(len(ref)):
            Set.add(ref[i])
        ### 2-gram
        for i in range(len(ref)-1):
            Set_2.add(ref[i]+" "+ref[i+1])
        #### 3-gram
        for i in range(len(ref)-2):
            Set_3.add(ref[i]+" "+ref[i+1]+" "+ref[i+2])
        #### 4-gram
        for i in range(len(ref)-3):
            Set_4.add(ref[i]+" "+ref[i+1]+" "+ref[i+2]+" "+ref[i+3])

    for item in list(Set):
        if item not in one_gram_dic:
            one_gram_dic[item] = 1
        else:
            one_gram_dic[item] += 1
    for item in list(Set_2):
        if item not in two_gram_dic:
            two_gram_dic[item] = 1
        else:
            two_gram_dic[item] += 1
    for item in list(Set_3):
        if item not in three_gram_dic:
            three_gram_dic[item] = 1
        else:
            three_gram_dic[item] += 1
    for item in list(Set_4):
        if item not in four_gram_dic:
            four_gram_dic[item] = 1
        else:
            four_gram_dic[item] += 1
print(two_gram_dic)
import json
result = {"1-gram":one_gram_dic,"2-gram":two_gram_dic,"3-gram":three_gram_dic,"four_gram":four_gram_dic}
json.dump(result, open("n_gram.json",'w'), indent=2)
