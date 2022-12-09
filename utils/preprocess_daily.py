topic_dic = {1: "Ordinary_Life", 2: "School_Life", 3: "Culture_and_Education",
                        4: "Attitude_and_Emotion", 5: "Relationship", 6: "Tourism" , 7: "Health", 8: "Work", 9: "Politics", 10: "Finance"}
'''
for topic in topic_dic.values():
    for split in ["test","train","val"]:
        output_lines = []
        text = open("../data/"+topic+"/"+split+"_ori.txt").read().split("\n")[:-1]
        for line in text:
            utterances = line.split("__eou__")[:-1]
            #print(utterances)
            for i in range(3,len(utterances)):
                history = "__eou__".join(utterances[i-3:i]).strip()
                next_utter = utterances[i].strip()
                output_lines.append(history+" __sou__ "+next_utter)
        with open("../data/"+topic+"/"+split+".txt","w") as f:
            for line in output_lines:
                f.write(line+"\n")
'''
## statistics
for topic in topic_dic.values():
    tot_len = 0
    tot = 0
    for split in ["train","val","test"]:
        output_lines = []
        text = open("../data/"+topic+"/"+split+"_ori.txt").read().split("\n")[:-1]
        print(topic,split,len(text))
        for line in text:
            utterances = line.split("__eou__")[:-1]
            for utterance in utterances:
                utterances_tokens = utterance.split()
                #print(utterances_tokens)
                tot_len += len(utterances_tokens)
                tot += 1
    print(tot_len/tot)

        
