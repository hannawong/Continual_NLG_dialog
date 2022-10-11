import os,re
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
