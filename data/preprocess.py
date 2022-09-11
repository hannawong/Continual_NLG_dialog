import json,os

with open("/data/jiayu_xiao/project/wzh/ToDCL/dataset.json") as file:
  dataset = json.load(file)

domains = list(dataset['train'].keys())

def preprocess_1(split):
    for domain in domains:
        print(domain)
        domain_ = domain[2:-2]
        if not os.path.exists(domain_):
            os.mkdir(domain_)
        train_dataset = dataset[split][domain]
        print(len(train_dataset))

        train = []
        for item in train_dataset:
            history_reply = item['history_reply'][:-5].split('[SOS]')
            intent = history_reply[0]
            reply = history_reply[1].strip()
          
            string = ""
            intents = intent.split(')')[:-1]
            for _ in intents:
                    _ = _.replace("\',","\';").strip()
                    _ = _.replace("\",","\";").strip()
                    string += _+")"+"@"
                    train.append([string[:-1],reply,reply])
                    #print(string[:-1],reply)
              
        with open(domain_ + "/"+split+".json",'w') as f:
                json.dump(train,f,indent=4)

preprocess_1("train")
preprocess_1("dev")
preprocess_1("test")

def preprocess_2(split):

  for domain in domains:
    string = ""
    print(domain)
  
    with open("/data/jiayu_xiao/project/wzh/SC-GPT/data/"+domain[2:-2]+"/"+split+".json") as file:
      raw = json.load(file)
    print(len(raw))
    for item_ in raw:
      try:
        s = ""
        item = item_[0].split("@")
        for items in item:
          intent = items.split("(")[0]
          slot = items.split("(")[1][:-1].split(";")
          s += intent + " ( "
          for slots in slot:
            slots = slots.split("=")
            s += slots[0] + " = "
            s += slots[1] + " ; "
          s = s[:-3] + ")"
          s += " & " + item_[1]
          if item_.count("@") > 0:
            print(s)
          string += s+"\n"
      except:
          print(item_)

    with open("/data/jiayu_xiao/project/wzh/SC-GPT/data/"+domain[2:-2]+"/"+split+".txt",'w') as file:
      file.write(string)
      

preprocess_2("train")
preprocess_2("test")
preprocess_2("dev")

    
    

