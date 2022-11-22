from __future__ import absolute_import, division, print_function, unicode_literals
import threading
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
import torch.multiprocessing as mp # 这里不需要使用pytorch的multiprocessing
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm
from utils.util import *
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}
tot = 0
err = 0

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_type", default=None, type=str, required=True,
                      help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                      help="")
  parser.add_argument("--prompt", type=str, default="")
  parser.add_argument("--padding_text", type=str, default="")
  parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
  parser.add_argument("--length", type=int, default=40)
  parser.add_argument("--mode", type = str, default="GPT2")
  parser.add_argument("--suffix", type = str, default="")
  parser.add_argument("--num_samples", type=int, default=1)
  parser.add_argument("--temperature", type=float, default=1.0,
                      help="temperature of 0 implies greedy sampling")
  parser.add_argument("--repetition_penalty", type=float, default=1.0,
                      help="primarily useful for CTRL model; in that case, use 1.2")
  parser.add_argument("--top_k", type=int, default=0)
  parser.add_argument("--top_p", type=float, default=1)
  parser.add_argument("--no_cuda", action='store_true',
                      help="Avoid using CUDA when available")
  parser.add_argument('--seed', type=int, default=1,
                      help="random seed for initialization")
  parser.add_argument('--stop_token', type=str, default=None,
                      help="Token at which text generation is stopped")
  parser.add_argument('--device', type=str, default="cuda",
                      help="Token at which text generation is stopped")
  parser.add_argument('--input_file', type=str, default=None,
                      help="file")

  parser.add_argument('--output_file', type=str, default=None,
                      help="file")

  parser.add_argument('--nc', type=int, default=1,
                      help="number of sentence")
  args = parser.parse_args()
  args.suffix = "".join(args.model_name_or_path.split("_")[1:])
  args.n_gpu = torch.cuda.device_count()
  args.model_type = args.model_type.lower()
  args.no_cuda = False
  return args

args = get_parser()
set_seed(args)
model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
model = Seq2SeqToD(args)
model= torch.load(args.model_name_or_path+"/adapter.ckpt")
model.cuda()
model.model.eval()
print("load model!!!!")




def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 
            logits: shape: [5,50527].
            top_k: keep only top k tokens with highest probability (top-k filtering).
            top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                 (http://arxiv.org/abs/1904.09751)
    """

    # top-k filtering: Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value ##大部分都是-inf,只有topk个不是-inf。 shape = [5,50527]
    ## top-p filtering:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True) ##sorted_logits = [-2.0444, -4.1561, -4.8751,  ...,    -inf,    -inf,    -inf]重复5次；sorted_indices = [ 1222,   425,   532,  ..., 16760, 16761, 16762]重复5次
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) ###[0.7994, 0.8962, 0.9433,  ..., 1.0000, 1.0000, 1.0000]重复5次
        # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p  ###[False, False,  True,  ...,  True,  True,  True]重复5次
        # 保证第一个一定是False，即不过滤掉。Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False ##第一个一定是False

        # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(args,model, length, context, num_samples=1, temperature=1.1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu',task_id = -1):
    assert task_id != -1
    ##length:80, num_samples:5, temperature = 1.0,top_k = 5, top_p = 0.9,repetition_penalty = 1.0
    context = torch.tensor(context, dtype=torch.long, device=device)  ##shape:([25])
    context = context.unsqueeze(0).repeat(num_samples, 1) ##shape:[5,25]
    
    generated = context
    
    with torch.no_grad():
        for _ in range(length): ##length = 80
            if args.mode == "GPT2":
                inputs = {'input_ids': generated} ###shape:[5,25]
            elif args.mode == "adapter":
                inputs = {'input_ids': generated, 'labels':None, 'task_id':task_id}
            if args.mode == "GPT2":
                _, outputs = model(**inputs) ###outputs[0].shape: [5,25,50527] 
                next_token_logits = outputs[0][:, -1, :] #/ temperature ###[5,50527]
            elif args.mode == "adapter" or args.mode == "ctr":
                _, outputs = model(generated,labels = None,task_id = task_id,s = 400)
                next_token_logits = outputs[:,-1, :]

            #next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  ###[5,50527], 只是大部分都变成了-inf
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1) ##[1222]重复5次
            generated = torch.cat((generated, next_token), dim=1) ###[5,26]
    return generated

final_ans = []

class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, counter,domain_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.domain_name = domain_name
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print("Starting " + self.name)
        generate_thread(self.domain_name)
        print("Exiting " + self.name)
        

            

def generate_thread(domainname):
        fin = open("./data/"+domainname+"/test.txt") #'data/restaurant/test.txt'
        inputs = [i.strip() for i in fin]
        output_tests = []
        for idx in tqdm(range(0, len(inputs), 1)):
            #if idx % 10 == 0:
            #    print(f"PROGRESS: {int(idx/len(inputs)*100)}%",domainname)
            #start = time.time()
            lines = inputs[idx]
            raw_text = lines.split(' & ')[0]  ##inform ( name = arabian nights restaurant ; food = arabian ; goodformeal = dinner ) &
            raw_text = raw_text.lower()
            #print(raw_text)
            if len(raw_text) == 0:
                continue
            '''
            copy_model = copy.deepcopy(model)
            
            similar_text,similarity = get_top_k(raw_text, 2, 1.1)
            #print(similar_text)
            if len(similar_text) != 0:
              train_dataset = TextSeqDataset(tokenizer, args, similar_text)
              train_dataloader = DataLoader(train_dataset,batch_size=len(similar_text))
              for epoch in range(5):
                for step, batch in enumerate(train_dataloader):
                  inputs_ids,mask,labels = batch
                  inputs_ids = inputs_ids.cuda()
                  labels = labels.cuda()
                  copy_model.train()
                  loss = copy_model(inputs_ids, labels=labels,task_id = 0)
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(copy_model.parameters(),1.0)
                  optimizer_grouped_parameters = [
                    {'params': [p for n, p in copy_model.named_parameters() if "adapter" not in str(n).lower() ], 'weight_decay': 0.0,'lr':0.00625 * 0.0001}
                  ]
                  parameters_to_update = [p for n, p in copy_model.named_parameters() ]#if "adapter" in str(n) or "ln" in str(n) or "lm" in str(n)]
                  optimizer = AdamW(optimizer_grouped_parameters,  eps=1e-8)
                  optimizer.step()
                  copy_model.model.zero_grad()
            '''
            context_tokens = tokenizer.encode(raw_text, add_special_tokens=False) ##[259, 687, 357, 1438, 796, 610, 397, 666, 12513, 7072, 2162, 2057, 796, 610, 397, 666, 2162, 922, 687, 2287, 796, 8073, 1267, 1222, 220]
            out = sample_sequence(
                args,
                model=model,
                context=context_tokens,
                num_samples=args.num_samples,
                length=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                is_xlnet=bool(args.model_type == "xlnet"),
                is_xlm_mlm=False,
                xlm_mask_token=False,
                xlm_lang=False,
                device=args.device,
                task_id = 0
            )
            out = out[:, len(context_tokens):].tolist() ###只取生成的后面
            examples = []
            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = text[: text.find('<|endoftext|>')] ##只取到<endoftext>之前
                examples.append(text)
            output_tests.append(examples)
            #end = time.time()
            #print(end-start)
        import os
        if not os.path.exists("./data/"+domainname+"/result/"):
            os.makedirs("./data/"+domainname+"/result/")
        json.dump(output_tests, open("./data/"+domainname+"/result/"+args.suffix+".json",'w'), indent=2)

# 创建新线程
threads = []
for domain in args.input_file.split(","):
    generate_thread(domain)
exit()
    #thread = myThread(1, domain, 1,domain)
    #threads.append(thread)
 
# 开启线程
for thread in threads:
    thread.start()
