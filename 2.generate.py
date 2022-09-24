from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import torch,json
import torch.nn.functional as F
import numpy as np
from model.Seq2SeqToD import Seq2SeqToD
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.multiprocessing as mp # 这里不需要使用pytorch的multiprocessing
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}
tot = 0
err = 0

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
    num_samples = 5
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
                outputs = model(**inputs) ###outputs[0].shape: [5,25,50527] 
                next_token_logits = outputs[0][:, -1, :] #/ temperature ###[5,50527]
            elif args.mode == "adapter" or args.mode == "ctr":
                outputs = model(generated,labels = None,task_id = task_id,s = 400)
                next_token_logits = outputs[:,-1, :]

            #next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  ###[5,50527], 只是大部分都变成了-inf
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1) ##[1222]重复5次
            generated = torch.cat((generated, next_token), dim=1) ###[5,26]
    return generated




def get_most_probable_task_id(context_tokens,model,tokenizer = None):
    range_adpt = len(model.task_list_seen)
    context_tokens = torch.tensor(context_tokens, dtype=torch.long, device="cuda")
    context_tokens = context_tokens.unsqueeze(0).repeat(1, 1)

    '''batched_history_out = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False, return_attention_mask=False)
    batched_history_out['input_ids'].masked_fill_(batched_history_out['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["output_id_PPL"] = batched_history_out['input_ids'] '''
    perplexity_dict = []
    for t in range(range_adpt):
        ppl = model.compute_PPL(context_tokens,label = context_tokens,task_id=t) ## one value per batch
        perplexity_dict.append([ppl,t])
    perplexity_dict = sorted(perplexity_dict)
    return perplexity_dict[0][1]

def generate_thread(domainname,tokenizer,model,args,task_id):
        global tot
        global err
        print("spawn thread",domainname)
        torch.set_num_threads(1)

        fin = open("./data/"+domainname+"/test.txt") #'data/restaurant/test.txt'
        inputs = [i.strip() for i in fin]
        output_tests = []
        for idx in range(0, len(inputs), 1):
            if idx % 10 == 0:
                print(f"PROGRESS: {int(idx/len(inputs)*100)}%",domainname)
            lines = inputs[idx]
            raw_text = lines.split(' & ')[0] + ' & ' ##inform ( name = arabian nights restaurant ; food = arabian ; goodformeal = dinner ) &
            raw_text = raw_text.lower()

            context_tokens = tokenizer.encode(raw_text, add_special_tokens=False) ##[259, 687, 357, 1438, 796, 610, 397, 666, 12513, 7072, 2162, 2057, 796, 610, 397, 666, 2162, 922, 687, 2287, 796, 8073, 1267, 1222, 220]
            guess_task_id = get_most_probable_task_id(context_tokens,model)
            if guess_task_id != task_id:
                err += 1
            tot += 1
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
                task_id = guess_task_id
            )
            out = out[:, len(context_tokens):].tolist() ###只取生成的后面
            examples = []
            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = text[: text.find('<|endoftext|>')] ##只取到<endoftext>之前
                examples.append(text)
            output_tests.append(examples)
            print(raw_text,":::::",examples)
        if args.mode == "adapter":
            json.dump(output_tests, open("./data/"+domainname+"/result.json",'w'), indent=2)
        elif args.mode == "ctr":
            json.dump(output_tests, open("./data/"+domainname+"/result_ctr.json",'w'), indent=2)
        print(domainname,tot,err)
        torch.set_num_threads(1)
        return text
        



if __name__ =='__main__':
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    args.model_type = args.model_type.lower()

    print(args)
    args.no_cuda = False
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.mode == "GPT2":
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)
        model.eval()
        model.share_memory()
    elif args.mode == "adapter" or args.mode == "ctr":
        model = Seq2SeqToD(args)
        model= torch.load(args.model_name_or_path+"/adapter.ckpt")
        model.to(args.device)
        model.model.eval()
        model.model.share_memory()

    processes = []
    perm1 = {0:'sgd_travel',1:'sgd_payment',2:"TMA_restaurant",3:"TMB_music",4:"sgd_ridesharing",5:"TMA_auto",6:"sgd_music",7:"sgd_buses",8:"TMB_restaurant",9:"MWOZ_attraction",10:"TMB_sport",11:"sgd_movies",12:"sgd_homes",13:"TMA_coffee",14:"sgd_restaurants",15:"sgd_hotels",16:"sgd_weather",17:"sgd_trains",18:"MWOZ_train",19:"sgd_flights",20:"sgd_media",21:"MWOZ_taxi",22:"sgd_alarm",23:"TMA_movie",24:"sgd_banks",25:"TMA_pizza",26:"TMB_flight",27:"sgd_rentalcars",28:"TMB_movie",29:"sgd_events",30:"MWOZ_restaurant",31:"sgd_services",32:"sgd_calendar",33:"TMB_food-ordering",34:"MWOZ_hotel",35:"TMA_uber",36:"TMB_hotel"}
    perm_domain2id = {}
    for key in perm1.keys():
        perm_domain2id[perm1[key]] = key
    if args.input_file == 'all':
        domains = list(perm1.values())
    else:
        domains = args.input_file.split(",")
    mp.set_start_method('spawn', force=True)
    n_proc = len(domains) # 开4进程
    for i in range(n_proc): 
        process = mp.Process(target=generate_thread, args=(domains[i],tokenizer,model,args,perm_domain2id[domains[i]])) 
        # 每个进程都执行prediction_by_dfsd
        process.start()
        processes.append(process)
    for p in processes:
        p.join()  # 等待所有进程执行完毕
    