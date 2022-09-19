from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import torch,json
import torch.nn.functional as F
import numpy as np
from model.Seq2SeqToD import Seq2SeqToD
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer,T5Config,T5ForConditionalGeneration,T5Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
    't5':(T5ForConditionalGeneration,T5Tokenizer)
}

import torch.multiprocessing as mp # 这里不需要使用pytorch的multiprocessing

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
            elif args.mode == "adapter":
                    outputs = model(**inputs)
                    next_token_logits = outputs[:,-1, :] #/ temperature
                    #next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    #generated = torch.cat((generated, next_tokens), dim=1)
                    #return generated

                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  ###[5,50527], 只是大部分都变成了-inf
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1) ##[1222]重复5次
            generated = torch.cat((generated, next_token), dim=1) ###[5,26]
    return generated
    





def generate_thread(domainname,tokenizer,model,args,task_id):
        print("spawn thread",domainname)
        torch.set_num_threads(1)
        fin = open("./data/"+domainname+"/test.txt") #'data/restaurant/test.txt'
        inputs = [i.strip() for i in fin]
        output_tests = []
        for idx in range(0, len(inputs), 1):
            print(f"PROGRESS: {int(idx/len(inputs)*100)}%")
            lines = inputs[idx]
            print(lines)
            raw_text = lines.split(' & ')[0] + ' & ' ##inform ( name = arabian nights restaurant ; food = arabian ; goodformeal = dinner ) &
            raw_text = raw_text.lower()
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
                task_id = task_id
            )
            out = out[:, len(context_tokens):].tolist() ###只取生成的后面
            examples = []
            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = text[: text.find('<|endoftext|>')] ##只取到<endoftext>之前
                examples.append(text)
            ##examples: [[' & the arabian night restaurant serves arabian food and is good for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' & arabian nights restaurant serves arabian food and the food is good for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' & arabian nights restaurant serves arabian food and is good for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' & arabian nights restaurant serves dinner and is a good arab restaurant<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' - arabian nights restaurant serves arabian food and is nice for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!']]
            output_tests.append(examples)
            print(examples)

        json.dump(output_tests, open("./data/"+domainname+"/result.json",'w'), indent=2)
        return text
        '''
            eos_token_id = tokenizer.eos_token_id
            input_ids, attention_mask, position_ids, past = get_example_inputs(model,tokenizer,raw_text,device = "cuda")
            length = input_ids.shape[0]
            input_ids = input_ids.reshape(1,length); attention_mask = attention_mask.reshape(1,length); position_ids = position_ids.reshape(1,length)
            ###[133,1]
            print("raw_text",raw_text)
            print(input_ids.shape)
            has_eos = torch.zeros(1, dtype=torch.bool).cuda()
            all_token_ids = input_ids.clone()
            for step in range(100):
                if task_id == -1:
                    outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past,labels = None)
                else:
                    outputs = model(input_ids, task_id=task_id,labels = None,attention_mask=attention_mask, position_ids=position_ids, past_key_values=past,)
                next_token_logits = outputs[0, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)

                has_eos = has_eos | (next_tokens == eos_token_id)
                tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
                all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                print(all_token_ids.shape)
                print(tokens_to_add.shape)
                # Update input_ids, attention_mask, position_ids and past
                input_ids = tokens_to_add.unsqueeze(0).clone().detach().cuda()
                print(input_ids)
                exit()
                position_ids = (position_ids[:,-1] + 1).reshape(1,1)
                attention_mask = torch.cat([attention_mask, torch.ones([1, 1]).type_as(attention_mask)], 1).cuda()

                past = list(outputs[0]) # past in torch output is tuple
                if torch.all(has_eos):
                    break

        responses = []
        responses_plain = []
        for i, output in enumerate(all_token_ids):
            responses_plain.append(tokenizer.decode(output, skip_special_tokens=True))
            res = tokenizer.decode(output, skip_special_tokens=True)
            print(res)
            responses.append(res[res.find("[SOS]"):].replace("[SOS]","").strip())
        torch.set_num_threads(1)
        return responses, responses_plain




        '''
        
        

def get_example_inputs(model,tokenizer,prompt_text,device):
    num_attention_heads = model.model.config.n_head
    hidden_size = model.model.config.n_embd
    num_layer = model.model.config.n_layer
    tokenizer.padding_side = "left"
    encodings_dict = tokenizer.batch_encode_plus(prompt_text)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids.to(device), attention_mask.to(device), position_ids.to(device), empty_past



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
    elif args.mode == "adapter":
        model = Seq2SeqToD(args)
        model.load_state_dict(torch.load(args.model_name_or_path+"/adapter.ckpt"))
    model.to(args.device)
    model.eval()
    processes = []
    model.share_memory()
    domains = args.input_file.split(",")
    mp.set_start_method('spawn', force=True)
    n_proc = len(domains) # 开4进程
    for i in range(n_proc): 
        process = mp.Process(target=generate_thread, args=(domains[i],tokenizer,model,args,i)) 
        # 每个进程都执行prediction_by_dfsd
        process.start()
        processes.append(process)
    for p in processes:
        p.join()  # 等待所有进程执行完毕