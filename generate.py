from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch,json
import torch.nn.functional as F
import numpy as np

import sys

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer



MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    ##length:80, num_samples:5, temperature = 1.0,top_k = 5, top_p = 0.9,repetition_penalty = 1.0
    context = torch.tensor(context, dtype=torch.long, device=device)  ##shape:([25])
    context = context.unsqueeze(0).repeat(num_samples, 1) ##shape:[5,25]
    
    generated = context
    
    with torch.no_grad():
        for _ in range(length): ##length = 80
            inputs = {'input_ids': generated} ###shape:[5,25]

            outputs = model(**inputs) ###outputs[0].shape: [5,25,50527] 
            next_token_logits = outputs[0][:, -1, :] ###[5,50527]
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  ###[5,50527], 只是大部分都变成了-inf
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1) ##[1222]重复5次
            generated = torch.cat((generated, next_token), dim=1) ###[5,26]
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")
    
    parser.add_argument('--input_file', type=str, default=None,
                        help="file")
    
    parser.add_argument('--output_file', type=str, default=None,
                        help="file")

    parser.add_argument('--nc', type=int, default=1,
                        help="number of sentence")
    
    parser.add_argument("--use_token", action='store_true',
                        help="Avoid using CUDA when available")
    
    # parser.add_argument('--use_token', type=int, default=1,
                        # help="number of sentence")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    print(args)

    fin = open(args.input_file) #'data/restaurant/test.txt'
    inputs = [i.strip() for i in fin]
    output_tests = []
    for idx in range(0, len(inputs), 1):
        print(f"PROGRESS: {int(idx/len(inputs)*100)}%")
        lines = inputs[idx]
        raw_text = lines.split(' & ')[0] + ' & ' ##inform ( name = arabian nights restaurant ; food = arabian ; goodformeal = dinner ) &
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False) ##[259, 687, 357, 1438, 796, 610, 397, 666, 12513, 7072, 2162, 2057, 796, 610, 397, 666, 2162, 922, 687, 2287, 796, 8073, 1267, 1222, 220]

        out = sample_sequence(
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
        )
        out = out[:, len(context_tokens):].tolist() ###只取生成的后面
        examples = []
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token) if args.stop_token else None] ##只取到<endoftext>之前
            examples.append(text)
        ##examples: [[' & the arabian night restaurant serves arabian food and is good for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' & arabian nights restaurant serves arabian food and the food is good for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' & arabian nights restaurant serves arabian food and is good for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' & arabian nights restaurant serves dinner and is a good arab restaurant<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ' - arabian nights restaurant serves arabian food and is nice for dinner<|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!']]
        output_tests.append(examples)

    json.dump(output_tests, open(args.output_file,'w'), indent=2)
    return text


if __name__ == '__main__':
    main()