
from re import M
from transformers import GPT2Model,GPT2PreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from typing import Any,Optional, Tuple,List
import numpy as np
import math
from numpy.linalg import matrix_rank
import copy


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.
    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape
            :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class Adapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(Adapter, self).__init__()
        nx = config.n_embd
        self.ln = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.project_down = nn.Linear(nx, bottleneck)
        self.relu = nn.ReLU()
        self.project_up = nn.Linear(bottleneck, nx)

    def forward(self, x):
        x_ = self.ln(x)
        x_ = self.project_down(x_)
        x_ = self.relu(x_)
        x_ = self.project_up(x_)
        x  = x + x_ #residual connection
        return x



class MixAdapter(nn.Module):
    def __init__(self, config, bottleneck_size=400, adapter_num=25):
        super(MixAdapter, self).__init__()
        # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
        self.mixadapter = nn.ModuleList([Adapter(config, bottleneck_size) for _ in range(adapter_num)])

    def forward(self, x, task_id=-1,s = -1):
        return self.mixadapter[0](x) ##change!!


class GPT2Adapter(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.project = nn.Linear(config.vocab_size, 1000)
        self.init_weights()
        self.config = config
        
    def get_output_embeddings(self):
        return self.lm_head

    def add_adapters(self,args,bottleneck_size=100,adapter_num=40):
        self.args = args
        self.adapter_blocks = nn.ModuleList([MixAdapter(self.config,bottleneck_size,adapter_num) for _ in range(self.config.n_layer)])
       
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create postion_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def get_attention_mask(self,lstm_output,length):
        length = np.array(length.detach().cpu())
        MAX_LEN = lstm_output.shape[1]
        BZ = lstm_output.shape[0]
        mask = [[1]*int(length[_])+[0]*(MAX_LEN-int(length[_])) for _ in range(BZ)]
        mask = torch.Tensor(mask)
        return mask

    def self_attention_layer(self,Q_last_hidden_state,K,V,length):  
        Q_last_hidden_state = Q_last_hidden_state.unsqueeze(1)
        attention_scores = torch.matmul(Q_last_hidden_state,K.permute(0,2,1))
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(K.shape[-1])))

        attention_mask = self.get_attention_mask(K,length) ##can only attend to hidden state that really exist
        adder = (1.0 - attention_mask.long()) * -10000.0  ##-infty, [batchsize,150]
        adder = torch.unsqueeze(adder,axis = 1).cuda()
        attention_scores += adder

        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()

    def forward_mix(self,input_ids_A, labels_A, input_ids_B, labels_B,mix_layer,BNM = False,length = None):
        output_attentions = False
        use_cache = True
        batchsize = min(input_ids_A.shape[0],input_ids_B.shape[0])
        input_ids_A = input_ids_A[:batchsize,:]; input_ids_B = input_ids_B[:batchsize,:]
        labels_A = labels_A[:batchsize,:]; labels_B = labels_B[:batchsize,:]  ###unify the batchsize

        input_shape = input_ids_A.size() ##input_ids:[32,80]
        input_ids_A = input_ids_A.view(-1, input_shape[-1])
        input_ids_B = input_ids_B.view(-1, input_shape[-1])

        past_length = 0
        past_key_values = [None] * len(self.transformer.h) ##12
        device = input_ids_A.device
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        encoder_attention_mask = None

        head_mask = self.transformer.get_head_mask(None, self.transformer.config.n_layer) ##[None]*12

        inputs_embeds_A = self.transformer.wte(input_ids_A) ### get token embedding
        inputs_embeds_B = self.transformer.wte(input_ids_B)
        position_embeds = self.transformer.wpe(position_ids) ##get position embedding
        hidden_states_A = inputs_embeds_A + position_embeds 
        hidden_states_A = self.transformer.drop(hidden_states_A)
        hidden_states_B = inputs_embeds_B + position_embeds 
        hidden_states_B = self.transformer.drop(hidden_states_B)

        l = np.random.beta(self.args.alpha,self.args.alpha)
        l = max(l,1-l) 

        #hidden_states_A = (1.0-l) * hidden_states_A + l * hidden_states_B

        output_shape = input_shape + (hidden_states_A.size(-1),) ##[32, 80, 768]

        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            if i <= mix_layer:
                hidden_states_A = block(hidden_states_A,layer_past=layer_past,use_cache=use_cache,output_attentions=output_attentions)[0]
                hidden_states_B = block(hidden_states_B,layer_past=layer_past,use_cache=use_cache,output_attentions=output_attentions)[0]
                #hidden_states_A = adapter(outputs_A[0]) ##Adapter, [32,80,768]
                #hidden_states_B = adapter(outputs_B[0]) ##Adapter, [32,80,768]
            if i == mix_layer:
                hidden_states_mix = (1.0-l) * hidden_states_B + l * hidden_states_A
            if i > mix_layer:
                hidden_states_mix = block(hidden_states_mix,layer_past=layer_past,use_cache=use_cache,output_attentions=output_attentions)[0]
                #hidden_states_mix = adapter(outputs_mix[0])
        hidden_states_mix = self.transformer.ln_f(hidden_states_mix)
        hidden_states_mix = hidden_states_mix.view(*output_shape) ##[32,60,768] -> [32,768] ->l2norm 
        '''
        hidden_states_mean = torch.mean(hidden_states_mix,1)
        attention_pool = self.self_attention_layer(hidden_states_mean,hidden_states_mix,hidden_states_mix,length)
        
        #attention_pool = hidden_states_mix
        hidden_states_norm = attention_pool / torch.norm(attention_pool,p = 2)
        hidden_states_norm = hidden_states_norm.view(-1,hidden_states_norm.shape[-1])
        print(hidden_states_norm.shape)
        _, s_tgt, _ = torch.svd(hidden_states_norm)
        transfer_loss = -torch.mean(s_tgt)
        '''
        lm_logits = self.lm_head(hidden_states_mix) ##50000+
        '''
        lm_logits_project = self.project(lm_logits)
        softmax_lm = nn.Softmax(dim=2)(lm_logits_project)
        softmax_lm = softmax_lm.view(-1,softmax_lm.shape[-1])
        _, s_tgt, _ = torch.svd(softmax_lm)
        transfer_loss = -torch.mean(s_tgt)
        '''

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels_A = labels_A[..., 1:].contiguous()
        shift_labels_B = labels_B[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_A.view(-1)) * l + \
              loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_B.view(-1)) * (1.0-l)
        return loss

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id = -1,
            s = -1,
            with_adapter = True,
            last_hidden = False,
            input_ids_prev = None,
            labels_prev = None,
            mix_layer = None,
            BNM = False,length = None,is_replay = None
        ):
        if input_ids_prev != None and mix_layer != None:
          return self.forward_mix(input_ids,labels,input_ids_prev,labels_prev,mix_layer,BNM,length)
        output_attentions = False
        use_cache = True

        input_shape = input_ids.size() ##input_ids:[32,80]
        input_ids = input_ids.view(-1, input_shape[-1])

        past_length = 0
        past_key_values = [None] * len(self.transformer.h) ##12

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        encoder_attention_mask = None

        head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.n_layer) ##[None]*12

        inputs_embeds = self.transformer.wte(input_ids) ### get token embedding
        position_embeds = self.transformer.wpe(position_ids) ##get position embedding
        hidden_states = inputs_embeds + position_embeds 
        hidden_states = self.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),) ##[32, 80, 768]

        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            outputs = block( ##block: ??????GPT2Block
                    hidden_states, ##[32,80,768]
                    layer_past=layer_past, ##None
                    attention_mask=attention_mask, ##None
                    head_mask=head_mask[i], ##None
                    encoder_hidden_states=encoder_hidden_states, ##None
                    encoder_attention_mask=encoder_attention_mask, ##None
                    use_cache=use_cache, ##True
                    output_attentions=output_attentions, ##False
                )
            if BNM and i == self.args.layer:
                hidden_states_bnm = outputs[0]
                hidden_states_bnm = self.transformer.ln_f_1(hidden_states_bnm)
            hidden_states = outputs[0]
        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        if BNM:

            real_last_hidden =  [hidden_states[i,length[i].long()-1,:] for i in range(hidden_states.shape[0])] ## the actual hidden state!
            real_last_hidden = torch.stack(real_last_hidden,axis = 0)
            hidden_states_mean = torch.mean(hidden_states,1)
            
            ok = False
    
            if self.args.only: ###only perform BNM on current task
                attention_pool = []
                for i in range(hidden_states_bnm.shape[0]):
                    if is_replay[i].item() == 0:
                        attention_pool.append(hidden_states_bnm[i])
                if len(attention_pool) != 0: 
                    attention_pool = torch.stack(attention_pool,axis = 0)
                    ok = True

            else:
                ok = True
                attention_pool = hidden_states
                #attention_pool = torch.mean(hidden_states,1)
                #attention_pool = real_last_hidden
                #attention_pool = self.self_attention_layer(hidden_states_mean,hidden_states,hidden_states,length)
                
            if ok:
                hidden_states_norm = attention_pool / torch.norm(attention_pool,p = 2)
                hidden_states_norm = hidden_states_norm.view(-1,hidden_states_norm.shape[-1])
                _, s_tgt, _ = torch.svd(hidden_states_norm)
                rank = torch.linalg.matrix_rank(hidden_states_norm)
                transfer_loss = -torch.mean(s_tgt)
            else:
                transfer_loss = 0.0

        lm_logits = self.lm_head(hidden_states)

        if labels is not None:  ##training stage
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if BNM: return rank,loss+transfer_loss*float(self.args.BNM_ratio)
            return hidden_states,loss
        else: ##testing stage
            return hidden_states, lm_logits


