
from site import abs_paths
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model.GPT2Adapter import GPT2Adapter

class Seq2SeqToD(nn.Module):

    def __init__(self,args,adapter_num = 40):
        super().__init__()
        model = GPT2Adapter.from_pretrained(args.model_name_or_path)
        model.add_adapters(args,bottleneck_size=50,adapter_num=adapter_num)
        print("hahah")

        self.model = model
        self.current_task = 0
        self.first_task = True
        self.reply_memory = []
        self.task_list_seen = []

    def set_number_of_tasks(self,n_tasks):
        self.n_tasks = n_tasks

    def get_masks(self,t,s):
        masks = {}
        for layer_id in range(12): ##12 hidden layers
            fc1_key = 'model.adapter_blocks.'+str(layer_id)+'.fc1' 
            fc2_key = 'model.adapter_blocks.'+str(layer_id)+'.fc2' 
            masks[fc1_key],masks[fc2_key] = self.model.adapter_blocks[layer_id].mask(t,s)  ##gfc1, gfc2
            key = 'model.adapter_blocks.'+str(layer_id)+'.capsule_net.tsv_capsules.larger' #gfc1
            masks[key] =  self.model.adapter_blocks[layer_id].capsule_net.tsv_capsules.mask(t,s)
        return masks

    def compute_PPL(self,input_ids,label,task_id=-1,device='cuda'):
        with torch.no_grad():
            lm_logits, *_ = self.model(
                            input_ids=input_ids.to(device),
                            attention_mask=None,
                            labels=None,
                            task_id=task_id
                            )
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = label.to(device)[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = torch.reshape(loss, shift_labels.size())
        return (loss.sum(1)/(loss!=0).sum(1)).tolist()

    def forward(self, input_ids, labels = None, task_id = -1,attention_mask = None,position_ids=None,past_key_values = None,s = -1):

        loss = self.model(input_ids=input_ids,labels=labels,task_id=task_id,attention_mask = attention_mask,position_ids = position_ids,past_key_values = past_key_values,s = s)

        return loss
