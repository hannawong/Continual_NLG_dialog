
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ = 300
class MyAdapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(MyAdapter, self).__init__()
        nx = config.n_embd
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.project_down = nn.Linear(nx, bottleneck)
        self.relu = nn.ReLU()
        self.project_up = nn.Linear(bottleneck, 3)

    def forward(self, x):
        x_ = self.ln(x)
        x_ = self.project_down(x_)
        x_ = self.relu(x_)
        x_ = self.project_up(x_)
        #x_  = x + x_ #residual connection
        return x_
class CapsuleLayerSemantic(nn.Module): #it has its own number of capsule for output
    def __init__(self, config, adapter_num):
        super().__init__()
        self.fc1 = nn.ModuleList([MyAdapter(config,50) for _ in range(adapter_num)])
        self.config = config

    def forward(self, x):
        outputs = [fc1(x).view(x.size(0), -1, 1) for fc1 in self.fc1]
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.squash(outputs)
        return outputs.transpose(2,1)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


class CapsuleLayerTSV(nn.Module): #it has its own number of capsule for output
    def __init__(self, config,adapter_num):
        super().__init__()

        self.num_routes = adapter_num ##40
        self.adapter_num = adapter_num ##40
        self.num_capsules = 3 ##config.semantic_cap_size
        self.class_dim = MAX_SEQ ###config.max_seq_length
        self.in_channel = self.num_capsules * self.class_dim ##config.max_seq_length*config.semantic_cap_size

        self.elarger=torch.nn.Embedding(adapter_num,768)
        self.larger=torch.nn.Linear(3,768)##config.semantic_cap_size,config.bert_hidden_size),each task has its own larger way
        self.gate=torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.num_iterations = 3
        self.route_weights = nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))
        self.tsv = torch.ones(adapter_num,adapter_num).data.cuda()# for backward
        self.config = config

    def forward(self, t,x,s):
        batch_size = x.size(0) ##x:[10,40,80*3]; self.route_weights:[3,40,80*3,80]
        length = int(x.size(2) // 3)
        try:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :length*3, :length] ##[1,10,40,1,80*3],[3,1,40,80*3,80]
        except:
            print(x[None, :, :, None, :].shape,self.route_weights[:, None, :, :length*3, :length].shape)

        logits = torch.zeros(*priors.size()).cuda()
        mask=torch.zeros(self.adapter_num).data.cuda()
        for x_id in range(self.adapter_num):
                if self.tsv[t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same

        for i in range(self.num_iterations):
            logits = logits*self.tsv[t].data.view(1,1,-1,1,1) #multiply 0 to future task
            logits = logits + mask.data.view(1,1,-1,1,1) #add a very small negative number
            probs = self.my_softmax(logits, dim=2)
            vote_outputs = (probs * priors).sum(dim=2, keepdim=True) #voted
            outputs = self.squash(vote_outputs)

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

        h_output = vote_outputs.view(batch_size,length,-1)

        h_output= self.larger(h_output)
        glarger=self.mask(t=t,s=s)
        h_output=h_output*glarger.expand_as(h_output)
        return h_output

    def mask(self,t,s):
        glarger=self.gate(s*self.elarger(torch.LongTensor([t]).cuda()))
        return glarger

    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

class CapsNet(nn.Module):
    def __init__(self,config,adapter_num):
        super().__init__()
        self.semantic_capsules = CapsuleLayerSemantic(config,adapter_num=adapter_num)
        self.tsv_capsules = CapsuleLayerTSV(config,adapter_num=adapter_num)

    def forward(self, t, x,s):
        semantic_output = self.semantic_capsules(x)
        tsv_output = self.tsv_capsules(t,semantic_output,s)
        return tsv_output


class BertAdapterCapsuleMask(nn.Module):
    def __init__(self,config, bottleneck, adapter_num):
        super().__init__()
        self.capsule_net = CapsNet(config,adapter_num)
        self.config = config
        self.adapter_num = adapter_num
        self.gelu = torch.nn.GELU()
        self.efc1=torch.nn.Embedding(adapter_num,bottleneck)
        self.efc2=torch.nn.Embedding(adapter_num,768)
        self.gate=torch.nn.Sigmoid()

        self.fc1=torch.nn.Linear(768,bottleneck)
        self.fc2=torch.nn.Linear(bottleneck,768)
        self.activation = torch.nn.GELU()

    def mask(self,t,s=1):
        efc1 = self.efc1(torch.LongTensor([t]).cuda())
        efc2 = self.efc2(torch.LongTensor([t]).cuda())
        gfc1=self.gate(s*efc1)
        gfc2=self.gate(s*efc2)
        return [gfc1,gfc2]

    def forward(self,x,task_id,s):
        # knowledge sharing module (KSM)
        capsule_output = self.capsule_net(task_id,x,s) ##x:[10,80,768]
        h = x + capsule_output #skip-connection, [10,80,768]

        # task specifc module (TSM)
        gfc1,gfc2=self.mask(t=task_id,s=s) 
        h=self.gelu(self.fc1(h)) ##[10,80,50]
        h=h*gfc1.expand_as(h)
        h=self.gelu(self.fc2(h))
        h=h*gfc2.expand_as(h)

        return x+h
