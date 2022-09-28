import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ = 200
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

class MixAdapter(nn.Module):
    def __init__(self, config, bottleneck_size=400, adapter_num=25):
        super(MixAdapter, self).__init__()
        # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
        self.mixadapter = nn.ModuleList([MyAdapter(config, bottleneck_size) for _ in range(adapter_num)])

    def forward(self, x, task_id=-1,s = -1):
        if task_id==-1:
            return x
        else:
            return self.mixadapter[0](x) ###change!!!

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
        self.softmax = torch.nn.Softmax(dim = -1)
        self.num_iterations = 3
        #self.route_weights = nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))
        self.route_weights = nn.ParameterList([nn.Parameter(torch.randn(self.num_capsules, 1, self.in_channel, self.class_dim)) for _ in range(adapter_num)] )
        #self.W = nn.ParameterList([nn.Parameter(torch.randn(3,240,80)) for _ in range(adapter_num)])
        self.tsv = torch.tril(torch.ones(40,40)).data.cuda()# for backward
        self.config = config

    def forward(self, t,x,s):
        '''
        x = x[:,t,:]
        length = x.shape[1] // 3
        bz = x.shape[0]
        print(x.shape) ##[16,240]
        u_j_i = x[None, :, None, None, :] @ self.W[t][:, None, None, :length*3, :length] ##[1,10,40,1,80*3],[3,1,40,80*3,80]
        print(u_j_i.shape)
        u_j_i = u_j_i.view(bz,3,length)
        #u_j_i = torch.cat([self.W[t](x),self.W[t+1](x),self.W[t+2](x)],dim = 1) ####[16,3,80]
        print(u_j_i.shape) ##[16,3,80]
        b_j_i = torch.zeros(16,3).cuda() + 0.00001
        s_j = u_j_i
        for iter in range(3):
            v_j = self.squash(s_j) ##[16,3,80]
            a_i_j = (u_j_i * v_j).sum(dim=-1) ##[16,3]
            print(b_j_i.shape,a_i_j.shape)
            print(b_j_i)
            b_j_i = torch.add(b_j_i,a_i_j) ##[16,3]
            print(b_j_i)
            c_i_j = self.my_softmax(b_j_i,dim=1)
            print(c_i_j)
            c_i_j = c_i_j.unsqueeze(2)

            s_j = c_i_j * u_j_i #s_j, [16,3,80]
            print(s_j.shape)
        
        h_output = s_j.view(bz,length,-1)

        h_output= self.larger(h_output)
        #glarger=self.mask(t=t,s=s)
        #h_output=h_output*glarger.expand_as(h_output)
        #print(h_output.shape)
        return h_output
        '''

        
        
        
        batch_size = x.size(0) ##x:[10,40,80*3]; self.route_weights:[3,40,80*3,80]
        
        length = int(x.size(2) // 3)
        priors = []
        for k in range(40):
            x_ = x[:,k,:]
            priors.append(x_[None, :, None, None, :] @ self.route_weights[k][:, None, :, :length*3, :length]) ##[1,10,40,1,80*3],[3,1,40,80*3,80]
        priors = torch.cat(priors,2)

        logits = torch.zeros(*priors.size()).cuda()
        mask=torch.zeros(self.adapter_num).data.cuda()
        for x_id in range(self.adapter_num):
                if self.tsv[t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same

        for i in range(self.num_iterations):
            logits = logits*self.tsv[t].data.view(1,1,-1,1,1) #multiply 0 to future task
            logits = logits + mask.data.view(1,1,-1,1,1) #add a very small negative number
            probs = self.my_softmax(logits, dim=2)
            vote_outputs = (probs * priors).sum(dim=2, keepdim=True) #s_j
            outputs = self.squash(vote_outputs) ##v_j

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True) ##logits: b_ij. prior: u_j|i; outputs: v_j
                logits = logits + delta_logits   ##delta_logits: a_ij

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

'''
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
        self.mix_ffn = MixAdapter(config,bottleneck,adapter_num)

    def mask(self,t,s=1):
        efc1 = self.efc1(torch.LongTensor([t]).cuda())
        efc2 = self.efc2(torch.LongTensor([t]).cuda())
        gfc1=self.gate(s*efc1)
        gfc2=self.gate(s*efc2)
        return [gfc1,gfc2]

    def forward(self,x,task_id,s):
        # knowledge sharing module (KSM)
        #capsule_output = self.capsule_net(task_id,x,s) ##x:[10,80,768]
        #h = x + capsule_output #skip-connection, [10,80,768]

        # task specifc module (TSM)
        gfc1,gfc2=self.mask(t=task_id,s=s) 
        h=self.gelu(self.fc1(x)) ##[10,80,50]
        h=h*gfc1.expand_as(h)
        h=self.gelu(self.fc2(h))
        h=h*gfc2.expand_as(h)

        x = x+h
        x = self.mix_ffn(x,task_id)

        return h
'''