import torch

def get_view_for(n,p,masks):
    for layer_id in range(12):
        if n=='model.adapter_blocks.'+str(layer_id)+'.fc1.weight':
            return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
        if n=='model.adapter_blocks.'+str(layer_id)+'.fc1.bias':
            return masks[n.replace('.bias','')].data.view(-1)
        if n=='model.adapter_blocks.'+str(layer_id)+'.fc2.weight':
            post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
            return torch.min(post,pre)
        if n=='model.adapter_blocks.'+str(layer_id)+'.fc2.bias':
            return masks[n.replace('.bias','')].data.view(-1)
        if n == 'model.adapter_blocks.'+str(layer_id)+'.capsule_net.tsv_capsules.larger.weight': #gfc1
            return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
        if n == 'model.adapter_blocks.'+str(layer_id)+'.capsule_net.tsv_capsules.larger.bias': #gfc1
            return masks[n.replace('.bias','')].data.view(-1)

    return None