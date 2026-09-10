import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool as gap
from torch.nn import LayerNorm, Parameter
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict
import argparse

from util import *


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class TCPConfidenceLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TCPConfidenceLayer, self).__init__()

        self.fc1 = LinearLayer(hidden_dim,1)

    def forward(self, x):  
        x = self.fc1(x)

        return x
    
class Multi_TCP(nn.Module):
    def __init__(self, hidden_dim):
        super(Multi_TCP, self).__init__()
        self.heads_pos = TCPConfidenceLayer(hidden_dim)
        self.heads_neg = TCPConfidenceLayer(hidden_dim)
        
    def forward(self, x):
        outputs_pos = self.heads_pos(x)
        outputs_pos = torch.sigmoid(outputs_pos)           
        outputs_neg = self.heads_neg(x)
        outputs_neg = F.softplus(outputs_neg) + 1.0

        return [outputs_pos, outputs_neg]

class Fusion(nn.Module):
    def __init__(self, num_class, num_views, hidden_dim, dropout, alpha, in_dim, topk_indices):
        super().__init__()
        self.topk_indices = topk_indices
        
        self.gat1 = TWGAT_3(dropout=dropout, alpha=alpha, dim=in_dim[0])
        self.gat2 = TWGAT_3(dropout=dropout, alpha=alpha, dim=in_dim[1])
        self.gat3 = TWGAT_3(dropout=dropout, alpha=alpha, dim=in_dim[2])
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])   
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.Multi_TCPConfidenceLayer = nn.ModuleList([Multi_TCP(hidden_dim[0]) for _ in range(self.views)])
        
        fusion_input_dim = sum(hidden_dim)
        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(fusion_input_dim, hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1],num_class))
        else:
            self.MMClasifier.append(LinearLayer(fusion_input_dim, num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, omic1, omic2, omic3, adj_1, adj_2, adj_3, label=None, infer=False, epoch=None, save_dir=None, run_id=None):
        output1, gat_output1 = self.gat1(omic1, adj_1, self.topk_indices[0]) 
        output2, gat_output2 = self.gat2(omic2, adj_2, self.topk_indices[1])
        output3, gat_output3 = self.gat3(omic3, adj_3, self.topk_indices[2])

        feature = dict()  
        feature[0], feature[1], feature[2] = output1, output2, output3

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_function = nn.CrossEntropyLoss()
        
        FeatureInfo, TCPLogit, TCPConfidence, Daoshu_TCPConfidence, Pingjun_TCP= dict(), dict(), dict(), dict(), dict()
        for view in range(self.views):
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.Multi_TCPConfidenceLayer[view](feature[view])[0]
            Daoshu_TCPConfidence[view] = self.Multi_TCPConfidenceLayer[view](feature[view])[1]
            TCPConfidence[view] = torch.clamp(TCPConfidence[view], min=0.1, max=1.0)
            Daoshu_TCPConfidence[view] = torch.clamp(Daoshu_TCPConfidence[view], min=1.0, max=10.0)
            Pingjun_TCP[view] = 2/(1/TCPConfidence[view]+Daoshu_TCPConfidence[view])
            feature[view] = feature[view] * Pingjun_TCP[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))

        for view in range(self.views):
            MMLoss = MMLoss 
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            p_target = torch.clamp(p_target, min=0.1, max=1.0)
            
            Daoshu_p_target= 1 / p_target
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + F.mse_loss(Daoshu_TCPConfidence[view].view(-1), Daoshu_p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss
            
            if epoch is not None and save_dir is not None:
                save_tcp_values_csv(
                    epoch=epoch,
                    view=view,
                    p_target=p_target,
                    daoshu_p_target=Daoshu_p_target,
                    tcp_conf=TCPConfidence[view].view(-1),
                    daoshu_tcp_conf=Daoshu_TCPConfidence[view].view(-1),
                    save_dir=save_dir
                )
 
        return MMLoss, MMlogit, gat_output1, gat_output2, gat_output3, output1, output2, output3

    def infer(self, omic1, omic2, omic3, adj_1, adj_2, adj_3):
        MMlogit = self.forward(omic1, omic2, omic3, adj_1, adj_2, adj_3, infer=True)
        return MMlogit
    
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



def define_act_layer(act_type='relu', inplace=True, negative_slope=0.01):
    act_type = act_type.lower()
    
    if act_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == 'elu':
        return nn.ELU(inplace=inplace)
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ['silu', 'swish']:
        return nn.SiLU()
    elif act_type == 'mish':
        return Mish()
    elif act_type == 'tanh':
        return nn.Tanh()
    elif act_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")

class TWGAT_3(nn.Module):
    def __init__(self, dropout, alpha, dim, act_type='gelu'):
        super(TWGAT_3, self).__init__()
        self.dropout = dropout
        self.dim = dim
        self.nhids = [8, 16, 12]
        self.nheads = [4, 3, 4]
        self.fc_dim = [696, 256, 64, 32]
        self.act_fn = define_act_layer(act_type)
        self.enhance_fn = define_act_layer('LeakyReLU')  
        self.attentions1 = [GraphAttentionLayer(1, self.nhids[0], dropout=dropout, alpha=alpha, concat=True)
                            for _ in range(self.nheads[0])]
        for i, att in enumerate(self.attentions1):
            self.add_module(f'attention1_{i}', att)

        self.attentions2 = [GraphAttentionLayer(self.nhids[0] * self.nheads[0], self.nhids[1],
                                                dropout=dropout, alpha=alpha, concat=True)
                            for _ in range(self.nheads[1])]
        for i, att in enumerate(self.attentions2):
            self.add_module(f'attention2_{i}', att)

        self.attentions3 = [GraphAttentionLayer(self.nhids[1] * self.nheads[1], self.nhids[2],
                                                dropout=dropout, alpha=alpha, concat=True)
                            for _ in range(self.nheads[2])]
        for i, att in enumerate(self.attentions3):
            self.add_module(f'attention3_{i}', att)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.pool1 = nn.Linear(self.nhids[0] * self.nheads[0], 1)
        self.pool2 = nn.Linear(self.nhids[1] * self.nheads[1], 1)
        self.pool3 = nn.Linear(self.nhids[2] * self.nheads[2], 1)
        self.proj_enh_to_gat2 = nn.Linear(1, self.nhids[0] * self.nheads[0])

        lin_input_dim = 6 * self.dim
        self.fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            self.act_fn,
            nn.Dropout(p=self.dropout))
        self.fc1.apply(xavier_init)

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            self.act_fn,
            nn.Dropout(p=self.dropout))
        self.fc2.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            self.act_fn,
            nn.Dropout(p=self.dropout))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], self.fc_dim[3]),
            self.act_fn,
            nn.Dropout(p=self.dropout))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_dim[3], 2))
        self.fc5.apply(xavier_init)

    def forward(self, x, adj, selected_indices):
        batch_size = x.size(0)
        device = x.device

        if not isinstance(selected_indices, torch.Tensor):
            selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
        else:
            selected_indices = selected_indices.to(device)

        x0 = x.squeeze(-1)

        mask0 = torch.zeros_like(x0)
        mask0[:, selected_indices] = x0[:, selected_indices]
        x0_enh = self.enhance_fn(mask0 + x0)

        x1_input = self.dropout_layer(x0_enh).unsqueeze(-1)
        x1 = torch.cat([att(x1_input, adj) for att in self.attentions1], dim=-1)
        x1 = self.dropout_layer(x1)
        x1_pooled = self.pool1(x1).squeeze(-1)
        mask1 = torch.zeros_like(x1)
        mask1[:, selected_indices,:] = x1[:, selected_indices,:]
        x1_enh = self.enhance_fn(mask1 + x1)
        x2_input = x1_enh
        x1_enh= self.pool1(x1_enh).squeeze(-1)

        x2 = torch.cat([att(x2_input, adj) for att in self.attentions2], dim=-1)
        x2 = self.dropout_layer(x2)
        x2_pooled = self.pool2(x2).squeeze(-1)
        mask2 = torch.zeros_like(x2)
        mask2[:, selected_indices,:] = x2[:, selected_indices,:]
        x2_enh = self.enhance_fn(mask2*2 + x2)
        x2_enh= self.pool2(x2_enh).squeeze(-1)

        x_cat = torch.cat([x0_enh, x1_enh, x2_enh], dim=1) 
##########################################################################################
        xxx = self.dropout_layer(x)
        xxx = torch.cat([att(xxx, adj) for att in self.attentions1], dim=-1)
        xx1 = self.pool1(xxx).squeeze(-1)

        xxx = self.dropout_layer(xxx)
        xxx = torch.cat([att(xxx, adj) for att in self.attentions2], dim=-1)
        xx2 = self.pool2(xxx).squeeze(-1)

        xxx = torch.cat([x0, xx1, xx2], dim=1)
################################################################################################
        x = torch.cat([x_cat, xxx], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x1 = self.fc3(x)
        x = self.fc4(x1)
        x = self.fc5(x)

        output = x1
        gat_output = x

        return output, gat_output

    
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: (batch_zize, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_zize, number_nodes, out_features)
        """
        Wh = torch.matmul(h, self.W)  
        e = self.prepare_batch(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        B, N, E = Wh.shape  

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  

        e = Wh1 + Wh2.permute(0, 2, 1) 
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'  


# Paper-facing aliases used in the README and manuscript.
PMGAEncoder = TWGAT_3
PredictiveReliabilityLayer = Multi_TCP
MEPGNN = Fusion
