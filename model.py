import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch import autograd
import math
from torch.nn import Parameter

"""
Residual block
"""
class ResBlock_sleep(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock_sleep, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)     
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class ResBlock_seizure(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock_seizure, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


"""
SoftCEL Loss
"""
def SoftCEL2(y_hat, y):
    # weighted average version
    p = F.log_softmax(y_hat, 1)
    return - torch.mean((p*y).sum(1) / y.sum(1))

def SoftCEL(y_hat, y):
    # weighted average version, but considers the large votes
    # for example, the vote distribution is [8, 5, 4, 1, 1, 1]
    # instead of using [8, 5, 4, 1, 1, 1] as weights, we consider the votes that not smaller than
    # half of the largest weight and then use [1, 1, 1, 0, 0, 0] as the weights
    p = F.log_softmax(y_hat, 1)
    y2 = torch.where(y >= torch.max(y, 1, keepdim=True)[0].repeat(1, 6) * 0.5, torch.ones_like(y), torch.zeros_like(y))
    return - torch.mean((p*y2).sum(1) / y2.sum(1))


"""
FeatureCNN
"""
class FeatureCNN_seizure(nn.Module):
    def __init__(self, fft=64):
        super(FeatureCNN_seizure, self).__init__()
        self.fft = fft
        self.conv1 = ResBlock_seizure(16, 32, 2, True, True)
        self.conv2 = ResBlock_seizure(32, 64, 2, True, True)
        self.conv3 = ResBlock_seizure(64, 128, 2, True, True)

    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = self.fft,
                hop_length = 32,
                normalized = True,
                center = True,
                onesided = True)
            signal.append(spectral)
        stacked = torch.stack(signal)
        signal1 = stacked[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = stacked[:, :, :, :, 1].permute(1, 0, 2, 3)
        signal = (signal1 ** 2 + signal2 ** 2) ** 0.5
        return torch.clip(torch.log(torch.clip(signal, min=1e-5)), min=0)

    @staticmethod
    def functional_res_block(x, conv1_weight, conv1_bias, bn1_weight, bn1_bias, \
            conv2_weight, conv2_bias, bn2_weight, bn2_bias, ds_conv_weight, ds_conv_bias,
            ds_bn_weight, ds_bn_bias):
        out = F.conv2d(x, conv1_weight, conv1_bias, stride=2, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=bn1_weight, bias=bn1_bias, training=True)
        out = F.elu(out)
        out = F.conv2d(out, conv2_weight, conv2_bias, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=bn2_weight, bias=bn2_bias, training=True)
        residual = F.conv2d(x, ds_conv_weight, ds_conv_bias, stride=2, padding=1)
        residual = F.batch_norm(residual, running_mean=None, running_var=None, weight=ds_bn_weight, bias=ds_bn_bias, training=True)
        out += residual
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = F.dropout(out, 0.5)
        return out
    
    def functional_forward(self, x, fast_weights):
        x = self.torch_stft(x)
        fast_weights_ls = list(fast_weights.values())
        out = self.functional_res_block(x, *fast_weights_ls[0:12])
        out = self.functional_res_block(out, *fast_weights_ls[12:24])
        out = self.functional_res_block(out, *fast_weights_ls[24:36])
        out = out.squeeze(-1).squeeze(-1)
        return out

    def forward(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).squeeze(-1).squeeze(-1)
        return x 

class FeatureCNN_sleep(nn.Module):
    def __init__(self, n_dim=128):
        super(FeatureCNN_sleep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock_sleep(6, 8, 2, True, False)
        self.conv3 = ResBlock_sleep(8, 16, 2, True, True)
        self.conv4 = ResBlock_sleep(16, 32, 2, True, True)
        self.n_dim = n_dim
        
    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)
        signal = (signal1 ** 2 + signal2 ** 2) ** 0.5
        return torch.clip(torch.log(torch.clip(signal, min=1e-8)), min=0)

    @staticmethod
    def functional_res_block(x, conv1_weight, conv1_bias, bn1_weight, bn1_bias, \
            conv2_weight, conv2_bias, bn2_weight, bn2_bias, ds_conv_weight, ds_conv_bias,
            ds_bn_weight, ds_bn_bias, pooling=True):
        out = F.conv2d(x, conv1_weight, conv1_bias, stride=2, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=bn1_weight, bias=bn1_bias, training=True)
        out = F.elu(out)
        out = F.conv2d(out, conv2_weight, conv2_bias, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=bn2_weight, bias=bn2_bias, training=True)
        residual = F.conv2d(x, ds_conv_weight, ds_conv_bias, stride=2, padding=1)
        residual = F.batch_norm(residual, running_mean=None, running_var=None, weight=ds_bn_weight, bias=ds_bn_bias, training=True)
        out += residual
        if pooling:
            out = F.max_pool2d(out, 2, stride=2)
        out = F.dropout(out, 0.5)
        return out
    
    def functional_forward(self, x, fast_weights):
        x = self.torch_stft(x)
        fast_weights_ls = list(fast_weights.values())
        out = F.conv2d(x, fast_weights_ls[0], fast_weights_ls[1], stride=1, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=fast_weights_ls[2], bias=fast_weights_ls[3], training=True)
        out = F.elu(out)
        out = self.functional_res_block(out, *fast_weights_ls[4:16], False)
        out = self.functional_res_block(out, *fast_weights_ls[16:28])
        out = self.functional_res_block(out, *fast_weights_ls[28:40])
        out = out.reshape(out.shape[0], -1)
        return out

    def forward(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        return x 


"""
Model for drug rec
"""

def get_last_visit(hidden_states, mask):
    last_visit = torch.sum(mask,1) - 1
    last_visit = last_visit.unsqueeze(-1)
    last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
    last_visit = torch.reshape(last_visit, hidden_states.shape)
    last_hidden_states = torch.gather(hidden_states, 1, last_visit)
    last_hidden_state = last_hidden_states[:, 0, :]
    return last_hidden_state
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def functional_forward(self, input, adj, weight, bias):
        support = input @ weight
        output = adj @ support
        if self.bias is not None:
            output = output + bias
        return output

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def functional_forward(self, w1, b1, w2, b2):
        node_embedding = self.gcn1.functional_forward(self.x, self.adj, w1, b1)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2.functional_forward(node_embedding, self.adj, w2, b2)
        return node_embedding

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(GAMENet, self).__init__()

        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.dropout = nn.Dropout(p=0.5)

        # parameters
        self.embedding = nn.ModuleList(
            [nn.Embedding(vocab_size[i] + 1, emb_dim) for i in range(2)])
        self.encoder = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim),
        )

        self.init_weights()

    def functional_forward(self, input, fast_weights):
        diagT, procT, drugT, mask = input
        #use it later
        mask_ = mask
        mask_[torch.arange(mask_.shape[0]), mask_.sum(dim=1)-1] = 0
        mask_[torch.arange(mask_.shape[0]), 0] = 1

        # encode the diag/prod/drug code
        diagT_emb = F.embedding(diagT, weight=fast_weights["feature_cnn.embedding.0.weight"])
        diagT_emb_masked = torch.sum(diagT_emb * mask.unsqueeze(-1).unsqueeze(-1), dim=2) # (batch, visit, dim)
        procT_emb = F.embedding(procT, weight=fast_weights["feature_cnn.embedding.1.weight"])
        procT_emb_masked = torch.sum(procT_emb * mask.unsqueeze(-1).unsqueeze(-1), dim=2)
        
        # use RNN encoder
        diag_emb = self.encoder[0](diagT_emb_masked)[0] # (batch, visit, dim*2)
        proc_emb = self.encoder[1](procT_emb_masked)[0] # (batch, visit, dim*2)

        patient_representations = torch.cat([diag_emb, proc_emb], dim=-1) # (batch, visit, dim*4)
        patient_representations = F.relu(patient_representations)
        queries = F.linear(patient_representations, \
            weight=fast_weights["feature_cnn.query.1.weight"], bias=fast_weights["feature_cnn.query.1.bias"]) # (batch, visit, dim)
        
        # graph memory module
        '''I:generate current input'''
        query = get_last_visit(queries, mask) # (batch, dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn.functional_forward(
                fast_weights["feature_cnn.ehr_gcn.gcn1.weight"],
                fast_weights["feature_cnn.ehr_gcn.gcn1.bias"],
                fast_weights["feature_cnn.ehr_gcn.gcn2.weight"],
                fast_weights["feature_cnn.ehr_gcn.gcn2.bias"]
            ) - self.ddi_gcn.functional_forward(
                fast_weights["feature_cnn.ddi_gcn.gcn1.weight"],
                fast_weights["feature_cnn.ddi_gcn.gcn1.bias"],
                fast_weights["feature_cnn.ddi_gcn.gcn2.weight"],
                fast_weights["feature_cnn.ddi_gcn.gcn2.bias"]
            ) * fast_weights["feature_cnn.inter"]  # (med_size, dim)
        else:
            drug_memory = self.ehr_gcn.functional_forward(
                fast_weights["feature_cnn.ehr_gcn.gcn1.weight"],
                fast_weights["feature_cnn.ehr_gcn.gcn1.bias"],
                fast_weights["feature_cnn.ehr_gcn.gcn2.weight"],
                fast_weights["feature_cnn.ehr_gcn.gcn2.bias"]
            )

        history_keys = queries # (batch, visit, dim)
        history_values = drugT # (batch, visit, med_size)
            
        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = torch.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (batch, med_size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (batch, dim)

        # remove the last visit from mask
        
        visit_weight = torch.softmax(torch.einsum("bd,bvd->bv", query, history_keys) - (1-mask_) * 1e10, dim=1) # (batch, visit)
        weighted_values = torch.einsum("bv,bvz->bz", visit_weight, history_values.float()) # (batch, med_size)
        fact2 = torch.mm(weighted_values, drug_memory) # (batch, dim)

        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (batch, dim)

        return output


    def forward(self, input):
        diagT, procT, drugT, mask = input
        # use it later
        mask_ = mask
        mask_[torch.arange(mask_.shape[0]), mask_.sum(dim=1)-1] = 0
        mask_[torch.arange(mask_.shape[0]), 0] = 1

        # encode the diag/prod/drug code
        diagT_emb = self.embedding[0](diagT) # (batch, visit, code_len, dim)
        diagT_emb_masked = torch.sum(diagT_emb * mask.unsqueeze(-1).unsqueeze(-1), dim=2) # (batch, visit, dim)
        procT_emb = self.embedding[1](procT)
        procT_emb_masked = torch.sum(procT_emb * mask.unsqueeze(-1).unsqueeze(-1), dim=2)
        
        # use RNN encoder
        diag_emb = self.encoder[0](diagT_emb_masked)[0] # (batch, visit, dim*2)
        proc_emb = self.encoder[1](procT_emb_masked)[0] # (batch, visit, dim*2)

        patient_representations = torch.cat([diag_emb, proc_emb], dim=-1) # (batch, visit, dim*4)
        queries = self.query(patient_representations) # (batch, visit, dim)


        # graph memory module
        '''I:generate current input'''
        query = get_last_visit(queries, mask) # (batch, dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (med_size, dim)
        else:
            drug_memory = self.ehr_gcn()

        history_keys = queries # (batch, visit, dim)
        history_values = drugT # (batch, visit, med_size)
            
        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = torch.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (batch, med_size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (batch, dim)

        # remove the last visit from mask
        visit_weight = torch.softmax(torch.einsum("bd,bvd->bv", query, history_keys) - (1-mask_) * 1e10, dim=1) # (batch, visit)
        weighted_values = torch.einsum("bv,bvz->bz", visit_weight, history_values.float()) # (batch, med_size)
        fact2 = torch.mm(weighted_values, drug_memory) # (batch, dim)

        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (batch, dim)

        return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embedding:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)

class Retain(nn.Module):
    def __init__(self, voc_size, emb_dim=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.output_len = voc_size[2]

        #### parameters ###
        self.embedding = nn.ModuleList(
            [nn.Embedding(voc_size[i] + 1, emb_dim) for i in range(3)])
        
        self.dropout = nn.Dropout(p=0.5)
        self.compact = nn.Linear(emb_dim * 3, emb_dim)
        self.alpha_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.alpha_li = nn.Linear(emb_dim, 1)
        self.beta_li = nn.Linear(emb_dim, emb_dim)
        #################

    def forward(self, input):
        diagT, procT, drugT, mask = input

        # encode the diag/prod/drug code
        diagT_emb = self.embedding[0](diagT) # (batch, visit, code_len, dim)
        diagT_emb_masked = torch.sum(diagT_emb, dim=2) # (batch, visit, dim)
        procT_emb = self.embedding[1](procT)
        procT_emb_masked = torch.sum(procT_emb, dim=2)
        # drugT_emb = self.embedding[2](drugT)
        # drugT_emb_masked = torch.sum(drugT_emb, dim=2)
        # combine the embeddings
        # visit_emb = self.compact(self.dropout(torch.cat([diagT_emb_masked, procT_emb_masked, drugT_emb_masked], dim=2))) # (batch, visit, dim)
        visit_emb = (diagT_emb_masked + procT_emb_masked) / 2 # (batch, visit, dim)

        g, _ = self.alpha_gru(visit_emb) # (batch, visit, dim)
        h, _ = self.beta_gru(visit_emb) # (batch, visit, dim)

        # to mask out the visit
        attn_g = torch.softmax(self.alpha_li(g), dim=1) # (batch, visit, 1)
        attn_h = torch.tanh(self.beta_li(h)) # (batch, visit, emb)

        c = attn_g * attn_h * visit_emb # (batch, visit, emb)
        c = torch.sum(c, dim=1) # (batch, emb)

        return c


""" 
Model for mortality prediction
"""

def get_last_visit(hidden_states, mask):
    last_visit = torch.sum(mask,1) - 1
    last_visit = last_visit.unsqueeze(-1)
    last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
    last_visit = torch.reshape(last_visit, hidden_states.shape)
    last_hidden_states = torch.gather(hidden_states, 1, last_visit)
    last_hidden_state = last_hidden_states[:, 0, :]
    return last_hidden_state

class L_Concat(nn.Module):
    def __init__(self, uniq_diagstring, uniq_diagICD, uniq_physicalexam, uniq_treatment, \
                            uniq_medname, uniq_labname, emb_dim=128, device='cpu'):
        super().__init__()
        self.device = device
        self.diagstring = nn.Embedding(len(uniq_diagstring), emb_dim)
        self.diagICD = nn.Embedding(len(uniq_diagICD), emb_dim)
        self.physicalexam = nn.Embedding(len(uniq_physicalexam), emb_dim)
        self.treatment = nn.Embedding(len(uniq_treatment), emb_dim)
        self.medname = nn.Embedding(len(uniq_medname), emb_dim)
        self.device = device
        self.emb_dim = emb_dim

        self.default = [torch.nn.Parameter(torch.zeros(1, 2 * emb_dim)), \
                        torch.nn.Parameter(torch.zeros(1, len(uniq_labname))), \
                        torch.nn.Parameter(torch.zeros(1, emb_dim)), \
                        torch.nn.Parameter(torch.zeros(1, emb_dim)), \
                        torch.nn.Parameter(torch.zeros(1, emb_dim))]
        for i in range(5):
            self.register_parameter('default' + str(i), self.default[i])

        self.diagNet = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.labNet = nn.Sequential(
            nn.Linear(len(uniq_labname), emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.physicalexamNet = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.treatmentNet = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.medNet = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.encoder = [nn.GRU(emb_dim, emb_dim, 3, batch_first=True) for _ in range(5)]

        for i in range(5):
            self.add_module('encoder' + str(i), self.encoder[i])

        self.patientNet = nn.Sequential(
            nn.Linear(3, 8),
            nn.ELU(),
            nn.Linear(8, 3))
        
        self.dropout = nn.Dropout(0.25)
        self.readmission = nn.Linear(5*emb_dim+3, emb_dim)
        
    def agg_emb(self, x):
        return x.sum(dim=0).unsqueeze(dim=0)
    
    def initializePE(self):
        dim_x, dim_y = self.PE.shape
        for x in range(dim_x):
            for y in range(dim_y):
                if y % 2 == 1:
                    self.PE[x, y] = np.cos(x / (10000**((y-1)/dim_y)))
                else:
                    self.PE[x, y] = np.sin(x / (10000**(y/dim_y)))
        
    def forward(self, ALL):
        patientF, x, mask = ALL
        emb_out = []
        for seq in x:
            code_list = [[self.default[i]] for i in range(5)]
            for event in seq:
                event[1:-1] = [item.to(self.device) for item in event[1:-1]]
                if event[0] == 0:
                    feature = torch.cat([self.agg_emb(self.diagstring(event[1])), \
                                             self.agg_emb(self.diagICD(event[2]))], 1)
                elif event[0] == 1:
                    feature = event[1].reshape(1, -1)
                elif event[0] == 2:
                    feature = self.agg_emb(self.physicalexam(event[1]))
                elif event[0] == 3:
                    feature = self.agg_emb(self.treatment(event[1]))
                elif event[0] == 4:
                    feature = self.agg_emb(self.medname(event[1]))
                code_list[event[0]].append(feature)
            
            for i in range(5):
                if i == 0:
                    code_list[i] = self.diagNet(torch.cat(code_list[i], 0))
                elif i == 1:
                    code_list[i] = self.labNet(torch.cat(code_list[i], 0))
                elif i == 2:
                    code_list[i] = self.physicalexamNet(torch.cat(code_list[i], 0))
                elif i == 3:
                    code_list[i] = self.treatmentNet(torch.cat(code_list[i], 0))
                elif i == 4:
                    code_list[i] = self.medNet(torch.cat(code_list[i], 0))
            tmp = []
            for i in [0, 1, 2, 3, 4]:
                hidden = self.encoder[i](code_list[i].unsqueeze(0))[0]
                tmp.append(hidden[0, -1:, :])
            emb_out.append(torch.cat(tmp, 1))
        result = torch.cat(emb_out, 0)
        final_rep = torch.cat([result, self.patientNet(patientF)], 1)
        out = self.readmission(final_rep)
        return out

    def functional_forward(self, ALL, fast_weights):
        patientF, x, mask = ALL
        emb_out = []
        for seq in x:
            code_list = [[fast_weights["feature_cnn.default{}".format(i)]] for i in range(5)]
            for event in seq:
                event[1:-1] = [item.to(self.device) for item in event[1:-1]]
                if event[0] == 0:
                    feature = torch.cat([self.agg_emb(F.embedding(event[1], weight=fast_weights["feature_cnn.diagstring.weight"])), \
                                             self.agg_emb(F.embedding(event[2], weight=fast_weights["feature_cnn.diagICD.weight"]))], 1)
                elif event[0] == 1:
                    feature = event[1].reshape(1, -1)
                elif event[0] == 2:
                    feature = self.agg_emb(F.embedding(event[1], weight=fast_weights["feature_cnn.physicalexam.weight"]))
                elif event[0] == 3:
                    feature = self.agg_emb(F.embedding(event[1], weight=fast_weights["feature_cnn.treatment.weight"]))
                elif event[0] == 4:
                    feature = self.agg_emb(F.embedding(event[1], weight=fast_weights["feature_cnn.medname.weight"]))
                code_list[event[0]].append(feature)
            
            for i in range(5):
                if i == 0:
                    code_list[i] = F.linear(torch.cat(code_list[i], 0), weight=fast_weights["feature_cnn.diagNet.0.weight"], bias=fast_weights["feature_cnn.diagNet.0.bias"])
                    code_list[i] = F.elu(code_list[i])
                    code_list[i] = F.linear(code_list[i], weight=fast_weights["feature_cnn.diagNet.2.weight"], bias=fast_weights["feature_cnn.diagNet.2.bias"])
                elif i == 1:
                    code_list[i] = F.linear(torch.cat(code_list[i], 0), weight=fast_weights["feature_cnn.labNet.0.weight"], bias=fast_weights["feature_cnn.labNet.0.bias"])
                    code_list[i] = F.elu(code_list[i])
                    code_list[i] = F.linear(code_list[i], weight=fast_weights["feature_cnn.labNet.2.weight"], bias=fast_weights["feature_cnn.labNet.2.bias"])
                elif i == 2:
                    code_list[i] = F.linear(torch.cat(code_list[i], 0), weight=fast_weights["feature_cnn.physicalexamNet.0.weight"], bias=fast_weights["feature_cnn.physicalexamNet.0.bias"])
                    code_list[i] = F.elu(code_list[i])
                    code_list[i] = F.linear(code_list[i], weight=fast_weights["feature_cnn.physicalexamNet.2.weight"], bias=fast_weights["feature_cnn.physicalexamNet.2.bias"])
                elif i == 3:
                    code_list[i] = F.linear(torch.cat(code_list[i], 0), weight=fast_weights["feature_cnn.treatmentNet.0.weight"], bias=fast_weights["feature_cnn.treatmentNet.0.bias"])
                    code_list[i] = F.elu(code_list[i])
                    code_list[i] = F.linear(code_list[i], weight=fast_weights["feature_cnn.treatmentNet.2.weight"], bias=fast_weights["feature_cnn.treatmentNet.2.bias"])
                elif i == 4:
                    code_list[i] = F.linear(torch.cat(code_list[i], 0), weight=fast_weights["feature_cnn.medNet.0.weight"], bias=fast_weights["feature_cnn.medNet.0.bias"])
                    code_list[i] = F.elu(code_list[i])
                    code_list[i] = F.linear(code_list[i], weight=fast_weights["feature_cnn.medNet.2.weight"], bias=fast_weights["feature_cnn.medNet.2.bias"])
            tmp = []
            for i in [0, 1, 2, 3, 4]:
                hidden = self.encoder[i](code_list[i].unsqueeze(0))[0]
                tmp.append(hidden[0, -1:, :])
            emb_out.append(torch.cat(tmp, 1))
        result = torch.cat(emb_out, 0)
        patient_emb = F.linear(patientF, weight=fast_weights["feature_cnn.patientNet.0.weight"], bias=fast_weights["feature_cnn.patientNet.0.bias"])
        patient_emb = F.elu(patient_emb)
        patient_emb = F.linear(patient_emb, weight=fast_weights["feature_cnn.patientNet.2.weight"], bias=fast_weights["feature_cnn.patientNet.2.bias"])
        final_rep = torch.cat([result, patient_emb], 1)
        out = F.linear(final_rep, weight=fast_weights["feature_cnn.readmission.weight"], bias=fast_weights["feature_cnn.readmission.bias"])
        return out

class Transformer(nn.Module):
    def __init__(self, uniq_diagstring, uniq_diagICD, uniq_physicalexam, uniq_treatment, \
                            uniq_medname, uniq_labname, emb_dim=128, device='cpu'):
        super().__init__()
        self.device = device
        self.device = device
        self.emb_dim = emb_dim
        num_encoder_layers = 3

        # parameters()
        self.diagstring = nn.Embedding(len(uniq_diagstring), emb_dim)
        self.diagICD = nn.Embedding(len(uniq_diagICD), emb_dim)
        self.physicalexam = nn.Embedding(len(uniq_physicalexam), emb_dim)
        self.treatment = nn.Embedding(len(uniq_treatment), emb_dim)
        self.medname = nn.Embedding(len(uniq_medname), emb_dim)

        self.default = torch.nn.Parameter(torch.zeros(1, emb_dim)).to(self.device)

        self.diagNet = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.labNet = nn.Sequential(
            nn.Linear(len(uniq_labname), emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.physicalexamNet = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.treatmentNet = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        self.medNet = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, dim_feedforward=64, nhead=8, dropout=0.25, activation='relu')
        encoder_norm = nn.LayerNorm(emb_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.patientNet = nn.Sequential(
            nn.Linear(3, 8),
            nn.ELU(),
            nn.Linear(8, 3))
        
        self.dropout = nn.Dropout(0.25)
        self.readmission = nn.Linear(emb_dim + 3, emb_dim)

        self.PE = torch.zeros((2048, emb_dim)).to(device)
        self.PETransform = nn.Linear(emb_dim, emb_dim)
        self.initializePE()
        
    def agg_emb(self, x):
        return x.sum(dim=0).unsqueeze(dim=0)
    
    def initializePE(self):
        dim_x, dim_y = self.PE.shape
        for x in range(dim_x):
            for y in range(dim_y):
                if y % 2 == 1:
                    self.PE[x, y] = np.cos(x / (10000**((y-1)/dim_y)))
                else:
                    self.PE[x, y] = np.sin(x / (10000**(y/dim_y)))
            
    def embedding(self, x, LEN):
        emb = []
        for seq in x:
            tmp = []
            for event in seq:
                if event[0] == 0:
                    feature = torch.cat([self.agg_emb(self.diagstring(event[1])), \
                                             self.agg_emb(self.diagICD(event[2]))], 1)
                    embedding = self.diagNet(feature)
                elif event[0] == 1:
                    feature = event[1].reshape(1, -1)
                    embedding = self.labNet(feature)
                elif event[0] == 2:
                    feature = self.agg_emb(self.physicalexam(event[1]))
                    embedding = self.physicalexamNet(feature)
                elif event[0] == 3:
                    feature = self.agg_emb(self.treatment(event[1]))
                    embedding = self.treatmentNet(feature)
                elif event[0] == 4:
                    feature = self.agg_emb(self.medname(event[1]))
                    embedding = self.medNet(feature)
                tmp.append(embedding)
            tmp = [self.default for _ in range(LEN - len(tmp))] + tmp
            emb.append(torch.cat(tmp, 0).unsqueeze(dim=0))
        return torch.cat(emb,0)

    def functional_embedding(self, x, LEN, diagstring, diagICD, physicalexam, treatment, medname, \
            diagNet_weight1, diagNet_bias1, diagNet_weight2, diagNet_bias2, \
            labNet_weight1, labNet_bias1, labNet_weight2, labNet_bias2, \
            physicalexamNet_weight1, physicalexamNet_bias1, physicalexamNet_weight2, physicalexamNet_bias2, \
            treatmentNet_weight1, treatmentNet_bias1, treatmentNet_weight2, treatmentNet_bias2, \
            medNet_weight1, medNet_bias1, medNet_weight2, medNet_bias2):
        emb = []
        for seq in x:
            tmp = []
            for event in seq:
                if event[0] == 0:
                    feature = torch.cat([self.agg_emb(F.embedding(event[1], weight=diagstring)), \
                                             self.agg_emb(F.embedding(event[2], weight=diagICD))], 1)
                    embedding = F.linear(feature, weight=diagNet_weight1, bias=diagNet_bias1)
                    embedding = F.elu(embedding)
                    embedding = F.linear(embedding, weight=diagNet_weight2, bias=diagNet_bias2)
                elif event[0] == 1:
                    feature = event[1].reshape(1, -1)
                    embedding = F.linear(feature, weight=labNet_weight1, bias=labNet_bias1)
                    embedding = F.elu(embedding)    
                    embedding = F.linear(embedding, weight=labNet_weight2, bias=labNet_bias2)
                elif event[0] == 2:
                    feature = self.agg_emb(F.embedding(event[1], weight=physicalexam))
                    embedding = F.linear(feature, weight=physicalexamNet_weight1, bias=physicalexamNet_bias1)
                    embedding = F.elu(embedding)
                    embedding = F.linear(embedding, weight=physicalexamNet_weight2, bias=physicalexamNet_bias2)
                elif event[0] == 3:
                    feature = self.agg_emb(F.embedding(event[1], weight=treatment))
                    embedding = F.linear(feature, weight=treatmentNet_weight1, bias=treatmentNet_bias1)
                    embedding = F.elu(embedding)
                    embedding = F.linear(embedding, weight=treatmentNet_weight2, bias=treatmentNet_bias2)
                elif event[0] == 4:
                    feature = self.agg_emb(F.embedding(event[1], weight=medname))
                    embedding = F.linear(feature, weight=medNet_weight1, bias=medNet_bias1)
                    embedding = F.elu(embedding)
                    embedding = F.linear(embedding, weight=medNet_weight2, bias=medNet_bias2)
                tmp.append(embedding)
            tmp = [self.default for _ in range(LEN - len(tmp))] + tmp
            emb.append(torch.cat(tmp, 0).unsqueeze(dim=0))
        return torch.cat(emb,0)
        
    def forward(self, ALL):
        patientF, x, mask = ALL
        emb = self.embedding(x, mask.shape[1])
        PE = self.PETransform(self.PE[:mask.shape[1], :].unsqueeze(0).repeat(emb.shape[0], 1, 1))
        emb = emb + PE
        out = self.encoder(emb.permute(1,0,2), src_key_padding_mask=mask)
        result = out[-1, :, :]
        final_rep = torch.cat([result, self.patientNet(patientF)], 1)
        out = self.readmission(final_rep)
        return out

    def functional_forward(self, ALL, fast_weights):
        patientF, x, mask = ALL
        emb = self.functional_embedding(x, mask.shape[1], fast_weights['feature_cnn.diagstring.weight'], fast_weights['feature_cnn.diagICD.weight'], \
            fast_weights['feature_cnn.physicalexam.weight'], fast_weights['feature_cnn.treatment.weight'], fast_weights['feature_cnn.medname.weight'], \
                fast_weights['feature_cnn.diagNet.0.weight'], fast_weights['feature_cnn.diagNet.0.bias'], \
                fast_weights['feature_cnn.diagNet.2.weight'], fast_weights['feature_cnn.diagNet.2.bias'], \
                fast_weights['feature_cnn.labNet.0.weight'], fast_weights['feature_cnn.labNet.0.bias'], \
                fast_weights['feature_cnn.labNet.2.weight'], fast_weights['feature_cnn.labNet.2.bias'], \
                fast_weights['feature_cnn.physicalexamNet.0.weight'], fast_weights['feature_cnn.physicalexamNet.0.bias'], \
                fast_weights['feature_cnn.physicalexamNet.2.weight'], fast_weights['feature_cnn.physicalexamNet.2.bias'], \
                fast_weights['feature_cnn.treatmentNet.0.weight'], fast_weights['feature_cnn.treatmentNet.0.bias'], \
                fast_weights['feature_cnn.treatmentNet.2.weight'], fast_weights['feature_cnn.treatmentNet.2.bias'], \
                fast_weights['feature_cnn.medNet.0.weight'], fast_weights['feature_cnn.medNet.0.bias'], \
                fast_weights['feature_cnn.medNet.2.weight'], fast_weights['feature_cnn.medNet.2.bias'])
        PE = F.linear(self.PE[:mask.shape[1], :].unsqueeze(0).repeat(emb.shape[0], 1, 1), \
            weight=fast_weights['feature_cnn.PETransform.weight'], bias=fast_weights['feature_cnn.PETransform.bias'])
        emb = emb + PE
        out = self.encoder(emb.permute(1,0,2), src_key_padding_mask=mask)
        result = out[-1, :, :]
        patient_emb = F.linear(patientF, weight=fast_weights['feature_cnn.patientNet.0.weight'], bias=fast_weights['feature_cnn.patientNet.0.bias'])
        patient_emb = F.elu(patient_emb)
        patient_emb = F.linear(patient_emb, weight=fast_weights['feature_cnn.patientNet.2.weight'], bias=fast_weights['feature_cnn.patientNet.2.bias'])
        final_rep = torch.cat([result, patient_emb], 1)
        out = F.linear(final_rep, weight=fast_weights['feature_cnn.readmission.weight'], bias=fast_weights['feature_cnn.readmission.bias'])
        return out


"""
Core Module
"""
class Base(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(Base, self).__init__()
        self.dataset = dataset
        self.model = model
        if dataset == "seizure":
            self.feature_cnn = FeatureCNN_seizure()
            self.g_net = nn.Sequential(
                nn.Linear(128, 16),
                nn.ReLU(),
                nn.Linear(16, 6)
            )
        elif dataset == "sleep":
            self.feature_cnn = FeatureCNN_sleep()
            self.g_net = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
            )
        
        elif dataset == "drugrec": 
            if model == "Retain":
                self.feature_cnn = Retain(voc_size, emb_dim=64, device=device)
            elif model == "GAMENet":
                self.feature_cnn = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=True)
            self.g_net = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, voc_size[2]),
            )
        elif dataset == "mortality":
            diagICD2idx, diagstring2idx, labname2idx, physicalexam2idx, treatment2idx, medname2idx = self.model_initialization_params()
            if model == "Transformer":
                self.feature_cnn = Transformer(diagstring2idx, diagICD2idx, physicalexam2idx, treatment2idx, \
                                medname2idx, labname2idx, emb_dim=128, device=device)
            elif model == "L_Concat":
                self.feature_cnn = L_Concat(diagstring2idx, diagICD2idx, physicalexam2idx, treatment2idx, \
                                medname2idx, labname2idx, emb_dim=128, device=device)
            self.g_net = nn.Sequential(
                nn.Linear(128, 16),
                nn.ELU(),
                nn.Linear(16, 1),
            )

    @staticmethod
    def model_initialization_params():
        idx_path = './data/idxFile'
        diagstring2idx = pickle.load(open('{}/diagstring2idx.pkl'.format(idx_path), 'rb'))
        diagICD2idx = pickle.load(open('{}/diagICD2idx.pkl'.format(idx_path), 'rb'))
        labname2idx = pickle.load(open('{}/labname2idx.pkl'.format(idx_path), 'rb'))
        physicalexam2idx = pickle.load(open('{}/physicalexam2idx.pkl'.format(idx_path), 'rb'))
        treatment2idx = pickle.load(open('{}/treatment2idx.pkl'.format(idx_path), 'rb'))
        medname2idx = pickle.load(open('{}/medname2idx.pkl'.format(idx_path), 'rb'))
        return diagICD2idx, diagstring2idx, labname2idx, physicalexam2idx, treatment2idx, medname2idx

    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out, x

class BYOL(torch.nn.modules.loss._Loss):
    """
    "boost strap your own latent", search this paper
    """
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, y_par=None, threshold=1.0):
        # L2 normalize
        emb_anchor = nn.functional.normalize(emb_anchor, p=2, dim=1)
        emb_positive = nn.functional.normalize(emb_positive, p=2, dim=1)
        # compute the cosine similarity
        if y_par is None:
            # the original BYOL version
            l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        else:
            # we select the pairs to compute the loss based on label similarity
            l_pos = torch.einsum('nc,nc,n->n', [emb_anchor, emb_positive, y_par]).unsqueeze(-1)
        loss = - torch.clip(l_pos, max=threshold).mean()
        return loss

def vec_minus(v, z):
    z = F.normalize(z, p=2, dim=1)
    return v - torch.einsum('nc,nc,nd->nd', v, z, z)

class GNet(nn.Module):
    """ prototype based predictor for multi-class classification """
    def __init__(self, N, dim):
        super(GNet, self).__init__()
        self.N = N
        self.prototype = nn.Parameter(torch.randn(N, dim))
        self.prototype.requires_grad = True
        self.T = 0.5
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        logits = x @ F.normalize(self.prototype, p=2, dim=1).T / self.T
        return logits

class GNet_binary(nn.Module):
    """ prototype based predictor for binary classification """
    def __init__(self, N, dim):
        super(GNet, self).__init__()
        self.N = N
        self.prototype = nn.Parameter(torch.randn(N, dim))
        self.prototype.requires_grad = True
        self.T = 0.5
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        logits = x @ F.normalize(self.prototype, p=2, dim=1).T / self.T
        return torch.softmax(logits, 1)[:, 0]

class Dev(Base):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(Dev, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "seizure":
            self.q_net = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.p_net = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.g_net = GNet(6, 128)
        elif dataset == "sleep":
            self.q_net = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.p_net = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.g_net = GNet(5, 128)

        elif dataset == "drugrec": 
            self.q_net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )
            self.p_net = nn.Sequential(
                nn.Linear(64 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )
            self.g_net = GNet(voc_size[2], 64)
        elif dataset == "mortality":
            self.q_net = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.p_net = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.g_net = GNet_binary(2, 128)

    def forward(self, x):
        """
        feature CNN is h(x)
        proj is q(x)
        predictor is g(x)
        mutual reconstruction p(x)
        """
        v = self.feature_cnn(x)
        z = self.q_net(v)
        e = vec_minus(v, z)
        out = self.g_net(e)
        # rec = self.p_net(z)
        return out, z, z, v, e

def entropy_for_CondAdv(labels, base=None):
    """ Computes entropy of label distribution. """
    from math import log, e
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

class CondAdv(Base):
    """ ICML 2017 Rf-radio """
    def __init__(self, dataset, N_pat, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(CondAdv, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "seizure":
            self.discriminator = nn.Sequential(
                nn.Linear(128 + 6, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )
        elif dataset == "sleep":
            self.discriminator = nn.Sequential(
                nn.Linear(128 + 5, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )
        elif dataset == "mortality":
            self.discriminator = nn.Sequential(
                nn.Linear(128 + 1, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )
        elif dataset == "drugrec":
            self.discriminator = nn.Sequential(
                nn.Linear(64 + voc_size[2], 64),
                nn.ReLU(),
                nn.Linear(64, N_pat)
            )

    def forward(self, x):
        rep = self.feature_cnn(x)
        out = self.g_net(rep)
        rep = torch.cat([rep, out.detach()], dim=1)
        d_out = self.discriminator(rep)
        return out, d_out, rep

    def forward_with_rep(self, rep):
        d_out = self.discriminator(rep)
        return d_out

class ReverseLayerF(autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN(Base):
    """ ICML 2015, JMLR 2016 """
    def __init__(self, dataset, N_pat, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DANN, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "seizure":
            self.discriminator = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )
        elif dataset == "sleep":
            self.discriminator = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )
        elif dataset == "mortality":
            self.discriminator = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )
        elif dataset == "drugrec":
            self.discriminator = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, N_pat)
            )
            
    def forward(self, x, alpha):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        d_out = self.discriminator(reverse_feature)
        return out, d_out

class IRM(Base):
    """ arXiv 2020 """
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(IRM, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
    
    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out

class SagNet(Base):
    """ CVPR 2021 """
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(SagNet, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        self.dataset = dataset
        if dataset == "seizure":
            self.d_layer3 = ResBlock_seizure(64, 128, 2, True, True)
            self.d_net = nn.Sequential(
                nn.Linear(128, 16),
                nn.ReLU(),
                nn.Linear(16, 6)
            )
        elif dataset == "sleep":
            self.d_layer3 = ResBlock_sleep(16, 32, 2, True, True)
            self.d_net = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 5)
            )

    def seizure_pre_layers(self, x):
        x_random = x[torch.randperm(x.size()[0]), :, :]
        x = self.feature_cnn.torch_stft(x)
        x = self.feature_cnn.conv1(x)
        x = self.feature_cnn.conv2(x)
        x_random = self.feature_cnn.torch_stft(x_random)
        x_random = self.feature_cnn.conv1(x_random)
        x_random = self.feature_cnn.conv2(x_random)
        return x, x_random
    
    def sleep_pre_layers(self, x):
        x_random = x[torch.randperm(x.size()[0]), :, :]
        x = self.feature_cnn.torch_stft(x)
        x = self.feature_cnn.conv1(x)
        x = self.feature_cnn.conv2(x)
        x = self.feature_cnn.conv3(x)

        x_random = self.feature_cnn.torch_stft(x_random)
        x_random = self.feature_cnn.conv1(x_random)
        x_random = self.feature_cnn.conv2(x_random)
        x_random = self.feature_cnn.conv3(x_random)
        return x, x_random

    def seizure_post_layers(self, SR_rep, CR_rep):
        SR_rep = self.feature_cnn.conv3(SR_rep)
        out = self.g_net(SR_rep.squeeze(-1).squeeze(-1))

        CR_rep = self.d_layer3(CR_rep)
        out_random = self.d_net(CR_rep.squeeze(-1).squeeze(-1))
        return out, out_random

    def sleep_post_layers(self, SR_rep, CR_rep):
        SR_rep = self.feature_cnn.conv4(SR_rep)
        out = self.g_net(SR_rep.reshape(SR_rep.size(0), -1))

        CR_rep = self.d_layer3(CR_rep)
        out_random = self.d_net(CR_rep.reshape(CR_rep.size(0), -1))
        return out, out_random

    def forward_train(self, x):
        if self.dataset == "seizure":
            x, x_random = self.seizure_pre_layers(x)
        elif self.dataset == "sleep":
            x, x_random = self.sleep_pre_layers(x)
        # get statisics
        x_mean = torch.mean(x, keepdim=True, dim=(2, 3))
        x_random_mean = torch.mean(x_random, keepdim=True, dim=(2, 3))
        x_std = torch.std(x, keepdim=True, dim=(2, 3))
        x_random_std = torch.std(x_random, keepdim=True, dim=(2, 3))
        gamma = np.random.uniform(0, 1)

        # get style-random (SR) features
        mix_mean = gamma * x_mean + (1 - gamma) * x_random_mean
        mix_std = gamma * x_std + (1 - gamma) * x_random_std
        SR_rep = (x - x_mean) / (x_std+1e-5) * mix_std + mix_mean

        # get content-random (CR) features
        CR_rep = (x_random - x_random_mean) / (x_random_std+1e-5) * x_std + x_mean

        if self.dataset == "seizure":
            return self.seizure_post_layers(SR_rep, CR_rep)
        elif self.dataset == "sleep":
            return self.sleep_post_layers(SR_rep, CR_rep)
        else:
            return

    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out

class ProxyPLoss(nn.Module):
	'''
    borrowed from here
	https://github.com/yaoxufeng/PCL-Proxy-based-Contrastive-Learning-for-Domain-Generalization
	'''
	def __init__(self, num_classes, scale):
		super(ProxyPLoss, self).__init__()
		self.soft_plus = nn.Softplus()
		self.label = torch.LongTensor([i for i in range(num_classes)])
		self.scale = scale
	
	def forward(self, feature, target, proxy):
		feature = F.normalize(feature, p=2, dim=1)
		pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)
		label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
		pred = torch.masked_select(pred.transpose(1, 0), label)  # N,
		
		pred = pred.unsqueeze(1)  # (N, 1)
		
		feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
		
		index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
		index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix
		
		feature = feature * ~label_matrix  # get negative matrix
		feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)
		
		logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
		label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
		loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
		
		return loss

class PCL(Base):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super().__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "seizure":
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            self.g_net = nn.Parameter(torch.randn(6, 128))

        elif dataset == "sleep":
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            self.g_net = nn.Parameter(torch.randn(5, 128))

    def forward_train(self, x):
        x = self.feature_cnn(x)
        out = F.normalize(x, p=2, dim=1) @ F.normalize(self.g_net, p=2, dim=1).T

        x_rep = self.head(x)
        w_rep = self.head(self.g_net)
        return out, x_rep, w_rep
    
    def forward(self, x):
        x = self.feature_cnn(x)
        out = F.normalize(x, p=2, dim=1) @ F.normalize(self.g_net, p=2, dim=1).T
        return out

class MLDG(Base):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super().__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        self.dataset = dataset
    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out
    
    def functional_forward(self, x, fast_weights):
        x = self.feature_cnn.functional_forward(x, fast_weights)
        if self.dataset == "seizure":
            out = F.linear(x, fast_weights['g_net.0.weight'], fast_weights['g_net.0.bias'])
            out = F.relu(out)
            out = F.linear(out, fast_weights['g_net.2.weight'], fast_weights['g_net.2.bias'])
        elif self.dataset == "sleep":
            out = F.linear(x, fast_weights['g_net.0.weight'], fast_weights['g_net.0.bias'])
            out = F.relu(out)
            out = F.linear(out, fast_weights['g_net.2.weight'], fast_weights['g_net.2.bias'])
        elif self.dataset == "mortality":
            out = F.linear(x, fast_weights['g_net.0.weight'], fast_weights['g_net.0.bias'])
            out = F.relu(out)
            out = F.linear(out, fast_weights['g_net.2.weight'], fast_weights['g_net.2.bias'])
        elif self.dataset == "drugrec":
            out = F.linear(x, fast_weights['g_net.0.weight'], fast_weights['g_net.0.bias'])
            out = F.relu(out)
            out = F.linear(out, fast_weights['g_net.2.weight'], fast_weights['g_net.2.bias'])
        return out
