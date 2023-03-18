
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch import autograd
from utils_drugrec import multi_label_metric
import sys
sys.path.append("/srv/local/data/HealthDG")
from model import Base, Dev, CondAdv, DANN, IRM, SagNet, PCL, MLDG
from model import BYOL, entropy_for_CondAdv

"""
MIMIC-III drug recommendation
"""
class DrugRecBase(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DrugRecBase, self).__init__()
        self.voc_size = voc_size
        self.model_name = model
        self.model = Base(dataset, device=device, voc_size=voc_size, model=model, ehr_adj=ehr_adj, ddi_adj=ddi_adj) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, data_train, device):
        self.model.train()
        for diagT, prodT, medT, mask, target in data_train:
            diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
            out, _ = self.model([diagT, prodT, medT, mask])

            # loss1 binary_cross entropy
            loss = F.binary_cross_entropy_with_logits(out, target)
            self.optimizer.zero_grad()
            (loss).backward()
            self.optimizer.step()
        print ('train avg loss: {}'.format(loss.item()))

    def test(self, data_eval, device):
        self.model.eval()
        predicted = []
        gt = []
        with torch.no_grad():
            for diagT, prodT, medT, mask, target in data_eval:
                diagT, prodT, medT, mask = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device)

                out, _ = self.model([diagT, prodT, medT, mask])
                predicted.append(torch.sigmoid(out).cpu().numpy())
                gt.append(target.numpy())
                
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        ja, prauc, f1 = multi_label_metric(predicted, gt)
        return ja, prauc, f1

class DrugRecDev(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DrugRecDev, self).__init__()
        self.voc_size = voc_size
        self.model_name = model
        self.model = Dev(dataset, device=device, voc_size=voc_size, model=model, ehr_adj=ehr_adj, ddi_adj=ddi_adj) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.BYOL = BYOL(device)
    
    def train_base(self, data_train, device):
        self.model.train()
        for diagT, prodT, medT, mask, target in data_train:
            diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
            out, _, _, _, _ = self.model([diagT, prodT, medT, mask])

            # loss1 binary_cross entropy
            loss = F.binary_cross_entropy_with_logits(out, target)
            self.optimizer.zero_grad()
            (loss).backward()
            self.optimizer.step()
        print ('train avg loss: {}'.format(loss.item()))

    def train(self, data_train, device):
        self.model.train()
        loss_collection = [[] for _ in range(5)]
        for diagT, prodT, medT, mask, target, diagT2, prodT2, medT2, mask2, target2 in data_train:
            diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
            diagT2, prodT2, medT2, mask2, target2 = diagT2.to(device), prodT2.to(device), medT2.to(device), mask2.to(device), target2.to(device)

            out, rec, z, v, e = self.model([diagT, prodT, medT, mask])
            out2, rec2, z2, v2, e2 = self.model([diagT2, prodT2, medT2, mask2])

            ##### build rec
            prototype = target.unsqueeze(2).repeat(1, 1, 64) * self.model.g_net.prototype.unsqueeze(0).repeat(target.shape[0], 1, 1)
            prototype2 = target2.unsqueeze(2).repeat(1, 1, 64) * self.model.g_net.prototype.unsqueeze(0).repeat(target.shape[0], 1, 1)
            rec = self.model.p_net(torch.cat([z, prototype2.sum(1)], dim=1))
            rec2 = self.model.p_net(torch.cat([z2, prototype.sum(1)], dim=1))
            ######

            # loss1: cross entropy loss
            loss1 = F.binary_cross_entropy_with_logits(out, target) + F.binary_cross_entropy_with_logits(out2, target2)
            # loss2: same latent factor
            loss2 = self.BYOL(z, z2)
            # loss3: reconstruction loss
            loss3 = self.BYOL(rec, v2) + self.BYOL(rec2, v)
            # loss4: same embedding space
            z_mean = (torch.mean(z, dim=0) + torch.mean(z2, dim=0)) / 2.0
            v_mean = (torch.mean(v, dim=0) + torch.mean(v2, dim=0)) / 2.0
            loss4 = torch.sum(torch.pow(z_mean - v_mean, 2)) / torch.sum(torch.pow(v_mean.detach(), 2))
            # loss5: supervised contrastive loss
            loss5 = 0 * self.BYOL(z, z2)

            loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 0 * loss5
            loss_collection[0].append(loss1.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(loss3.item())
            loss_collection[3].append(loss4.item())
            loss_collection[4].append(loss5.item())
            # Backprop and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print ('train avg loss: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}'.format(
            np.sum(loss_collection) / len(data_train),
            np.mean(loss_collection[0]), 
            np.mean(loss_collection[1]), 
            np.mean(loss_collection[2]), 
            np.mean(loss_collection[3]), 
            np.mean(loss_collection[4])))

    @staticmethod
    def get_data(self, input):
        N_pat, N_visit, N_diag, N_prod = 1, len(input), 0, 0
        for visit in input:
            N_diag = max(N_diag, len(visit[0]))
            N_prod = max(N_prod, len(visit[1]))
        diagT = torch.ones(N_pat, N_visit, N_diag, dtype=torch.long) * self.voc_size[0]
        prodT = torch.ones(N_pat, N_visit, N_prod, dtype=torch.long) * self.voc_size[1]
        medT = torch.ones(N_pat, N_visit, self.voc_size[2], dtype=torch.long) * self.voc_size[2]
        target = torch.zeros(self.voc_size[2], dtype=torch.float)
        mask = torch.zeros(N_pat, N_visit, dtype=torch.long)

        for j, visit in enumerate(input):
            diagT[0, j, :len(visit[0])] = torch.LongTensor(visit[0])
            prodT[0, j, :len(visit[1])] = torch.LongTensor(visit[1])
            mask[0, j] = 1
            if j < len(input) - 1:
                medT[0, j, visit[2]] = 1
        target[input[-1][2]] = 1
    
        return diagT, prodT, medT, mask, target

    def test(self, data_eval, device):
        self.model.eval()
        predicted = []
        gt = []
        with torch.no_grad():
            for diagT, prodT, medT, mask, target in data_eval:
                diagT, prodT, medT, mask = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device)

                out, _, _, _, _ = self.model([diagT, prodT, medT, mask])
                predicted.append(torch.sigmoid(out).cpu().numpy())
                gt.append(target.numpy())
                
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        ja, prauc, f1 = multi_label_metric(predicted, gt)
        return ja, prauc, f1

class DrugRecCondAdv(nn.Module):
    def __init__(self, dataset, device=None, N_pat=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DrugRecCondAdv, self).__init__()
        self.voc_size = voc_size
        self.model_name = model
        self.model = CondAdv(dataset, N_pat=N_pat, device=device, voc_size=voc_size, model=model, ehr_adj=ehr_adj, ddi_adj=ddi_adj) 
        self.optimizer_encoder_predictor = torch.optim.Adam(
            [item for item in self.model.feature_cnn.parameters()] + [item for item in self.model.g_net.parameters()], lr=1e-3, weight_decay=1e-5)
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, data_train, device):
        self.model.train()
        for diagT, prodT, medT, mask, target, identities in data_train:
            diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
            delta_d = entropy_for_CondAdv(identities.numpy())
            identities = identities.to(device)
        
            # update encoder and predictor
            out, d_out, rep = self.model([diagT, prodT, medT, mask])
            loss = F.binary_cross_entropy_with_logits(out, target) - 1e-2 * nn.CrossEntropyLoss()(d_out, identities)

            self.optimizer_encoder_predictor.zero_grad()
            loss.backward()
            self.optimizer_encoder_predictor.step()

            # update discriminator
            L_d = nn.CrossEntropyLoss()(d_out, identities).item()
            t = 0
            while (L_d > delta_d) and (t < 5):
                d_out = self.model.forward_with_rep(rep.detach())
                loss2 = nn.CrossEntropyLoss()(d_out, identities)
                self.optimizer_discriminator.zero_grad()
                loss2.backward()
                self.optimizer_discriminator.step()
                L_d = loss2.item()
                t += 1

        print ('train avg loss: {:.4}, loss2: {:.4}'.format(loss.item(), L_d))
    

    def test(self, data_eval, device):
        self.model.eval()
        predicted = []
        gt = []
        with torch.no_grad():
            for diagT, prodT, medT, mask, target in data_eval:
                diagT, prodT, medT, mask = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device)

                out, _, _ = self.model([diagT, prodT, medT, mask])
                predicted.append(torch.sigmoid(out).cpu().numpy())
                gt.append(target.numpy())
                
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        ja, prauc, f1 = multi_label_metric(predicted, gt)
        return ja, prauc, f1

class DrugRecDANN(nn.Module):
    def __init__(self, dataset, device=None, N_pat=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DrugRecDANN, self).__init__()
        self.voc_size = voc_size
        self.model_name = model
        self.model = DANN(dataset, N_pat=N_pat, device=device, voc_size=voc_size, model=model, ehr_adj=ehr_adj, ddi_adj=ddi_adj) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, data_train, device, epoch, N_epoch):
        self.model.train()
        for i, (diagT, prodT, medT, mask, target, identities) in enumerate(data_train):
            diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
            identities = identities.to(device)

            p = float(i + epoch * len(data_train)) / N_epoch / len(data_train)
            alpha = 2. / (1. + np.exp(-10. * p)) - 1.
            # forward pass
            out, d_out = self.model([diagT, prodT, medT, mask], alpha)
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
            self.optimizer.zero_grad()
            (loss + 1e-1 * loss2).backward()
            self.optimizer.step()
        print ('train avg los: {}, loss2: {}'.format(loss.item(), loss2.item()))

    def test(self, data_eval, device):
        self.model.eval()
        predicted = []
        gt = []
        with torch.no_grad():
            for diagT, prodT, medT, mask, target in data_eval:
                diagT, prodT, medT, mask = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device)

                out, _ = self.model([diagT, prodT, medT, mask], 0)
                predicted.append(torch.sigmoid(out).cpu().numpy())
                gt.append(target.numpy())
                
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        ja, prauc, f1 = multi_label_metric(predicted, gt)
        return ja, prauc, f1

class DrugRecIRM(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DrugRecIRM, self).__init__()
        self.voc_size = voc_size
        self.model_name = model
        self.model = IRM(dataset, device=device, voc_size=voc_size, model=model, ehr_adj=ehr_adj, ddi_adj=ddi_adj) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    @staticmethod
    def penalty(logits, y, device):
        scale = torch.tensor([1.]).to(device).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def train(self, data_train, device):
        self.model.train()
        for i, (diagT, prodT, medT, mask, target) in enumerate(data_train):
            diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
            
            # forwars pass
            out = self.model([diagT, prodT, medT, mask])
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss2 = 5 * self.penalty(out, target, device)
            self.optimizer.zero_grad()
            (loss + loss2).backward()
            self.optimizer.step()
        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))
    
    def test(self, data_eval, device):
        self.model.eval()
        predicted = []
        gt = []
        with torch.no_grad():
            for diagT, prodT, medT, mask, target in data_eval:
                diagT, prodT, medT, mask = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device)

                out  = self.model([diagT, prodT, medT, mask])
                predicted.append(torch.sigmoid(out).cpu().numpy())
                gt.append(target.numpy())
                
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        ja, prauc, f1 = multi_label_metric(predicted, gt)
        return ja, prauc, f1

class DrugRecMLDG(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DrugRecMLDG, self).__init__()
        self.model = MLDG(dataset, device=device, voc_size=voc_size, model=model, ehr_adj=ehr_adj, ddi_adj=ddi_adj) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, generator_ls, loader_ls, val_idx, device, inner_lr=1e-3, inner_loop=3):
        self.model.train()
        
        for _ in range(inner_loop):
            # meta-train
            meta_train_loss = 0
            for idx, generator in enumerate(generator_ls):
                if idx in val_idx: continue
                # refill the buffer
                if generator._num_yielded == len(generator):
                    generator_ls[idx] = iter(loader_ls[idx])
                    generator = generator_ls[idx]
                diagT, prodT, medT, mask, target = next(generator)
                diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
                with torch.backends.cudnn.flags(enabled=False):
                    out = self.model([diagT, prodT, medT, mask])
                meta_train_loss_single = F.binary_cross_entropy_with_logits(out, target)
                meta_train_loss_single.backward(retain_graph=True)
                meta_train_loss += meta_train_loss_single

            fast_weights = OrderedDict(self.model.named_parameters())
            gradients = autograd.grad(meta_train_loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
            
            # meta_val
            meta_val_loss = 0
            for idx, generator in enumerate(generator_ls):
                if idx not in val_idx: continue
                # refill the buffer
                if generator._num_yielded == len(generator):
                    generator_ls[idx] = iter(loader_ls[idx])
                    generator = generator_ls[idx]
                diagT, prodT, medT, mask, target = next(generator)
                diagT, prodT, medT, mask, target = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device), target.to(device)
                with torch.backends.cudnn.flags(enabled=False):
                    out = self.model.functional_forward([diagT, prodT, medT, mask], fast_weights)
                meta_val_loss_single = F.binary_cross_entropy_with_logits(out, target)
                meta_val_loss_single.backward(retain_graph=True)
                meta_val_loss += meta_val_loss_single

            # update
            self.optimizer.zero_grad()
            (meta_train_loss + meta_val_loss).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(meta_train_loss.item(), meta_val_loss.item()))
        return generator_ls

    def test(self, data_eval, device):
        self.model.eval()
        predicted = []
        gt = []
        with torch.no_grad():
            for diagT, prodT, medT, mask, target in data_eval:
                diagT, prodT, medT, mask = diagT.to(device), prodT.to(device), medT.to(device), mask.to(device)

                out = self.model([diagT, prodT, medT, mask])
                predicted.append(torch.sigmoid(out).cpu().numpy())
                gt.append(target.numpy())
                
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        ja, prauc, f1 = multi_label_metric(predicted, gt)
        return ja, prauc, f1

