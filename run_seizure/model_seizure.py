
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd
from collections import OrderedDict
import sys
sys.path.append("/srv/local/data/HealthDG")
from model import Base, Dev, CondAdv, DANN, IRM, SagNet, PCL, MLDG
from model import SoftCEL, SoftCEL2, BYOL, entropy_for_CondAdv, ProxyPLoss


"""
EEG seizure detection
"""
class SeizureBase(nn.Module):
    def __init__(self, device, dataset):
        super(SeizureBase, self).__init__()
        self.model = Base(dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.dataset = dataset

    def train_SoftCEL(self, train_loader, device):
        self.model.train()
        loss_collection = []
        count = 0
        for X, y in train_loader:
            convScore, _ = self.model(X.to(device))
            y = y.to(device)
            count += convScore.shape[0]
            loss = SoftCEL(convScore, y)
            loss_collection.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train avg loss: {:.4}, count: {}'.format(sum(loss_collection) / len(loss_collection), len(loss_collection)))
            
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore, _ = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizureDev(nn.Module):
    """ This is our model """
    def __init__(self, device, dataset):
        super(SeizureDev, self).__init__()
        self.model = Dev(dataset)
        self.BYOL = BYOL(device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        
    def train_SoftCEL_base(self, train_loader, device):
        self.model.train()
        loss_collection = []
        for X, y in train_loader:
            out, _, _, _, _ = self.model(X.to(device))
            y = y.to(device)
            loss = SoftCEL(out, y)
            loss_collection.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train avg loss: {:.4}, count: {}'.format(sum(loss_collection) / len(loss_collection), len(loss_collection)))
    
    def train_SoftCEL(self, train_loader, device):
        self.model.train()
        loss_collection = [[], [], [], [], []]

        for _, (X, X2, label, label2) in enumerate(train_loader):
            X = X.to(device)
            X2 = X2.to(device)
            label = label.to(device)
            label2 = label2.to(device)
            hard_label = torch.max(label, 1)[1]
            hard_label2 = torch.max(label2, 1)[1]
            y_par = (hard_label != hard_label2).float()

            out, _, z, v, e = self.model(X)
            out2, _, z2, v2, e2 = self.model(X2)

            ##### build rec
            prototype = self.model.g_net.prototype[hard_label]
            prototype2 = self.model.g_net.prototype[hard_label2]
            rec = self.model.p_net(torch.cat([z, prototype2], dim=1))
            rec2 = self.model.p_net(torch.cat([z2, prototype], dim=1))
            ######

            # loss1: cross entropy loss
            loss1 = (SoftCEL(out, label) + SoftCEL(out2, label2)) / 2
            # loss2: same latent factor
            loss2 = self.BYOL(z, z2)
            # loss3: reconstruction loss
            loss3 = self.BYOL(rec, v2) + self.BYOL(rec2, v)
            # loss4: same embedding space
            z_mean = (torch.mean(z, dim=0) + torch.mean(z2, dim=0)) / 2.0
            v_mean = (torch.mean(v, dim=0) + torch.mean(v2, dim=0)) / 2.0
            loss4 = torch.sum(torch.pow(z_mean - v_mean, 2)) / torch.sum(torch.pow(v_mean.detach(), 2))
            # loss5: supervised contrastive loss
            sim = F.normalize(e, p=2, dim=1) @ F.normalize(e2, p=2, dim=1).T
            y_cross = (hard_label.reshape(-1, 1) == hard_label2.reshape(1, -1)).float().to(device)
            avg_pos = - torch.sum(sim * y_cross) / torch.sum(y_cross)
            avg_neg = torch.sum(sim * (1 - y_cross)) / torch.sum(1 - y_cross)
            loss5 = avg_pos + avg_neg
            loss = 1 * loss1  + 1 * loss2 + 1 * loss3 + 1 * loss4 + 0 * loss5
            loss_collection[0].append(loss1.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(loss3.item())
            loss_collection[3].append(loss4.item())
            loss_collection[4].append(loss5.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print ('train avg loss: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, count: {}'.format(
                np.sum(loss_collection) / len(train_loader), 
                sum(loss_collection[0]) / len(train_loader), 
                sum(loss_collection[1]) / len(train_loader), 
                sum(loss_collection[2]) / len(train_loader), 
                sum(loss_collection[3]) / len(train_loader), 
                sum(loss_collection[4]) / len(train_loader), 
                len(train_loader)
                )
            )   
      
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                out, _, _, _, _ = self.model(X.to(device))
                result = np.append(result, torch.max(out, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizureCondAdv(nn.Module):
    def __init__(self, device, dataset, N_pat):
        super(SeizureCondAdv, self).__init__()
        self.model = CondAdv(dataset, N_pat)
        self.dataset = dataset
        self.optimizer_encoder_predictor = torch.optim.Adam(
            [item for item in self.model.feature_cnn.parameters()] + [item for item in self.model.g_net.parameters()], lr=5e-4, weight_decay=1e-5)
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=5e-3, weight_decay=1e-5)

    def train_SoftCEL(self, train_loader, device):
        self.model.train()
        # count the number of patients
        last_loss = []
        for X, label, identities in train_loader:
            X = X.to(device)
            label = label.to(device)
            delta_d = entropy_for_CondAdv(identities.numpy())
            identities = identities.to(device)

            # update encoder and predictor
            out, d_out, rep = self.model(X)
            loss = SoftCEL(out, label) - 1e-1 * nn.CrossEntropyLoss()(d_out, identities)

            self.optimizer_encoder_predictor.zero_grad()
            loss.backward()
            self.optimizer_encoder_predictor.step()

            # update the discriminator
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
            
        print ('train avg loss: {}, loss2: {}'.format(loss.item(), L_d))
      
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore, _, _ = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizureDANN(nn.Module):
    def __init__(self, device, dataset, N_pat):
        super(SeizureDANN, self).__init__()
        self.model = DANN(dataset, N_pat)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        
    def train_SoftCEL(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        for i, (X, label, identities) in enumerate(train_loader):
            X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Forward pass
            out, d_out = self.model(X, alpha)
            loss = SoftCEL(out, label)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
            self.optimizer.zero_grad()
            (loss + 1e-1 * loss2).backward()
            self.optimizer.step()
        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))
      
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore, _ = self.model(X.to(device), 0)
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizureIRM(nn.Module):
    def __init__(self, device, dataset):
        super(SeizureIRM, self).__init__()
        self.model = IRM(dataset)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        
    @staticmethod
    def penalty(logits, y, device):
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss = SoftCEL(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.mean(grad ** 2)

    def train_SoftCEL(self, train_loader, device):
        self.model.train()
        for i, (X, label) in enumerate(train_loader):
            X = X.to(device)
            label = label.to(device)
            
            # Forward pass
            out = self.model(X)
            loss = SoftCEL(out, label)
            loss2 = 1 * self.penalty(out, label, device)        
            self.optimizer.zero_grad()
            (loss + loss2).backward()
            self.optimizer.step()
        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizureSagNet(nn.Module):
    def __init__(self, device, dataset):
        super(SeizureSagNet, self).__init__()
        self.model = SagNet(dataset)
        self.dataset = dataset
        self.optimizer_fc = torch.optim.Adam([item for item in self.model.feature_cnn.parameters()] + [item for item in self.model.g_net.parameters()], lr=5e-4, weight_decay=1e-5)
        self.optimizer_s = torch.optim.Adam([item for item in self.model.d_net.parameters()] + [item for item in self.model.d_layer3.parameters()], lr=5e-4, weight_decay=1e-5)

    def train_SoftCEL(self, train_loader, device):
        self.model.train()
        for i, (X, label) in enumerate(train_loader):
            X = X.to(device)
            label = label.to(device)
            
            # Forward pass
            out, out_random = self.model.forward_train(X)

            # update feature_cnn and g_net
            loss = SoftCEL(out, label)
            self.optimizer_fc.zero_grad()
            loss.backward(retain_graph=True)

            # update d_layer3 and d_net
            loss2 = SoftCEL(out_random, label)
            self.optimizer_s.zero_grad()
            loss2.backward(retain_graph=True)

            # further update feature_cnn and g_net
            uniform_target = torch.ones_like(out).to(device)
            loss3 = 1e-1 * SoftCEL2(out, uniform_target)
            loss3.backward(retain_graph=True)

            self.optimizer_fc.step()
            self.optimizer_s.step()
        print ('train avg loss1: {:.4}, loss2: {:.4}, loss3: {:.4}'.format(loss.item(), loss2.item(), loss3.item()))


    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizurePCL(nn.Module):
    def __init__(self, device, dataset):
        super(SeizurePCL, self).__init__()
        self.model = PCL(dataset)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.ProxyLoss = ProxyPLoss(6, 1.0) # use their recommended hyperparameters

    def train_SoftCEL(self, train_loader, device):
        self.model.train()
        for i, (X, label) in enumerate(train_loader):
            X = X.to(device)
            label = label.to(device)
            hard_label = torch.max(label, 1)[1]

            # Forward pass
            out, x_rep, w_rep = self.model.forward_train(X)
            loss = SoftCEL(out, label)
            loss2 = self.ProxyLoss(x_rep, hard_label, w_rep)

            self.optimizer.zero_grad()
            (loss + loss2).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

class SeizureMLDG(nn.Module):
    def __init__(self, device, dataset):
        super(SeizureMLDG, self).__init__()
        self.model = MLDG(dataset)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

    def train_SoftCEL(self, generator_ls, loader_ls, val_idx, device, inner_lr=1e-3, inner_loop=3):
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
                X, label = next(generator)
                X = X.to(device)
                label = label.to(device)
                out = self.model(X)
                meta_train_loss_single = SoftCEL(out, label)
                meta_train_loss += meta_train_loss_single
                meta_train_loss_single.backward(retain_graph=True)

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
                X, label = next(generator)
                X = X.to(device)
                label = label.to(device)
                out = self.model.functional_forward(X, fast_weights)
                meta_val_loss_single = SoftCEL(out, label)
                meta_val_loss_single.backward(retain_graph=True)
                meta_val_loss += meta_val_loss_single

            # update
            self.optimizer.zero_grad()
            (meta_train_loss + meta_val_loss).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(meta_train_loss.item(), meta_val_loss.item()))
        return generator_ls

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, torch.max(y, 1)[1].numpy())
        return result, gt

