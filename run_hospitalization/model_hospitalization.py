import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch import autograd
from sklearn.metrics import fbeta_score, cohen_kappa_score, average_precision_score
from collections import OrderedDict
import sys
sys.path.append("/home/chaoqiy2/Seizure/src_cq_new")
from model import Base, Dev, CondAdv, DANN, IRM, PCL, MLDG
from model import BYOL, entropy_for_CondAdv, ProxyPLoss

criterion = torch.nn.BCEWithLogitsLoss()

"""
eICU hospitalization prediction 
"""
class ReadmissionBase(nn.Module):
    def __init__(self, device, dataset, model):
        super(ReadmissionBase, self).__init__()
        self.model = Base(dataset=dataset, model=model, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device

    def train(self, data_loader):
        loss_list, y_list, prob_list, pred_list = [], [], [], []
        self.model.train()
        for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
            y = y.to(self.device)
            probs, _ = self.model([patientF, x, masks])
            loss = criterion(probs, y)

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # torch.cuda.empty_cache()

            # calculate metrics
            loss_list.append(loss.item())
            y_list += y.cpu().numpy().flatten().tolist()
            probs = probs.cpu().detach().numpy().flatten()
            prob_list += probs.tolist()
            pred = np.zeros(probs.shape)
            pred[probs>=0.5] = 1
            pred_list += pred.tolist()
        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("training loss: {:.4}, PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(sum(loss_list), prauc, f1, cohen))

    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = torch.sigmoid(self.model([patientF, x, masks])[0])
                
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc

class ReadmissionDev(nn.Module):
    def __init__(self, device, dataset, model):
        super(ReadmissionDev, self).__init__()
        self.model = Dev(dataset=dataset, model=model, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device
        self.BYOL = BYOL(device)

    def train_base(self, data_loader):
        loss_list, y_list, prob_list, pred_list = [], [], [], []
        self.model.train()
        for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
            y = y.to(self.device)
            probs, _, _, _, _ = self.model([patientF, x, masks])
            loss = criterion(probs, y)

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # torch.cuda.empty_cache()

            # calculate metrics
            loss_list.append(loss.item())
            y_list += y.cpu().numpy().flatten().tolist()
            probs = probs.cpu().detach().numpy().flatten()
            prob_list += probs.tolist()
            pred = np.zeros(probs.shape)
            pred[probs>=0.5] = 1
            pred_list += pred.tolist()
        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("training loss: {:.4}, PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(sum(loss_list), prauc, f1, cohen))


    def train(self, data_loader):
        loss_list, y_list, prob_list, pred_list = [], [], [], []
        self.model.train()
        loss_collection = [[] for i in range(5)]
        for _, (patientF, x, masks, y, patientF2, x2, masks2, y2) in enumerate(tqdm(data_loader)):
            y = y.to(self.device)
            y2 = y2.to(self.device)
            y_par = (y != y2).float().reshape(-1,)
            out, _, z, v, e = self.model([patientF, x, masks])
            out2, _, z2, v2, e2 = self.model([patientF2, x2, masks2])

            ##### build rec
            prototype = self.model.g_net.prototype[y]
            prototype2 = self.model.g_net.prototype[y2]
            rec = self.model.p_net(torch.cat([z, prototype2], dim=1))
            rec2 = self.model.p_net(torch.cat([z2, prototype], dim=1))
            ######

            # loss1: binary cross entropy
            loss1 = criterion(out, y) + criterion(out2, y2)
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
            y_cross = (y.reshape(-1, 1) != y2.reshape(1, -1)).float().to(self.device)
            avg_pos = - torch.sum(sim * y_cross) / torch.sum(y_cross)
            avg_neg = torch.sum(sim * (1 - y_cross)) / torch.sum(1 - y_cross)
            loss5 = avg_pos + avg_neg

            loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 0 * loss5
            loss_collection[0].append(loss1.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(loss3.item())
            loss_collection[3].append(loss4.item())
            loss_collection[4].append(loss5.item())
            
            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # torch.cuda.empty_cache()

            # calculate metrics
            loss_list.append(loss.item())
            y_list += y.cpu().numpy().flatten().tolist()
            probs = out.cpu().detach().numpy().flatten()
            prob_list += probs.tolist()
            pred = np.zeros(probs.shape)
            pred[probs>=0.5] = 1
            pred_list += pred.tolist()
        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("training PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}, loss: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, count: {}".format(
                prauc,
                f1, 
                cohen,
                np.sum(loss_collection) / len(data_loader),
                sum(loss_collection[0]) / len(data_loader),
                sum(loss_collection[1]) / len(data_loader),
                sum(loss_collection[2]) / len(data_loader),
                sum(loss_collection[3]) / len(data_loader),
                sum(loss_collection[4]) / len(data_loader),
                len(data_loader),
            )
        )

    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = torch.sigmoid(self.model([patientF, x, masks])[0])
                
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc

class ReadmissionCondAdv(nn.Module):
    def __init__(self, dataset, N_pat, model, device):
        super(ReadmissionCondAdv, self).__init__()
        self.model = CondAdv(dataset=dataset, N_pat=N_pat, model=model, device=device)
        self.optimizer_encoder_predictor = torch.optim.Adam(
            [item for item in self.model.feature_cnn.parameters()] + [item for item in self.model.g_net.parameters()], lr=5e-4, weight_decay=1e-5)
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device

    def train(self, data_loader):
        self.model.train()
        for _, (patientF, x, masks, y, identities) in enumerate(tqdm(data_loader)):
            y = y.to(self.device)
            delta_d = entropy_for_CondAdv(identities.numpy())
            identities = identities.to(self.device)

            # updat encoder and predictor
            out, d_out, rep = self.model([patientF, x, masks])
            loss = criterion(out, y) - 1e-1 * nn.CrossEntropyLoss()(d_out, identities)

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

        print ('train avg loss: {}, loss2: {}'.format(loss.item(), L_d))

    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = torch.sigmoid(self.model([patientF, x, masks])[0])
                
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc

class ReadmissionDANN(nn.Module):
    def __init__(self, dataset, N_pat, model, device):
        super(ReadmissionDANN, self).__init__()
        self.model = DANN(dataset=dataset, N_pat=N_pat, model=model, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device

    def train(self, data_loader, epoch, N_epoch):
        criterion1 = torch.nn.BCELoss()
        self.model.train()
        for i, (patientF, x, masks, y, identities) in enumerate(tqdm(data_loader)):
            y = y.to(self.device)
            identities = identities.to(self.device)

            p = float(i + epoch * len(data_loader)) / N_epoch / len(data_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Forward pass
            out, d_out = self.model([patientF, x, masks], alpha)
            loss = criterion(out, y)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)

            self.optimizer.zero_grad()
            (loss + 1e-1 * loss2).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))

    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = torch.sigmoid(self.model([patientF, x, masks], 0)[0])
                
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc

class ReadmissionIRM(nn.Module):
    def __init__(self, dataset, model, device):
        super(ReadmissionIRM, self).__init__()
        self.model = IRM(dataset=dataset, model=model, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device

    @staticmethod
    def penalty(logits, y, device):
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss = criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def train(self, data_loader):
        self.model.train()
        for i, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
            y = y.to(self.device)

            # Forward pass
            out = self.model([patientF, x, masks])
            loss = criterion(out, y)
            loss2 = 5e1 * self.penalty(out, y, self.device)
            self.optimizer.zero_grad()
            (loss + loss2).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))

    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = torch.sigmoid(self.model([patientF, x, masks]))
                
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc

class ReadmissionPCL(nn.Module):
    def __init__(self, dataset, model, device):
        super(ReadmissionPCL, self).__init__()
        self.model = PCL(dataset=dataset, model=model, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device
        self.ProxyLoss = ProxyPLoss(2, 1.0)

    def train(self, data_loader):
        self.model.train()
        for i, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
            y = y.to(self.device).reshape(-1,)
            # Forward pass
            out, x_rep, w_rep = self.model.forward_train([patientF, x, masks])
            
            # use softmax [0] to denote the output sigmoid probability
            loss = criterion(torch.softmax(out, dim=1)[:, 0], y)
            loss2 = self.ProxyLoss(x_rep, y, w_rep)
            self.optimizer.zero_grad()
            (loss + loss2).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(loss.item(), loss2.item()))

    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = self.model([patientF, x, masks])
                probs = torch.softmax(probs, dim=1)[:, 0]
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc

class ReadmissionMLDG(nn.Module):
    def __init__(self, dataset, model, device):
        super(ReadmissionMLDG, self).__init__()
        self.model = MLDG(dataset=dataset, model=model, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.device = device

    def train(self, generator_ls, loader_ls, val_idx, device, inner_lr=1e-3, inner_loop=3):
        self.model.train()

        for _ in range(inner_loop):
            # meta-train
            meta_train_loss = 0
            for idx, generator in enumerate(tqdm(generator_ls)):
                if idx in val_idx: continue
                # refill the buffer
                if generator._num_yielded == len(generator):
                    generator_ls[idx] = iter(loader_ls[idx])
                    generator = generator_ls[idx]
                patientF, x, masks, y = next(generator)
                y = y.to(device)
                with torch.backends.cudnn.flags(enabled=False):
                    out = self.model([patientF, x, masks])
                meta_train_loss_single = criterion(out, y)
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
            for idx, generator in enumerate(tqdm(generator_ls)):
                if idx not in val_idx: continue
                # refill the buffer
                if generator._num_yielded == len(generator):
                    generator_ls[idx] = iter(loader_ls[idx])
                    generator = generator_ls[idx]
                patientF, x, masks, y = next(generator)
                y = y.to(device)
                with torch.backends.cudnn.flags(enabled=False):
                    out = self.model.functional_forward([patientF, x, masks], fast_weights)
                meta_val_loss_single = criterion(out, y)
                meta_val_loss_single.backward(retain_graph=True)
                meta_val_loss += meta_val_loss_single

            # update
            self.optimizer.zero_grad()
            (meta_train_loss + meta_val_loss).backward()
            self.optimizer.step()

        print ('train avg loss: {}, loss2: {}'.format(meta_train_loss.item(), meta_val_loss.item()))
        return generator_ls


    def test(self, data_loader):
        y_list, prob_list, pred_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for _, (patientF, x, masks, y) in enumerate(tqdm(data_loader)):
                probs = torch.sigmoid(self.model([patientF, x, masks]))
                y_list += y.numpy().flatten().tolist()
                probs = probs.cpu().numpy().flatten()
                prob_list += probs.tolist()
                pred = np.zeros(probs.shape)
                pred[probs>=0.5] = 1
                pred_list += pred.tolist()

        f1 = fbeta_score(y_list, pred_list, beta = 1)
        cohen = cohen_kappa_score(y_list, pred_list)
        prauc = average_precision_score(y_list, prob_list)
        print ("Test result - PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}".format(prauc, f1, cohen))
        return f1, cohen, prauc