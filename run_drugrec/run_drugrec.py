import argparse
import utils_drugrec
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from model_drugrec import DrugRecBase, DrugRecDev, DrugRecCondAdv, DrugRecDANN, DrugRecIRM, DrugRecMLDG

def collate_fn(data):
    N_pat, N_visit, N_diag, N_prod, N_med = len(data), 0, 0, 0, 0
    for pat in data:
        N_visit = max(N_visit, len(pat))
        for visit in pat:
            N_diag = max(N_diag, len(visit[0]))
            N_prod = max(N_prod, len(visit[1]))
            N_med = max(N_med, len(visit[2]))
    
    diagT = torch.ones(N_pat, N_visit, N_diag, dtype=torch.long) * voc_size[0]
    prodT = torch.ones(N_pat, N_visit, N_prod, dtype=torch.long) * voc_size[1]
    medT = torch.ones(N_pat, N_visit, voc_size[2], dtype=torch.long) * voc_size[2]
    target = torch.zeros(N_pat, voc_size[2], dtype=torch.float)
    mask = torch.zeros(N_pat, N_visit, dtype=torch.long)

    for i, pat in enumerate(data):
        for j, visit in enumerate(pat):
            diagT[i, j, :len(visit[0])] = torch.LongTensor(visit[0])
            prodT[i, j, :len(visit[1])] = torch.LongTensor(visit[1])
            if j < len(pat) - 1:
                medT[i, j, visit[2]] = 1
            mask[i, j] = 1
        target[i, pat[-1][2]] = 1

    return diagT, prodT, medT, mask, target

def double_collate_fn(data):
    data1, data2 = zip(*data)
    diagT, prodT, medT, mask, target = collate_fn(data1)
    diagT2, prodT2, medT2, mask2, target2 = collate_fn(data2)
    return diagT, prodT, medT, mask, target, diagT2, prodT2, medT2, mask2, target2

def adv_collate_fn(data):
    N_pat, N_visit, N_diag, N_prod, N_med = len(data), 0, 0, 0, 0
    for pat, _ in data:
        N_visit = max(N_visit, len(pat))
        for visit in pat:
            N_diag = max(N_diag, len(visit[0]))
            N_prod = max(N_prod, len(visit[1]))
            N_med = max(N_med, len(visit[2]))
    
    diagT = torch.ones(N_pat, N_visit, N_diag, dtype=torch.long) * voc_size[0]
    prodT = torch.ones(N_pat, N_visit, N_prod, dtype=torch.long) * voc_size[1]
    medT = torch.ones(N_pat, N_visit, N_med, dtype=torch.long) * voc_size[2]
    target = torch.zeros(N_pat, voc_size[2], dtype=torch.float)
    identities = torch.zeros(N_pat, dtype=torch.long)
    mask = torch.zeros(N_pat, N_visit, dtype=torch.long)
    for i, (pat, idenity) in enumerate(data):
        for j, visit in enumerate(pat):
            diagT[i, j, :len(visit[0])] = torch.LongTensor(visit[0])
            prodT[i, j, :len(visit[1])] = torch.LongTensor(visit[1])
            if j < len(pat) - 1:
                medT[i, j, :len(visit[2])] = torch.LongTensor(visit[2])
            mask[i, j] = 1
        identities[i] = idenity
        target[i, pat[-1][2]] = 1

    return diagT, prodT, medT, mask, target, identities

def DrugDataLoader(dataset, dev=False, train=False, batch_size=256, shuffle=True):
    if train:
        if dev:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=double_collate_fn)
        elif ("adv" in args.model) or ("DANN" in args.model):
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=adv_collate_fn)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="choose from base, dev, condadv, DANN, IRM, MLDG")
    parser.add_argument('--cuda', type=int, default=0, help="which cuda")
    parser.add_argument('--N_pat', type=int, default=500, help="number of patients")
    parser.add_argument('--dataset', type=str, default="drugrec", help="dataset name")
    parser.add_argument('--epochs', type=int, default=50, help="number of epochs")
    args = parser.parse_args()

    if "MLDG" in args.model:
        args.epochs *= 5

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print ('device:', device)
    
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # load data
    train_pat_map, test_pat_map, val_pat_map, voc_size, ehr_adj, ddi_adj = utils_drugrec.load()
    print (len(train_pat_map), len(test_pat_map), len(val_pat_map))


    # data loader
    def trainloader_for_base():
        train_X = []
        for idx, patient_visit_ls in enumerate(train_pat_map):
            if idx == args.N_pat: break
            train_X += patient_visit_ls
        train_loader = DrugDataLoader(train_X, train=True, batch_size=64, shuffle=True)
        return train_loader
    
    def trainloader_for_dev():
        train_X = []
        remaining_X = []
        for idx, patient_visit_ls in enumerate(train_pat_map):
            if idx == args.N_pat: break
            if len(patient_visit_ls) == 1:
                remaining_X += patient_visit_ls
            else:
                indices = np.arange(len(patient_visit_ls) - 1)
                np.random.shuffle(indices)
                train_X += [[patient_visit_ls[i], patient_visit_ls[j]] for i, j in zip(indices[-len(indices)//2-1:], indices[:len(indices)//2+1])]
        indices = np.arange(len(remaining_X))
        np.random.shuffle(indices)
        train_X += [[remaining_X[i], remaining_X[j]] for i, j in zip(indices[-len(indices)//2-1:], indices[:len(indices)//2+1])]
        train_loader = DrugDataLoader(train_X, dev=True, train=True, batch_size=32, shuffle=True)
        return train_loader
    
    def trainloader_for_adv():
        train_X = []
        cur_count = 0
        t = 0
        for idx, patient_visit_ls in enumerate(train_pat_map):
            if idx == args.N_pat: break
            if cur_count <= 128:
                train_X += [(item, t) for item in patient_visit_ls]
                cur_count += len(patient_visit_ls)
            else:
                 t += 1
                 cur_count = 0
        train_loader = DrugDataLoader(train_X, train=True, batch_size=64, shuffle=True)
        return train_loader, t
    
    def train_loader_for_MLDG(data):
        loader_ls = []
        cur_X = []
        for idx, patient_visit_ls in enumerate(data):
            if idx == args.N_pat: break
            if len(cur_X) < 128:
                cur_X += patient_visit_ls
            else:
                loader_ls.append(DrugDataLoader(cur_X, train=True, batch_size=4, shuffle=True))
                cur_X = []
        loader_ls.append(DrugDataLoader(cur_X, train=True, batch_size=4, shuffle=True))    
        return loader_ls

    def test_val_loader(dataset):
        test_X = []
        for _, patient_visit_ls in enumerate(dataset):
            test_X += patient_visit_ls
        test_loader = DrugDataLoader(test_X, train=False, batch_size=256, shuffle=False)
        return test_loader
        

    if args.model == "baseRetain":
        train_loader = trainloader_for_base()
        model = DrugRecBase(device=device, dataset=args.dataset, voc_size=voc_size, model="Retain").to(device)
    elif args.model == "baseGAMENet":
        train_loader = trainloader_for_base()
        model = DrugRecBase(voc_size=voc_size, dataset=args.dataset, model='GAMENet', ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device).to(device)
    elif args.model == "devRetain":
        train_loader = trainloader_for_base()
        model = DrugRecDev(device=device, dataset=args.dataset, voc_size=voc_size, model="Retain").to(device)
    elif args.model == "devGAMENet":
        train_loader = trainloader_for_base()
        model = DrugRecDev(voc_size=voc_size, dataset=args.dataset, model='GAMENet', ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device).to(device)
    elif args.model == "condadvRetain":
        train_loader, t = trainloader_for_adv()
        model = DrugRecCondAdv(device=device, N_pat=t+1, dataset=args.dataset, voc_size=voc_size, model="Retain").to(device)
    elif args.model == "condadvGAMENet":
        train_loader, t = trainloader_for_adv()
        model = DrugRecCondAdv(voc_size=voc_size, N_pat=t+1, dataset=args.dataset, model='GAMENet', ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device).to(device)
    elif args.model == "DANNRetain":
        train_loader, t = trainloader_for_adv()
        model = DrugRecDANN(voc_size=voc_size, N_pat=t+1, dataset=args.dataset, model="Retain").to(device)
    elif args.model == "DANNGAMENet":
        train_loader, t = trainloader_for_adv()
        model = DrugRecDANN(voc_size=voc_size, N_pat=t+1, dataset=args.dataset, model='GAMENet', ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device).to(device)
    elif args.model == "IRMRetain":
        train_loader = trainloader_for_base()
        model = DrugRecIRM(voc_size=voc_size, dataset=args.dataset, model="Retain").to(device)
    elif args.model == "IRMGAMENet":
        train_loader = trainloader_for_base()
        model = DrugRecIRM(voc_size=voc_size, dataset=args.dataset, model='GAMENet', ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device).to(device)
    elif args.model == "MLDGRetain":
        loader_ls = train_loader_for_MLDG(train_pat_map)
        generator_ls = [iter(item) for item in loader_ls]
        model = DrugRecMLDG(voc_size=voc_size, dataset=args.dataset, model="Retain").to(device)
    elif args.model == "MLDGGAMENet":
        loader_ls = train_loader_for_MLDG(train_pat_map)
        generator_ls = [iter(item) for item in loader_ls]
        model = DrugRecMLDG(voc_size=voc_size, dataset=args.dataset, model='GAMENet', ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device).to(device)

    test_loader = test_val_loader(test_pat_map)
    val_loader = test_val_loader(val_pat_map)
    
    model_name = (args.dataset + '_' + args.model + '_' + str(args.N_pat) + '_{}').format(time.time())
    print (model_name)

    test_ja_array, test_prauc_array, test_f1_array = [], [], []
    val_ja_array, val_prauc_array, val_f1_array = [], [], []

    for i in range (args.epochs):
        tic = time.time()
        if "dev" in args.model:
            train_loader_dev = trainloader_for_dev()
            model.train(train_loader_dev, device)
        elif "DANN" in args.model:
            model.train(train_loader, device, i, args.epochs)
        elif "MLDG" in args.model:
            N = len(generator_ls)
            val_idx = np.random.choice(np.arange(N), round(N * 0.1), replace=False)
            print (N)
            model.train(generator_ls, loader_ls, val_idx, device)
        else:
            model.train(train_loader, device)
        
        ja, prauc, f1 = model.test(test_loader, device)
        print ('{}-th train jaccard: {:.4}, prauc: {:.4}, f1: {:.4}, time: {}s'.format(i, ja, prauc, f1, time.time() - tic))
        test_ja_array.append(ja)
        test_prauc_array.append(prauc)
        test_f1_array.append(f1)
        with open('log/drugrec/{}.log'.format(model_name), 'a') as outfile:
            print ('{}-th test jaccard: {:.4}, prauc: {:.4}, f1: {:.4}'.format(i, ja, prauc, f1), file=outfile)
        
        ja, prauc, f1 = model.test(val_loader, device)
        print ('{}-th val jaccard: {:.4}, prauc: {:.4}, f1: {:.4}, time: {}s'.format(i,ja, prauc, f1, time.time() - tic))
        val_ja_array.append(ja)
        val_prauc_array.append(prauc)
        val_f1_array.append(f1)
        with open('log/drugrec/{}.log'.format(model_name), 'a') as outfile:
            print ('{}-th val jaccard: {:.4}, prauc: {:.4}, f1: {:.4}'.format(i, ja, prauc, f1), file=outfile)

        # save model
        torch.save(model.state_dict(), 'pre-trained/drugrec/{}-{}.pt'.format(i, model_name))

        print ()