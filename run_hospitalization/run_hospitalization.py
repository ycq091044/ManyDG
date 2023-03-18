from cgi import test
import numpy as np
from utils_hospitalization import load
import torch
import argparse
import time
from torch.utils.data import DataLoader
from model_hospitalization import ReadmissionBase, ReadmissionDev, ReadmissionCondAdv, ReadmissionDANN, ReadmissionIRM, ReadmissionMLDG

def collate_fn(data):
    """ data is a list of sample from the dataset """
    patientF, sequences, yreadmission, ymortality, _ = zip(*data)
    patientF, x, mask, ymortality = collate_post(patientF, sequences, yreadmission, ymortality)
    return patientF, x, mask, yreadmission

def double_collate_fn(data):
    patientF, sequences, yreadmission, ymortality, _, patientF2, sequences2, yreadmission2, ymortality2, _ = zip(*data)
    patientF, x, mask, yreadmission = collate_post(patientF, sequences, yreadmission, ymortality)
    patientF2, x2, mask2, yreadmission2 = collate_post(patientF2, sequences2, yreadmission2, ymortality2)
    return patientF, x, mask, yreadmission, patientF2, x2, mask2, yreadmission2

def adv_collate_fn(data):
    patientF, sequences, yreadmission, ymortality, _, identities = zip(*data)
    patientF, x, mask, yreadmission = collate_post(patientF, sequences, yreadmission, ymortality)
    identities = torch.LongTensor(np.array(identities))
    return patientF, x, mask, yreadmission, identities

def collate_post(patientF, sequences, yreadmission, ymortality):
    max_len = max([len(i) for i in sequences])
    mask = np.ones((len(sequences), max_len)) # (batch, max_len)
    x = [] # to store the raw feature
    mortality_list = [] # to store mortality label
    readmission_list = [] # to store readmission label
    patient_list = [] # to store patient informtion

    for idx, seq in enumerate(sequences):
        tmp = []
        for event in seq:
            if event[1] == 0:
                tmp.append([0, torch.LongTensor(event[2][0]).to(device), torch.LongTensor(event[2][1]).to(device)])
            elif event[1] == 1:
                tmp.append([1, torch.FloatTensor(event[2]).to(device)])
            elif event[1] == 2:
                tmp.append([2, torch.LongTensor(event[2]).to(device)])
            elif event[1] == 3:
                tmp.append([3, torch.LongTensor(event[2]).to(device)])
            elif event[1] == 4:
                tmp.append([4, torch.LongTensor(event[2][0]).to(device)])
            
        mask[idx, -len(tmp):] = 0
        x.append(tmp)
        mortality_list.append(ymortality[idx])
        readmission_list.append(yreadmission[idx])
        patient_list.append(patientF[idx])
            
    mask = torch.BoolTensor(mask).to(device)
    yreadmission = torch.FloatTensor(np.array(readmission_list)).reshape(-1,1)
    ymortality = torch.FloatTensor(np.array(mortality_list)).reshape(-1,1)
    patientF = torch.FloatTensor(np.array(patient_list)).to(device)
    
    return patientF, x, mask, yreadmission

def ReadmissionDataLoader(dataset, batch_size=256, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def ReadmissionDoubleDataLoader(dataset, batch_size=256, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=double_collate_fn)

def ReadmissionAdvDataLoader(dataset, batch_size=256, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=adv_collate_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseTransformer', help="base, dev, condadv, DANN, IRM, MLDG x LConcat, Transformer")
    parser.add_argument('--cuda', type=int, default=1, help="which GPU")
    parser.add_argument('--emb_dim', type=int, default=128, help="emb dimension")
    parser.add_argument('--N_pat', type=int, default=15000, help="how many patient to use")
    parser.add_argument('--dataset', type=str, default="readmission", help="the dataset name")
    parser.add_argument('--epochs', type=int, default=50, help="how many epochs to train")
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

    train_data, val_data, test_data = load()
    print (len(train_data) + len(val_data) + len(test_data))

    def train_loader_for_base(data):
        train_X = []
        for idx, patient_visit_ls in enumerate(data):
            if idx == args.N_pat: break
            train_X += patient_visit_ls
        train_loader = ReadmissionDataLoader(train_X, batch_size=256, shuffle=True)
        return train_loader

    def train_loader_for_adv(data):
        train_X = []
        cur_count = 0
        t = 0
        for idx, patient_visit_ls in enumerate(data):
            if idx == args.N_pat: break
            if cur_count <= 1024:
                train_X += [[*item, t] for item in patient_visit_ls]
                cur_count += len(patient_visit_ls)
            else:
                t += 1
                cur_count = 0
        train_loader = ReadmissionAdvDataLoader(train_X, batch_size=256, shuffle=True)
        return train_loader, t + 1

    def train_loader_for_dev(data):
        train_X = []
        train_remain_X = []
        for idx, patient_visit_ls in enumerate(data):
            if idx == args.N_pat: break
            indices = np.arange(len(patient_visit_ls))
            if len(patient_visit_ls) == 1:
                train_remain_X += patient_visit_ls
            else:
                np.random.shuffle(indices)
                train_X += [[*patient_visit_ls[i], *patient_visit_ls[j]] for i, j in zip(indices[:len(indices)//2], indices[-len(indices)//2:])]
        # for remaining
        indices = np.arange(len(train_remain_X))
        np.random.shuffle(indices)
        train_X += [[*train_remain_X[i], *train_remain_X[j]] for i, j in zip(indices[:len(indices)//2], indices[-len(indices)//2:])]
        train_loader = ReadmissionDoubleDataLoader(train_X, batch_size=256, shuffle=True)
        return train_loader

    def train_loader_for_MLDG(data):
        loader_ls = []
        cur_X = []
        for idx, patient_visit_ls in enumerate(data):
            if idx == args.N_pat: break
            if len(cur_X) < 1024:
                cur_X += patient_visit_ls
            else:
                loader_ls.append(ReadmissionDataLoader(cur_X, batch_size=8, shuffle=True))
                cur_X = []
        loader_ls.append(ReadmissionDataLoader(cur_X, batch_size=8, shuffle=True))    
        return loader_ls

    def test_and_val_loader(data):
        test_X = []
        for idx, patient_visit_ls in enumerate(data):
            test_X += patient_visit_ls
        print (len(test_X))
        test_loader = ReadmissionDataLoader(test_X, batch_size=256, shuffle=False)
        return test_loader
    

    if args.model == "baseTransformer":
        train_loader = train_loader_for_base(train_data)
        model = ReadmissionBase(device=device, dataset=args.dataset, model="Transformer").to(device)
    elif args.model == "baseLConcat":
        train_loader = train_loader_for_base(train_data)
        model = ReadmissionBase(device=device, dataset=args.dataset, model="L_Concat").to(device)
    elif args.model == "devTransformer":
        train_loader = train_loader_for_base(train_data)
        model = ReadmissionDev(device=device, dataset=args.dataset, model="Transformer").to(device)
    elif args.model == "devLConcat":
        train_loader = train_loader_for_base(train_data)
        model = ReadmissionDev(device=device, dataset=args.dataset, model="L_Concat").to(device)
    elif args.model == "condadvTransformer":
        train_loader, t = train_loader_for_adv(train_data)
        model = ReadmissionCondAdv(device=device, N_pat=t, dataset=args.dataset, model="Transformer").to(device)
    elif args.model == "condadvLConcat":
        train_loader, t = train_loader_for_adv(train_data)
        model = ReadmissionCondAdv(device=device, N_pat=t, dataset=args.dataset, model="L_Concat").to(device)
    elif args.model == "DANNTransformer":
        train_loader, t = train_loader_for_adv(train_data)
        model = ReadmissionDANN(device=device, N_pat=t, dataset=args.dataset, model="Transformer").to(device)
    elif args.model == "DANNLConcat":
        train_loader, t = train_loader_for_adv(train_data)
        model = ReadmissionDANN(device=device, N_pat=t, dataset=args.dataset, model="L_Concat").to(device)
    elif args.model == "IRMTransformer":
        train_loader = train_loader_for_base(train_data)
        model = ReadmissionIRM(device=device, dataset=args.dataset, model="Transformer").to(device)
    elif args.model == "IRMLConcat":
        train_loader = train_loader_for_base(train_data)
        model = ReadmissionIRM(device=device, dataset=args.dataset, model="L_Concat").to(device)
    elif args.model == "MLDGTransformer":
        loader_ls = train_loader_for_MLDG(train_data)
        generator_ls = [iter(item) for item in loader_ls]
        model = ReadmissionMLDG(device=device, dataset=args.dataset, model="Transformer").to(device)
    elif args.model == "MLDGLConcat":
        loader_ls = train_loader_for_MLDG(train_data)
        generator_ls = [iter(item) for item in loader_ls]
        model = ReadmissionMLDG(device=device, dataset=args.dataset, model="L_Concat").to(device)

    test_loader = test_and_val_loader(test_data)
    val_loader = test_and_val_loader(val_data)

    model_name = (args.dataset + '_' + args.model + '_{}').format(time.time())
    print (model_name)

    f1_array, kappa_array, prauc_array = [], [], []
    f1_array2, kappa_array2, prauc_array2 = [], [], []
    for i in range (args.epochs):      
        tic = time.time()
        print ("-------train: {}------".format(i))

        if args.model in ["devTransformer", "devLConcat"]:
            train_loader_dev = train_loader_for_dev(train_data)
            model.train(train_loader_dev)
        elif args.model in ["DANNTransformer", "DANNLConcat"]:
            model.train(train_loader, i, 50)    
        elif args.model in ["MLDGTransformer", "MLDGLConcat"]:
            N = len(generator_ls)
            val_idx = np.random.choice(np.arange(N), round(N * 0.1), replace=False)
            model.train(generator_ls, loader_ls, val_idx, device)
        else:
            model.train(train_loader)

        f1, kappa, prauc = model.test(test_loader)
        f12, kappa2, prauc2 = model.test(val_loader)
        f1_array.append(f1); kappa_array.append(kappa); prauc_array.append(prauc)
        f1_array2.append(f12); kappa_array2.append(kappa2); prauc_array2.append(prauc2)
        with open('log/hospitalization/{}.log'.format(model_name), 'a') as outfile:
            print ('{}-th test PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}'.format(i, prauc, f1, kappa), file=outfile)
            print ('{}-th val PRAUC: {:.4}, F1: {:.4}, Kappa: {:.4}'.format(i, prauc2, f12, kappa2), file=outfile)
        
        # save model
        torch.save(model.state_dict(), 'pre-trained/hospitalization/{}-{}.pt'.format(i, model_name))
        print ()





