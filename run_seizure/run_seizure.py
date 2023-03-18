import os
import argparse
import utils_seizure
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import time
from model_seizure import SeizureBase, SeizureDev, SeizureCondAdv, SeizureDANN, SeizureIRM, SeizureSagNet, SeizurePCL, SeizureMLDG

def accuracy_score(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)

def confusion_matrix(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    n_classes = int(max(np.max(actual), np.max(pred)) + 1)
    confusion = np.zeros([n_classes, n_classes])
    for i in range(n_classes):
        for j in range(n_classes):
            confusion[i, j] = np.sum((actual == i) & (pred == j))
    return confusion.astype('int')

def weighted_f1(gt, pre):
    confusion = confusion_matrix(gt, pre)
    f1_ls = []
    for i in range(confusion.shape[0]):
        if confusion[i, i] == 0:
            f1_ls.append(0)
        else:
            precision_tmp = confusion[i, i] / confusion[i].sum()
            recall_tmp = confusion[i, i] / confusion[:, i].sum()
            f1_ls.append(2 * precision_tmp * recall_tmp / (precision_tmp + recall_tmp))
    return np.mean(f1_ls)

def cohen_kappa_score(y1, y2):
    confusion = confusion_matrix(y1, y2)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="base", required=True, help="choose from base, dev, condadv, DANN, IRM, SagNet, PCL, MLDG")
    parser.add_argument('--cuda', type=int, default=0, help="which cuda")
    parser.add_argument('--N_vote', type=int, default=5, help="vote threshold")
    parser.add_argument('--N_pat', type=int, default=2000, help="number of patients")
    parser.add_argument('--dataset', type=str, default="seizure", help="dataset name")
    parser.add_argument('--MLDG_threshold', type=int, default=1024, help="threshold for MLDG")
    parser.add_argument('--epochs', type=int, default=1, help="N of epochs")
    args = parser.parse_args()

    if args.model == "MLDG":
        args.epochs *= 5
        if args.N_pat < 1000:
            args.MLDG_threshold = 256


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
    path = "data/Seizure/test_pat_map_seizure.pkl"
    if os.path.exists(path):
        train_pat_map, test_pat_map, val_pat_map = utils_seizure.load()
        print (len(train_pat_map), len(test_pat_map), len(val_pat_map))
    else:   
        train_pat_map, test_pat_map, val_pat_map = utils_seizure.load_and_dump()

    def trainloader_for_other():
        train_X, train_Y = [], []
        for i, (_, (X, Y)) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            valid_idx = np.where((np.sum(np.array(Y), axis=1) >= args.N_vote))[0]
            X = [X[item] for item in valid_idx]
            Y = [Y[item] for item in valid_idx]
            train_X += X
            train_Y += Y
        train_loader = torch.utils.data.DataLoader(utils_seizure.IIICLoader(train_X, train_Y),
                    batch_size=128, shuffle=True)
        return train_loader
    
    def trainloader_for_adv():
        train_X, train_Y, train_ID = [], [], []

        for i, (_, (X, Y)) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            valid_idx = np.where((np.sum(np.array(Y), axis=1) >= args.N_vote))[0]
            X = [X[item] for item in valid_idx]
            Y = [Y[item] for item in valid_idx]
            train_X += X
            train_Y += Y
            train_ID += [i for _ in X]
        train_loader = torch.utils.data.DataLoader(utils_seizure.IIICIDLoader(train_X, train_Y, train_ID),
                batch_size=128, shuffle=True)
        return train_loader

    def trainloader_for_dev():
        train_X, train_Y = [], []
        train_X_aux, train_Y_aux = [], []

        for i, (_, (X, Y)) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            valid_idx = np.where((np.sum(np.array(Y), axis=1) >= args.N_vote))[0]
            np.random.shuffle(valid_idx)
            X = [X[item] for item in valid_idx]
            Y = [Y[item] for item in valid_idx]
            train_X += X[:len(X)//2+1]
            train_Y += Y[:len(X)//2+1]
            train_X_aux += X[-len(X)//2-1:]
            train_Y_aux += Y[-len(X)//2-1:]
        
        train_loader = torch.utils.data.DataLoader(utils_seizure.IIICDoubleLoader(train_X, train_X_aux, train_Y, train_Y_aux),
                batch_size=128, shuffle=True)
        return train_loader

    def trainloader_for_MLDG():
        loader_ls = []
        cur_X, cur_Y = [], []
        for i, (_, (X, Y)) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            valid_idx = np.where((np.sum(np.array(Y), axis=1) >= args.N_vote))[0]
            cur_X += [X[item] for item in valid_idx]
            cur_Y += [Y[item] for item in valid_idx]
            if len(cur_X) >= args.MLDG_threshold:
                loader_ls.append(
                    torch.utils.data.DataLoader(
                        utils_seizure.IIICLoader(cur_X, cur_Y),
                        batch_size=32, 
                        shuffle=True)
                )
                cur_X, cur_Y = [], []
        loader_ls.append(
                torch.utils.data.DataLoader(
                    utils_seizure.IIICLoader(cur_X, cur_Y),
                    batch_size=128, 
                    shuffle=True)
            )
        return loader_ls

    def valloader_for_all():
        val_X, val_Y = [], []
        for _, (X, Y) in val_pat_map.items():
            valid_idx = np.where((np.sum(np.array(Y), axis=1) >= args.N_vote))[0]
            X = [X[item] for item in valid_idx]
            Y = [Y[item] for item in valid_idx]
            val_X += X
            val_Y += Y
        val_loader = torch.utils.data.DataLoader(utils_seizure.IIICLoader(val_X, val_Y),
                batch_size=128, shuffle=False)
        return val_loader

    def testloader_for_all():
        test_X, test_Y = [], []
        for _, (X, Y) in test_pat_map.items():
            valid_idx = np.where((np.sum(np.array(Y), axis=1) >= args.N_vote))[0]
            X = [X[item] for item in valid_idx]
            Y = [Y[item] for item in valid_idx]
            test_X += X
            test_Y += Y
        test_loader = torch.utils.data.DataLoader(utils_seizure.IIICLoader(test_X, test_Y),
                batch_size=128, shuffle=False)
        return test_loader

    # load model
    if args.model == "base":
        train_loader = trainloader_for_other()
        model = SeizureBase(device, args.dataset).to(device)
    elif args.model == "dev":
        train_loader = trainloader_for_other()
        model = SeizureDev(device, args.dataset).to(device)
    elif args.model == "condadv":
        train_loader = trainloader_for_adv()
        model = SeizureCondAdv(device, args.dataset, args.N_pat).to(device)
    elif args.model == "DANN":
        train_loader = trainloader_for_adv()
        model = SeizureDANN(device, args.dataset, args.N_pat).to(device)
    elif args.model == "IRM":
        train_loader = trainloader_for_other()
        model = SeizureIRM(device, args.dataset).to(device)
    elif args.model == "SagNet":
        train_loader = trainloader_for_other()
        model = SeizureSagNet(device, args.dataset).to(device)
    elif args.model == "PCL":
        train_loader = trainloader_for_other()
        model = SeizurePCL(device, args.dataset).to(device)
    elif args.model == "MLDG":
        loader_ls = trainloader_for_MLDG()
        generator_ls = [iter(item) for item in loader_ls]
        model = SeizureMLDG(device, args.dataset).to(device)

    test_loader = testloader_for_all()
    val_loader = valloader_for_all()

    model_name = (args.dataset + '_' + args.model + '_' + str(args.N_pat) + '_{}').format(time.time())
    print (model_name)

    test_array, val_array = [], []
    test_kappa_array, val_kappa_array = [], []
    test_f1_array, val_f1_array = [], []
    
    for i in range (args.epochs):
        tic = time.time()
        if args.model == "DANN":
            model.train_SoftCEL(train_loader, device, i, 50)
        elif args.model == "dev":
            train_loader_dev = trainloader_for_dev()
            model.train_SoftCEL(train_loader_dev, device)
        elif args.model == "MLDG":
            N = len(generator_ls)
            val_idx = np.random.choice(np.arange(N), round(N * 0.1), replace=False)
            model.train_SoftCEL(generator_ls, loader_ls, val_idx, device)
        else:
            model.train_SoftCEL(train_loader, device)
            
        result, gt = model.test(test_loader, device)
        print ('{}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
            i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result), time.time() - tic))
        test_array.append(accuracy_score(gt, result))
        test_kappa_array.append(cohen_kappa_score(gt, result))
        test_f1_array.append(weighted_f1(gt, result))
        with open('log_new/Seizure/{}.log'.format(model_name), 'a') as outfile:
            print ('{}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}'.format(
                i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result)), file=outfile)
        
        result, gt = model.test(val_loader, device)
        print ('{}-th val accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
            i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result), time.time() - tic))
        val_array.append(accuracy_score(gt, result))
        val_kappa_array.append(cohen_kappa_score(gt, result))
        val_f1_array.append(weighted_f1(gt, result))
        with open('log_new/Seizure/{}.log'.format(model_name), 'a') as outfile:
             print ('{}-th val accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}'.format(
                i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result)), file=outfile)

        # save model
        torch.save(model.state_dict(), 'pre-trained/seizure/{}-{}.pt'.format(i, model_name))
        print ()