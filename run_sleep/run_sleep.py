import os
import argparse
import utils_sleep
import torch
import numpy as np
import time
from model_sleep import SleepBase, SleepDev, SleepCondAdv, SleepDANN, SleepIRM, SleepSagNet, SleepPCL, SleepMLDG

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
    parser.add_argument('--model', type=str, required=True, help="choose from base, dev, condadv, DANN, IRM, SagNet, PCL, MLDG")
    parser.add_argument('--cuda', type=int, default=0, help="which cuda")
    parser.add_argument('--N_pat', type=int, default=100, help="number of patients")
    parser.add_argument('--dataset', type=str, default="sleep", help="dataset name")
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
    path = "data/sleep/test_pat_map_sleep.pkl"
    if os.path.exists(path):
        train_pat_map, test_pat_map, val_pat_map = utils_sleep.load()
    else:   
        train_pat_map, test_pat_map, val_pat_map = utils_sleep.load_and_dump()

    def trainloader_for_other():
        train_X = []

        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            train_X += X
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepLoader(train_X),
                    batch_size=256, shuffle=True, num_workers=20)
        return train_loader

    def trainloader_for_adv():
        train_X, train_ID = [], []

        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            train_X += X
            train_ID += [i for _ in X]
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepIDLoader(train_X, train_ID),
                    batch_size=256, shuffle=True, num_workers=20)
        return train_loader

    def trainloader_for_dev():
        train_X = []
        train_X_aux = []
        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            np.random.shuffle(X)
            train_X += X[:len(X)//2 + 1]
            train_X_aux += X[-len(X)//2 - 1:]
        
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepDoubleLoader(train_X, train_X_aux),
                batch_size=256, shuffle=True, num_workers=20)
        return train_loader

    def trainloader_for_MLDG():
        loader_ls = []
        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            loader_ls.append(
                torch.utils.data.DataLoader(
                    utils_sleep.SleepLoader(X),
                    batch_size=16, 
                    shuffle=True)
            )
        return loader_ls

    def valloader_for_all():
        val_X = []
        for _, X in val_pat_map.items():
            val_X += X
        val_loader = torch.utils.data.DataLoader(utils_sleep.SleepLoader(val_X),
                batch_size=256, shuffle=False, num_workers=20)
        return val_loader

    def testloader_for_all():
        test_X = []
        for _, X in test_pat_map.items():
            test_X += X
        test_loader = torch.utils.data.DataLoader(utils_sleep.SleepLoader(test_X),
                batch_size=256, shuffle=False, num_workers=20)
        return test_loader

    # load model
    if args.model == "base":
        train_loader = trainloader_for_other()
        model = SleepBase(device, args.dataset).to(device)
    elif args.model == "dev":
        train_loader = trainloader_for_other()
        model = SleepDev(device, args.dataset).to(device)
    elif args.model == "condadv":
        train_loader = trainloader_for_adv()
        model = SleepCondAdv(device, args.dataset, args.N_pat).to(device)
    elif args.model == "DANN":
        train_loader = trainloader_for_adv()
        model = SleepDANN(device, args.dataset, args.N_pat).to(device)
    elif args.model == "IRM":
        train_loader = trainloader_for_other()
        model = SleepIRM(device, args.dataset).to(device)
    elif args.model == "SagNet":
        train_loader = trainloader_for_other()
        model = SleepSagNet(device, args.dataset).to(device)
    elif args.model == "PCL":
        train_loader = trainloader_for_other()
        model = SleepPCL(device, args.dataset).to(device)
    elif args.model == "MLDG":
        loader_ls = trainloader_for_MLDG()
        generator_ls = [iter(item) for item in loader_ls]
        model = SleepMLDG(device, args.dataset).to(device)

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
            model.train(train_loader, device, i, 50)
        elif args.model == "dev":
            train_loader_dev = trainloader_for_dev()
            model.train(train_loader_dev, device)
        elif args.model == "MLDG":
            N = len(generator_ls)
            val_idx = np.random.choice(np.arange(N), round(N * 0.1), replace=False)
            model.train(generator_ls, loader_ls, val_idx, device)
        else:
            model.train(train_loader, device)
        
        result, gt = model.test(test_loader, device)
        print ('{}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
            i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result), time.time() - tic))
        test_array.append(accuracy_score(gt, result))
        test_kappa_array.append(cohen_kappa_score(gt, result))
        test_f1_array.append(weighted_f1(gt, result))
        with open('log_new/sleep/{}.log'.format(model_name), 'a') as outfile:
            print ('{}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}'.format(
                i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result)), file=outfile)
        
        result, gt = model.test(val_loader, device)
        print ('{}-th val accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
            i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result), time.time() - tic))
        val_array.append(accuracy_score(gt, result))
        val_kappa_array.append(cohen_kappa_score(gt, result))
        val_f1_array.append(weighted_f1(gt, result))
        with open('log_new/sleep/{}.log'.format(model_name), 'a') as outfile:
             print ('{}-th val accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}'.format(
                i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result)), file=outfile)

        # save model
        torch.save(model.state_dict(), 'pre-trained/sleep/{}-{}.pt'.format(i, model_name))
        print ()