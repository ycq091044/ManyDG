from cProfile import label
import pickle
from pickletools import pyset 
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, patientF, Xdata, ydata1, ydata2, index):
        self.patientF = [patientF[idx] for idx in index]
        self.Xdata = [Xdata[idx] for idx in index]
        self.ydata1 = [ydata1[idx] for idx in index]
        self.ydata2 = [ydata2[idx] for idx in index]
        
    def __len__(self):
        return len(self.Xdata)
    
    def __getitem__(self, index):
        return self.patientF[index], self.Xdata[index], self.ydata1[index], self.ydata2[index]

def load():
    data_path = './data'
    patientData = list(pickle.load(open('{}/hospitalization/pat_level_dataset.pkl'.format(data_path), 'rb')).values())

    all_idx = np.arange(len(patientData))
    train_idx = list(np.random.choice(all_idx, round(len(all_idx) * 0.8), replace=False))
    val_idx = list(np.random.choice(list(set(all_idx) - set(train_idx)), round(len(all_idx) * 0.1), replace=False))
    test_idx = list(set(all_idx) - set(train_idx) - set(val_idx))
    train_data = [patientData[i] for i in train_idx]
    val_data = [patientData[i] for i in val_idx]
    test_data = [patientData[i] for i in test_idx]
    print("Length of train dataset:", sum([len(item) for item in train_data]))
    print("Length of val dataset:", sum([len(item) for item in val_data]))
    print("Length of test dataset:", sum([len(item) for item in test_data]))
    return train_data, val_data, test_data