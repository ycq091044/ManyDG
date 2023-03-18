import torch
import numpy as np
import os
import pickle

def set_random_seed(seed=1234):
    seed = seed
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

def y_transform(y):
    if y == 'W':
        y = 0
    elif y == 'R':
        y = 4
    elif y in ['1', '2', '3']:
        y = int(y)
    elif y == '4':
        y = 3
    else:
        y = 0
    return y

class SleepIDLoader(torch.utils.data.Dataset):
    def __init__(self, indices, ID):
        self.indices = indices
        self.ID = ID
        self.dir = '/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/cassette_processed'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.indices[index])
        sample = pickle.load(open(path, 'rb'))
        X, y, id = sample['X'], y_transform(sample['y']), self.ID[index]
        return torch.FloatTensor(X), y, id

class SleepLoader(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.indices = indices
        self.dir = '/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/cassette_processed'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.indices[index])
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], y_transform(sample['y'])
        return torch.FloatTensor(X), y

class SleepDoubleLoader(torch.utils.data.Dataset):
    def __init__(self, indices, indices_aux):
        self.indices = indices
        self.indices_aux = indices_aux
        self.dir = '/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/cassette_processed'
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.indices[index])
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], y_transform(sample['y'])

        path_aux = os.path.join(self.dir, self.indices_aux[index])
        sample_aux = pickle.load(open(path_aux, 'rb'))
        X_aux, y_aux = sample_aux['X'], y_transform(sample_aux['y'])

        return torch.FloatTensor(X), torch.FloatTensor(X_aux), y, y_aux

def load_and_dump(seed=1234):

    seed = seed
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    
    root = '/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/cassette_processed'

    all_file = os.listdir(root)
    all_pats = np.unique([item.split('-')[1][:-1] for item in all_file])

    pat_dict = {pat: [] for pat in all_pats}
    for file in all_file:
        pat_dict[file.split('-')[1][:-1]].append(file)
        
    train_pat = np.random.choice(all_pats, round(len(all_pats) * 0.8), replace=False)
    test_pat = np.random.choice(list(set(all_pats) - set(train_pat.tolist())), round(len(all_pats) * 0.1), replace=False)
    val_pat = list(set(all_pats) - set(train_pat.tolist()) - set(test_pat.tolist()))

    train_pat_dict = {pat: pat_dict[pat] for pat in train_pat}
    test_pat_dict = {pat: pat_dict[pat] for pat in test_pat}
    val_pat_dict = {pat: pat_dict[pat] for pat in val_pat}

    train_size = 0
    for pat in train_pat:
        train_size += len(train_pat_dict[pat])
    test_size = 0
    for pat in test_pat:
        test_size += len(test_pat_dict[pat])
    val_size = 0
    for pat in val_pat:
        val_size += len(val_pat_dict[pat])

    print ('number of train/test/val patients: {}, {}, {}'.format(len(train_pat), len(test_pat), len(val_pat)))
    print ('number of train/test/val samples: {}, {}, {}'.format(train_size, test_size, val_size))
    
    pickle.dump(test_pat_dict, open('data/sleep/test_pat_map_sleep.pkl', 'wb'))
    pickle.dump(val_pat_dict, open('data/sleep/val_pat_map_sleep.pkl', 'wb'))
    pickle.dump(train_pat_dict, open('data/sleep/train_pat_map_sleep.pkl', 'wb'))
    return train_pat_dict, test_pat_dict, val_pat_dict

def load():
    test_pat_map = pickle.load(open('data/sleep/test_pat_map_sleep.pkl', 'rb'))
    val_pat_map = pickle.load(open('data/sleep/val_pat_map_sleep.pkl', 'rb'))
    train_pat_map = pickle.load(open('data/sleep/train_pat_map_sleep.pkl', 'rb'))
    return train_pat_map, test_pat_map, val_pat_map