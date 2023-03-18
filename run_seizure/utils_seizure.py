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

class IIICIDLoader(torch.utils.data.Dataset):
    def __init__(self, train_X, train_Y, train_ID):
        self.X = train_X
        self.Y = train_Y
        self.ID = train_ID
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y, id = self.X[index], self.Y[index], self.ID[index]
        return torch.FloatTensor(X), y, id

class IIICLoader(torch.utils.data.Dataset):
    def __init__(self, train_X, train_Y):
        self.X = train_X
        self.Y = train_Y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y = self.X[index], self.Y[index]
        return torch.FloatTensor(X), y

class IIICDoubleLoader(torch.utils.data.Dataset):
    def __init__(self, train_X, train_X_aux, train_Y, train_Y_aux):
        self.X = train_X
        self.Y = train_Y
        self.X_aux = train_X_aux
        self.Y_aux = train_Y_aux

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y = self.X[index], self.Y[index]
        X_aux, y_aux = self.X_aux[index], self.Y_aux[index]
        return torch.FloatTensor(X), torch.FloatTensor(X_aux), y, y_aux

def load_and_dump(seed=1234):

    seed = seed
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)

    # load the key/Y/X file
    # key: patient identify file
    # Y: label file
    # X: signal file
    key_label_path = '/srv/local/data/IIIC_data/data_1109'
    train_key = np.load(os.path.join(key_label_path, 'training_key.npy'))
    train_Y = np.load(os.path.join(key_label_path, 'training_Y.npy'))
    test_key = np.load(os.path.join(key_label_path, 'test_key.npy'))
    test_Y = np.load(os.path.join(key_label_path, 'test_Y.npy'))

    data_path = '/srv/local/data/IIIC_data/17_data_EEG_1115'
    train_X = np.load(os.path.join(data_path, 'training_X.npy'))
    test_X = np.load(os.path.join(data_path, 'test_X.npy'))

    key = np.concatenate([train_key, test_key], axis=0)
    X = np.concatenate([train_X, test_X], axis=0)
    Y = np.concatenate([train_Y, test_Y], axis=0)

    # first combine all the train and test patients and then re-split into train/test/val \
    # (since the original split is not even, has different distribution in train and test)
    all_patient = set([item.split('_')[0] for item in train_key]).union(set([item.split('_')[0] for item in test_key]))
    print ('number of all patients: {}'.format(len(all_patient)))
    all_patient = list(all_patient)

    # [remove all sklearn code to work on machine without sklearn]
    # from sklearn.model_selection import train_test_split
    # train_pat, test_pat, _, _ = train_test_split(all_patient, all_patient, test_size=0.2, random_state=seed)
    # val_pat, test_pat, _, _ = train_test_split(test_pat, test_pat, test_size=0.5, random_state=seed)
    train_pat = np.random.choice(all_patient, round(len(all_patient) * 0.8), replace=False)
    test_pat = np.random.choice(list(set(all_patient) - set(train_pat.tolist())), round(len(all_patient) * 0.1), replace=False)
    val_pat = list(set(all_patient) - set(train_pat.tolist()) - set(test_pat.tolist()))
    print ('number of train/test/val patients: {}, {}, {}'.format(len(train_pat), len(test_pat), len(val_pat)))

    # get three dict
    # key: pat_id
    # value: [[signals], [labels]]
    train_idx, test_idx, val_idx = [], [], []
    train_pat_map = {}
    val_pat_map = {}
    test_pat_map = {}
    for idx, item in enumerate(key):
        pat_id = item.split('_')[0]
        tmp = X[idx].max()
        if (tmp == 0) or (tmp == 500): # anomaly data
            continue

        if pat_id in train_pat:
            train_idx.append(idx)
            if pat_id not in train_pat_map:
                train_pat_map[pat_id] = [[X[idx]], [Y[idx]]]
            else:
                train_pat_map[pat_id][0].append(X[idx])
                train_pat_map[pat_id][1].append(Y[idx])
        elif pat_id in test_pat:
            test_idx.append(idx)
            if pat_id not in test_pat_map:
                test_pat_map[pat_id] = [[X[idx]], [Y[idx]]]
            else:
                test_pat_map[pat_id][0].append(X[idx])
                test_pat_map[pat_id][1].append(Y[idx])
        elif pat_id in val_pat:
            val_idx.append(idx)
            if pat_id not in val_pat_map:
                val_pat_map[pat_id] = [[X[idx]], [Y[idx]]]
            else:
                val_pat_map[pat_id][0].append(X[idx])
                val_pat_map[pat_id][1].append(Y[idx])

    # the indices
    train_idx, test_idx, val_idx = np.array(train_idx), np.array(test_idx), np.array(val_idx)

    print ('number of train/test/val samples: {}, {}, {}'.format(len(train_idx), len(test_idx), len(val_idx)))

    pickle.dump(test_pat_map, open('data/Seizure/test_pat_map_seizure.pkl', 'wb'))
    pickle.dump(val_pat_map, open('data/Seizure/val_pat_map_seizure.pkl', 'wb'))
    pickle.dump(train_pat_map, open('data/Seizure/train_pat_map_seizure.pkl', 'wb'))

    return train_pat_map, test_pat_map, val_pat_map

def load():
    test_pat_map = pickle.load(open('data/Seizure/test_pat_map_seizure.pkl', 'rb'))
    train_pat_map = pickle.load(open('data/Seizure/train_pat_map_seizure.pkl', 'rb'))
    val_pat_map = pickle.load(open('data/Seizure/val_pat_map_seizure.pkl', 'rb'))
    
    return train_pat_map, test_pat_map, val_pat_map