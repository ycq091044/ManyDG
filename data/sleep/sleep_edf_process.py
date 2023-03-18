import sys
import mne
import numpy as np
import os
from multiprocessing import Process
import pickle
import argparse

def sample_process(root_folder, k, N, epoch_sec, index):
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))

            root_folder = os.path.join(root, 'sleep-cassette')
            pat_files = list(filter(lambda x: x[:5] == j, os.listdir(root_folder)))
            pat_nights = [item[:6] for item in pat_files]
            for pat_per_night in pat_nights:
                # load signal "X" part
                data = mne.io.read_raw_edf(root_folder + '/' + list(filter(lambda x: (x[:6] == pat_per_night) and ('PSG' in x), pat_files))[0])
                X = data.get_data()[:2, :]

                # load label "Y" part
                ann = mne.read_annotations(root_folder + '/' + list(filter(lambda x: (x[:6] == pat_per_night) and ('Hypnogram' in x), pat_files))[0])
                labels = []
                for dur, des in zip(ann.duration, ann.description):
                    for i in range(int(dur) // 30):
                        labels.append(des[-1])

                # slice the EEG signals into non-overlapping windows, window size = sampling rate per second * second time = 100 * windowsize
                for slice_index in range(X.shape[1] // (100 * epoch_sec)):
                    # ingore the no labels
                    if labels[slice_index] == '?': continue
                    path = os.path.join(root, 'cassette_processed/cassette-' + pat_per_night + '-' + str(slice_index) + '.pkl')
                    pickle.dump({'X': X[:, slice_index * 100 * epoch_sec: (slice_index+1) * 100 * epoch_sec], \
                        'y': labels[slice_index]}, open(path, 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=20, help="How many processes to use")
    args = parser.parse_args()

    root = '/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0'
    out_root = os.path.join(root, "cassette_processed")
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    data_folder = os.path.join(root, 'sleep-cassette')

    all_index = np.unique([path[:5] for path in os.listdir(data_folder)])
    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(target=sample_process, args=(root, k, N, epoch_sec, all_index))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()