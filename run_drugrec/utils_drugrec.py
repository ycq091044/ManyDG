import dill
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import sys

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def ddi_rate_score(record, path='data/drugrec/ddi_A_final.pkl'):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt

def load():
    data_path = 'data/drugrec/records_final.pkl'
    voc_path = 'data/drugrec/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    ehr_adj_path = 'data/drugrec/ehr_adj_final.pkl'
    ddi_adj_path = 'data/drugrec/ddi_A_final.pkl'
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))

    train_pat_map = []
    for i, patient in enumerate(data_train):
        single_pat_data = []
        cur_feature = []
        for visit in patient:
            cur_feature.append(visit)
            single_pat_data.append(cur_feature.copy())
        train_pat_map.append(single_pat_data)
    
    test_pat_map = []
    for i, patient in enumerate(data_test):
        single_pat_data = []
        cur_feature = []
        for visit in patient:
            cur_feature.append(visit)
            single_pat_data.append(cur_feature.copy())
        test_pat_map.append(single_pat_data)
    
    val_pat_map = []
    for i, patient in enumerate(data_eval):
        single_pat_data = []
        cur_feature = []
        for visit in patient:
            cur_feature.append(visit)
            single_pat_data.append(cur_feature.copy())
        val_pat_map.append(single_pat_data)

    return train_pat_map, test_pat_map, val_pat_map, voc_size, ehr_adj, ddi_adj

def multi_label_metric(pre, gt, threshold=0.4):
    """
    pre is a float matrix in [0, 1]
    gt is a binary matrix
    """
    def jaccard(pre, gt):
        score = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(pre[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            union = set(predicted) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)
    
    def precision_auc(pre, gt):
        all_micro = []
        for b in range(gt.shape[0]):
            all_micro.append(average_precision_score(gt[b], pre[b], average='macro'))
        return np.mean(all_micro)
    
    def prc_recall(pre, gt):
        score_prc = []
        score_recall = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(pre[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            prc_score = 0 if len(predicted) == 0 else len(inter) / len(predicted)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score_prc.append(prc_score)
            score_recall.append(recall_score)
        return score_prc, score_recall

    def average_f1(prc, recall):
        score = []
        for idx in range(len(prc)):
            if prc[idx] + recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*prc[idx]*recall[idx] / (prc[idx] + recall[idx]))
        return np.mean(score)

    ja = jaccard(pre, gt)
    prauc = precision_auc(pre, gt)
    prc_ls, recall_ls = prc_recall(pre, gt)
    f1 = average_f1(prc_ls, recall_ls)

    return ja, prauc, f1

def old_multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)
