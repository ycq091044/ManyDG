from email.encoders import encode_noop
import pickle 
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import sys

class EncounterInfo(object):
    def __init__(self, patient_id, encounter_id, encounter_timestamp, expired,
           readmission):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.encounter_timestamp = encounter_timestamp
        self.mortality = expired
        self.readmission = readmission
        self.dx_ids = dict()
        self.rx_ids = []
        self.labs = {}
        self.physicals = []
        self.treatments = dict()


def process_patient(infile, encounter_dict, hour_threshold):
    inff = open(infile, 'r')
    patient_dict = {}
    for line in tqdm(csv.DictReader(inff)):
        patient_id = line['patienthealthsystemstayid']
        encounter_id = line['patientunitstayid']
        encounter_timestamp = -int(line['hospitaladmitoffset'])
        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append((encounter_timestamp, encounter_id))
    inff.close()

    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples)

    enc_readmission_dict = {}
    for patient_id, time_enc_tuples in patient_dict_sorted.items():
        for idx, time_enc_tuple in enumerate(time_enc_tuples[:-1]):
            time_current = time_enc_tuple[0]
            time_next = time_enc_tuples[idx+1][0]
            enc_id = time_enc_tuple[1]
            if (time_next - time_current) / 60 / 24 < 15:
                enc_readmission_dict[enc_id] = True
            else:
                enc_readmission_dict[enc_id] = False
        last_enc_id = time_enc_tuples[-1][1]
        enc_readmission_dict[last_enc_id] = False

    inff = open(infile, 'r')
    for line in tqdm(csv.DictReader(inff)):
        patient_id = line['patienthealthsystemstayid']
        encounter_id = line['patientunitstayid']
        encounter_timestamp = -int(line['hospitaladmitoffset'])
        discharge_status = line['unitdischargestatus']
        duration_minute = float(line['unitdischargeoffset'])
        expired = True if discharge_status == 'Expired' else False
        readmission = enc_readmission_dict[encounter_id]
        if duration_minute > 60. * hour_threshold:
            continue
        ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired,
                           readmission)
        if encounter_id in encounter_dict:
            print('Duplicate encounter ID!!')
            sys.exit(0)
        encounter_dict[int(encounter_id)] = ei
    inff.close()

    return encounter_dict


def feature_process(input_path, days=3):
    patient_file = input_path + '/patient.csv'
    encounter_dict = {}
    print('Processing patient.csv')
    encounter_dict = process_patient(patient_file, encounter_dict, hour_threshold=24*days)
    return encounter_dict


def process_drugname(uniq_medname):
    """
    Preprocess drug name
    """

    # remove the name start with value
    tmp = [name.lower() for name in uniq_medname if name[0] not in ['.', '<', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']]

    # # split the name and only keep the first word
    # tmp = [name.split(' ')[0].lower() for name in tmp]

    # dropduplicates
    tmp = list(set(tmp))
    tmp.remove('nan')

    # delete the numeric values
    tmp2 = []
    for medname in tmp:
        tmp3 = ""
        for s in medname:
            if s in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                break
            tmp3 += s
        tmp3 = tmp3.rstrip()
        if len(tmp3) >= 3:
            tmp2.append(tmp3)

    tmp2 = sorted(list(set(tmp2)))

    """
    Postprocess drugname
    """
    tmp2 = list(set([name.split(' ')[0] for name in tmp2]))

    return tmp2


def get_idxfile(uniq_diagstring, uniq_diagICD, uniq_labname, uniq_physicalexam, \
                            uniq_treatment, uniq_medname, uniq_meddosage, uniq_medfreq):
    # transform the entity into index

    idx2diagstring = {}
    diagstring2idx = {}

    idx2diagICD = {}
    diagICD2idx = {}

    idx2labname = {}
    labname2idx = {}

    idx2physicalexam = {}
    physicalexam2idx = {}

    idx2treatment = {}
    treatment2idx = {}

    idx2medname = {}
    medname2idx = {}

    idx2meddosage = {}
    meddosage2idx = {}

    idx2medfreq = {}
    medfreq2idx = {}

    for idx, entity in enumerate(uniq_diagstring):
        idx2diagstring[idx] = entity
        diagstring2idx[entity] = idx

    for idx, entity in enumerate(uniq_diagICD):
        idx2diagICD[idx] = entity
        diagICD2idx[entity] = idx
        
    for idx, entity in enumerate(uniq_labname):
        idx2labname[idx] = entity
        labname2idx[entity] = idx

    for idx, entity in enumerate(uniq_physicalexam):
        idx2physicalexam[idx] = entity
        physicalexam2idx[entity] = idx

    for idx, entity in enumerate(uniq_treatment):
        idx2treatment[idx] = entity
        treatment2idx[entity] = idx

    for idx, entity in enumerate(uniq_medname):
        idx2medname[idx] = entity
        medname2idx[entity] = idx

    for idx, entity in enumerate(uniq_meddosage):
        idx2meddosage[idx] = entity
        meddosage2idx[entity] = idx

    for idx, entity in enumerate(uniq_medfreq):
        idx2medfreq[idx] = entity
        medfreq2idx[entity] = idx


    pickle.dump(idx2diagstring, open('./idxFile/idx2diagstring.pkl', 'wb'))
    pickle.dump(diagstring2idx, open('./idxFile/diagstring2idx.pkl', 'wb'))
    pickle.dump(idx2diagICD, open('./idxFile/idx2diagICD.pkl', 'wb'))
    pickle.dump(diagICD2idx, open('./idxFile/diagICD2idx.pkl', 'wb'))
    pickle.dump(idx2labname, open('./idxFile/idx2labname.pkl', 'wb'))
    pickle.dump(labname2idx, open('./idxFile/labname2idx.pkl', 'wb'))
    pickle.dump(idx2physicalexam, open('./idxFile/idx2physicalexam.pkl', 'wb'))
    pickle.dump(physicalexam2idx, open('./idxFile/physicalexam2idx.pkl', 'wb'))
    pickle.dump(idx2treatment, open('./idxFile/idx2treatment.pkl', 'wb'))
    pickle.dump(treatment2idx, open('./idxFile/treatment2idx.pkl', 'wb'))
    pickle.dump(idx2medname, open('./idxFile/idx2medname.pkl', 'wb'))
    pickle.dump(medname2idx, open('./idxFile/medname2idx.pkl', 'wb'))
    pickle.dump(idx2meddosage, open('./idxFile/idx2meddosage.pkl', 'wb'))
    pickle.dump(meddosage2idx, open('./idxFile/meddosage2idx.pkl', 'wb'))
    pickle.dump(idx2medfreq, open('./idxFile/idx2medfreq.pkl', 'wb'))
    pickle.dump(medfreq2idx, open('./idxFile/medfreq2idx.pkl', 'wb'))

    return diagstring2idx, diagICD2idx, labname2idx, physicalexam2idx, treatment2idx, \
                                                    medname2idx, meddosage2idx, medfreq2idx


def collect_med_code(data):
    """ structure of data
    Data: key -> value
    key: patient + ICU ID
    value: the sequence of that visit [event1, event2, event3, ...]
    event: [offset, eventtype, property list]
    """

    # code collection
    diagstring = []
    diagICD = []
    labname = []
    physicalExam = []
    treatment = []
    medname = []
    meddosage = []
    medfreq = []

    for key, seq in data.items():
        for event in seq:
            if event[1] == 'diagnosis':
                diagstring += event[2][::2]
                diagICD += event[2][1::2]
            elif event[1] == 'lab':
                labname += event[2][::2]
            elif event[1] == 'physicalExam':
                physicalExam += event[2]
            elif event[1] == 'treatment':
                treatment += event[2]
            elif event[1] == 'medication':
                medname += event[2][::4]
                meddosage += event[2][1::4]
                medfreq += event[2][2::4]
            
    uniq_diagstring = np.unique(diagstring)
    uniq_diagICD = np.unique(diagICD)
    uniq_labname = np.unique(labname)
    uniq_physicalexam = np.unique(physicalExam)
    uniq_treatment = np.unique(treatment)
    uniq_medname = np.unique(medname)
    uniq_meddosage = np.unique(meddosage)
    uniq_medfreq = np.unique(medfreq)


    # process drugname
    uniq_medname = process_drugname(uniq_medname)

    print ('#. diagstring:', len(uniq_diagstring))
    print ('#. diagICD:', len(uniq_diagICD))
    print ('#. labname:', len(uniq_labname))
    print ('#. physicalexam:', len(uniq_physicalexam))
    print ('#. treatment:', len(uniq_treatment))
    print ('#. medname:', len(uniq_medname))
    print ('#. meddosage:', len(uniq_meddosage))
    print ('#. medfreq:', len(uniq_medfreq))

    # get idx files and dump
    return get_idxfile(uniq_diagstring, uniq_diagICD, uniq_labname, uniq_physicalexam, \
                                    uniq_treatment, uniq_medname, uniq_meddosage, uniq_medfreq)


def aggregate_consecutive(data):
    refine_data = dict()
    for k, seq in tqdm(data.items()):
        collect_seq = []
        tmp = []
        flag = [None, None]
        for event in seq:
            if flag[0] is None:
                tmp = [event[0], event[1], event[2]]
                flag[0] = event[0]; flag[1] = event[1]
            elif (np.abs(flag[0] - event[0]) < 120) and (flag[1] == event[1]):
                flag[0] = event[0]; tmp[2] += event[2]
            else:
                collect_seq.append(tmp)
                tmp = [event[0], event[1], event[2]]
                flag[0] = event[0]; flag[1] = event[1]
        collect_seq.append(tmp)
        refine_data[k] = collect_seq
    return refine_data

def clean_sequence_data(encounter_dict, data):
    """ Structure of clean_data
    clean_data: key -> value
    key: the unique identifer, the same as data
    value: is the event feature list [event1, event2, event3, ...]

    diagnosis event: ['diagnosis', [idx list of string], [idx list of ICD]]
    lab event: ['lab', vector of labtest]
    infusion event: ['infusion', vector of infusion value]
    physicalexam event: ['physicalexam', idx list of physicalexam]
    treatment event: ['treatment', idx list of treatment]
    medication event: ['medication', [idx list of drug], vector of dosage, vector of frequency]
    """

    clean_data = {}
    # kdata = pickle.load(open('../idxFile/key2.pkl', 'rb'))
    
    for key, seq in tqdm(data.items()):
        # if key not in kdata or key not in encounter_dict.keys(): continue
        if key not in encounter_dict.keys(): continue
        tmp = []; tmp_seq = []
        for event in seq:
            if event[1] == 'diagnosis':
                tmp1, tmp2, tmp3 = [], [], []
                for x, y in zip(event[2][::2], event[2][1::2]):
                    if type(x) != type('a') or type(y) != type('a'):
                        continue
                    tmp1.append(diagstring2idx[x]); tmp2.append(diagICD2idx[y]); tmp3.append(x)
                if len(tmp1) > 0:
                    tmp.append([event[0], 0, [tmp1, tmp2], tmp3])
            elif event[1] == 'lab':
                zvec = np.zeros(len(labname2idx))
                tmp2 = []
                for i in event[2][1::2]:
                    if isinstance(i, int) or isinstance(i, float):
                        if not np.isnan(i):
                            tmp2.append(i)
                        else:
                            tmp2.append(0)
                    elif len(i.replace(".", "").replace(" ","")) == 0:
                        tmp2.append(0)
                    else:
                        i = i.replace(" ", "")
                        if (i[0] == '<' or i[0] == '>') and i[-1] == '%':
                            tmp2.append(float(i[1:-1]))
                        elif i[-1] == '%':
                            tmp2.append(float(i[:-1]))
                        elif i[0] == '<' or i[0] == '>':
                            tmp2.append(float(i[1:]))
                        else:
                            tmp2.append(float(i))
                zvec[[labname2idx[i] for i in event[2][::2]]] = tmp2
                tmp.append([event[0], 1, zvec, event[2][::2]])
            elif event[1] == 'physicalExam':
                tmp.append([event[0], 2, [physicalexam2idx[i] for i in event[2]], event[2]])
            elif event[1] == 'treatment':
                tmp.append([event[0], 3, [treatment2idx[i] for i in event[2]], event[2]])
            elif event[1] == 'medication':
                drug_list = []
                zvec1 = np.zeros(len(medname2idx))
                zvec2 = np.zeros(len(medname2idx))
                tmp3 = []
                for i, x, y in zip(event[2][::4], event[2][1::4], event[2][2::4]):
                    if  type(i) != type('a') or type(x) != type('a') or type(y) != type('a'):
                        continue
                    if i.split(' ')[0] not in medname2idx:
                        continue
                    tmp3.append(i.split(' ')[0])
                    zvec1[[medname2idx[i.split(' ')[0]]]] = meddosage2idx[x]
                    zvec2[[medname2idx[i.split(' ')[0]]]] = medfreq2idx[y]
                    drug_list.append(medname2idx[i.split(' ')[0]])
                if len(drug_list) > 0:
                    tmp.append([event[0], 4, [drug_list, zvec1, zvec2], tmp3])
        if len(tmp) >= 3:        
            clean_data[key] = tmp

    return clean_data


def patient_info(patient):
    patientMap = {}
    genderMap = {'Female':0, 'Male':1, 'Other':2, 'Unknown': 3}
    ethMap = {'Caucasian':0, 'African American':1, 'Hispanic':2, 'Asian':3, 'Native American':4, 'Other/Unknown':5}

    for k, gender, age, eth in patient[['patientunitstayid', 'gender', 'age', 'ethnicity']].values:
        if type(gender) != type('a') or type(age) != type('a') or type(eth) != type('a'):
            continue
        if age[0] == '>':
            age = age[2:]
        patientMap[k] = [genderMap[gender], int(age) / 10, ethMap[eth]]

    return patientMap


if __name__ == '__main__':

    # load data
    print ('load raw data...')
    data = pickle.load(open('./eICU_event_sequence.pkl', 'rb'))
    root = '/srv/local/data/physionet.org/files/eicu-crd/2.0/'
    patient = pd.read_csv(root + 'patient.csv')
    hospital = pd.read_csv(root + 'hospital.csv')
    print ()

    ##############
    update = False
    ##############


    if not update:
        # collect the med codes and their idx mapping
        print ('clean the medical codes...')
        diagstring2idx, diagICD2idx, labname2idx, physicalexam2idx, treatment2idx, \
                            medname2idx, meddosage2idx, medfreq2idx = collect_med_code(data)
        print ()

    else:
        print ('load medical code idxfiles...')
        diagstring2idx = pickle.load(open('./idxFile/diagstring2idx.pkl', 'rb'))
        diagICD2idx = pickle.load(open('./idxFile/diagICD2idx.pkl', 'rb'))
        labname2idx = pickle.load(open('./idxFile/labname2idx.pkl', 'rb'))
        physicalexam2idx = pickle.load(open('./idxFile/physicalexam2idx.pkl', 'rb'))
        treatment2idx = pickle.load(open('./idxFile/treatment2idx.pkl', 'rb'))
        medname2idx = pickle.load(open('./idxFile/medname2idx.pkl', 'rb'))
        meddosage2idx = pickle.load(open('./idxFile/meddosage2idx.pkl', 'rb'))
        medfreq2idx = pickle.load(open('./idxFile/medfreq2idx.pkl', 'rb'))
        print ()

    # get .readmission and .mortality labels (3-day stay as the threshold)
    print ('patient readmission and mortality label extraction...')
    encounter_dict = feature_process(root, 3)
    print ()

    # clean the sequence data
    print ('clean the sequence data...')
    # agg_data = aggregate_consecutive(data)
    clean_data = clean_sequence_data(encounter_dict, data)
    print ()

    # obtain patient-level info.
    print ('process patient-level feature...')
    patientMap = patient_info(patient)
    print ()

    # clean the feature (X) and the labels (y)
    print ('dump the X and y information...')
    patientF = []
    Xdata, Yreadmission, Ymortality, keydata = [], [], [], []

    pat_level_dataset = {}
    count = [0, 0]
    for idx, (k, seq) in tqdm(enumerate(clean_data.items())):
        if k not in patientMap or k not in encounter_dict: continue
        if (encounter_dict[k].patient_id not in pat_level_dataset) and (len(seq) >= 3):
            pat_level_dataset[encounter_dict[k].patient_id] = []
        if (len(seq) >= 3) and (len(seq) < 300):
            pat_level_dataset[encounter_dict[k].patient_id].append([
                patientMap[k],
                seq,
                encounter_dict[k].readmission,
                encounter_dict[k].mortality,
                k
            ])
            if encounter_dict[k].mortality:
                count[1] += 1
            else:
                count[0] += 1
        
    print (np.array(count) / np.sum(count))
    pickle.dump(pat_level_dataset, open('./pat_level_dataset.pkl', 'wb'))

