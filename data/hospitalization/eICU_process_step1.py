import pandas as pd
import pickle
from tqdm import tqdm

def transform(comp_list):
    result = []
    tmp = None
    for i, j, k in sorted(comp_list, key=lambda item: item[0]):
#         if k in ['nurseCharting']:
#             continue
#         print (i, j, k, "##", tmp, '##')
        if tmp is None:
            tmp = [i, j, k]
        elif (i == tmp[0]) and (j == tmp[1]):
            tmp[2] += k
        else:
            result.append(tmp)
            tmp = None
    if tmp is not None:
        result.append(tmp)
    return result

if __name__ == "__main__":

    # load eICU data
    root = '/srv/local/data/physionet.org/files/eicu-crd/2.0/'
    patient = pd.read_csv(root + 'patient.csv')
    diagnosis = pd.read_csv(root + 'diagnosis.csv')
    lab = pd.read_csv(root + 'lab.csv')
    allergy = pd.read_csv(root + 'allergy.csv')
    infusionDrug = pd.read_csv(root + 'infusionDrug.csv')
    intakeOutput = pd.read_csv(root + 'intakeOutput.csv')
    physicalExam = pd.read_csv(root + 'physicalExam.csv')
    nurseCare = pd.read_csv(root + 'nurseCare.csv')
    respiratoryCare = pd.read_csv(root + 'respiratoryCare.csv')
    treatment = pd.read_csv(root + 'treatment.csv')
    nurseCharting = pd.read_csv(root + 'nurseCharting.csv')
    nurseAssessment = pd.read_csv(root + 'nurseAssessment.csv')
    note = pd.read_csv(root + 'note.csv')
    medication = pd.read_csv(root + 'medication.csv')
    print ('loaded all files, ready!')

    Tables = [diagnosis, lab, allergy, infusionDrug, intakeOutput, physicalExam, \
          nurseCare, respiratoryCare, treatment, nurseCharting, nurseAssessment, \
          note, medication]

    # initialize the sequence, key: patientunitstayid
    Dict = {}
    for pat_unit_id in patient.patientunitstayid.unique():
        Dict[pat_unit_id] = []
    print ('preparing dict')

    # diagnosis
    for i, j in diagnosis.groupby('patientunitstayid'):
        for t, p1, p2 in j[['diagnosisoffset', 'diagnosisstring', 'icd9code']].values:
            Dict[i].append((t, 'diagnosis', [p1,p2]))
    print ('merge with diagnosis table')

    # lab
    for i, j in lab.groupby('patientunitstayid'):
        for t, p1, p2 in j[['labresultoffset', 'labname', 'labresulttext']].values:
            Dict[i].append((t, 'lab', [p1, p2]))
    print ('merge with lab table')

    # allergy
    for i, j in allergy.groupby('patientunitstayid'):
        for t, p in j[['allergyoffset', 'allergytype']].values:
            Dict[i].append((t, p, 'allergy'))
    print ('merge with allergy table')

    # infusion
    for i, j in infusionDrug.groupby('patientunitstayid'):
        for t, p1, p2 in j[['infusionoffset', 'drugname', 'drugrate']].values:
            Dict[i].append((t, 'infusion', [p1, p2]))
    print ('merge with infusion table')

    # intakeOutput
    for i, j in intakeOutput.groupby('patientunitstayid'):
        for t, p in j[['intakeoutputoffset', 'celllabel']].values:
            Dict[i].append((t, p, 'intakeOutput'))        
    print ('merge with intakeOutput table')

    # physicalExam
    for i, j in physicalExam.groupby('patientunitstayid'):
        for t, p in j[['physicalexamoffset', 'physicalexampath']].values:
            Dict[i].append((t, 'physicalExam', [p]))
    print ('merge with physical Exam table')

    # nurseCare
    for i, j in nurseCare.groupby('patientunitstayid'):
        for t, p in j[['nursecareoffset', 'celllabel']].values:
            Dict[i].append((t, p, 'nurseCare'))
    print ('merge with nurseCare table')

    # respiratoryCare
    for i, j in respiratoryCare.groupby('patientunitstayid'):
        for t, p in j[['respcarestatusoffset', 'currenthistoryseqnum']].values:
            Dict[i].append((t, p, 'respiratoryCare'))       
    print ('merge with erspiratoryCare table')

    # treatment
    for i, j in treatment.groupby('patientunitstayid'):
        for t, p in j[['treatmentoffset', 'treatmentstring']].values:
            Dict[i].append((t, 'treatment', [p]))
    print ('merge with treatment table')

    # nurseCharting
    for i, j in nurseCharting.groupby('patientunitstayid'):
        for t, p in j[['nursingchartoffset', 'nursingchartcelltypevallabel']].values:
            Dict[i].append((t, p, 'nurseCharting'))
    print ('merge with nurseCharting table')

    # nurseAssessment
    for i, j in nurseAssessment.groupby('patientunitstayid'):
        for t, p in j[['nurseassessoffset', 'celllabel']].values:
            Dict[i].append((t, p, 'nurseAssessment'))
    print ('merge with nurseAssessment table')

    # note
    for i, j in note.groupby('patientunitstayid'):
        for t, p in j[['noteoffset', 'notetype']].values:
            Dict[i].append((t, p, 'note'))
    print ('merge with note table')
    # medication
    for i, j in medication.groupby('patientunitstayid'):
        for t, p1, p2, p3, p4 in j[['drugstartoffset', 'drugname', 'dosage', 'frequency', 'drugstopoffset']].values:
            Dict[i].append((t, 'medication', [p1, p2, p3, p4]))       
    print ('merge with medication table')

    # transform the records
    Dict2 = {}
    for k, v in tqdm(Dict.items()):
        Dict2[k] = transform(v)

    # dump
    pickle.dump(Dict2, open('./eICU_event_sequence.pkl', 'wb'))

