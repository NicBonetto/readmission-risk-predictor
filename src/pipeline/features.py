import pandas as pd 
import duckdb
import numpy as np
from sklearn.preprocessing import OneHotEncoder

conn = duckdb.connect('../../data/duckdb.mimic')

admissions = conn.execute('SELECT * FROM admissions WHERE deathtime IS NULL;').df()
patients = conn.execute('SELECT subject_id, gender, anchor_age AS age FROM patients;').df()
diagnoses = conn.execute('SELECT * FROM diagnoses_icd;').df()
lab_events = conn.execute('SELECT * FROM labevents;').df()
prescriptions = conn.execute('SELECT * FROM prescriptions;').df()
procedures = conn.execute('SELECT * FROM procedures_icd;').df()

df = admissions.merge(patients, on='subject_id')

df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])

df = df.sort_values('admittime', ascending=True)

first_admissions = df.groupby('subject_id').first().reset_index()

def was_readmitted(row):
    subsequent = df[
        (df['subject_id'] == row['subject_id']) &
        (df['admittime'] > row['dischtime']) &
        (df['admittime'] <= row['dischtime'] + pd.Timedelta(days=30))
    ]
    return int(len(subsequent) > 0)

def map_admission_type(admission_type):
    if 'EMER.' in admission_type:
        return 'emergency'
    elif 'OBSERVATION' in admission_type:
        return 'observation'
    elif 'ELECTIVE' in admission_type:
        return 'elective'
    elif 'SURGICAL' in admission_type:
        return 'surgical'
    else:
        return 'other'

first_admissions['readmitted_30d'] = first_admissions.apply(was_readmitted, axis=1)

df = first_admissions.copy()

df['los_days'] = round((df['dischtime'] - df['admittime']).dt.total_seconds() / 86400, 2)
df['admission_type_consolidated'] = df['admission_type'].apply(map_admission_type)

diagnoses_counts = diagnoses.groupby('hadm_id')['icd_code'].size().reset_index().rename(columns={'icd_code': 'n_diagnoses'})
df = df.merge(diagnoses_counts, on='hadm_id', how='left')

df['n_diagnoses'] = df['n_diagnoses'].fillna(0)

ITEM_IDS = {
    50912: "creatinine",
    50931: "glucose",
    51222: "hemoglobin",
    50983: "sodium",
    50971: "potassium",
    51301: "wbc",
}

labs = lab_events[lab_events['itemid'].isin(ITEM_IDS.keys())]
labs['lab_name'] = labs['itemid'].map(ITEM_IDS)
labs["charttime"] = pd.to_datetime(labs["charttime"])

labs = (
    labs.sort_values("charttime")
    .groupby(["hadm_id", "lab_name"])["valuenum"]
    .last()
    .reset_index()
)

labs_pivot = labs.pivot_table(index='hadm_id', columns=['lab_name'], values='valuenum').reset_index()

df = df.merge(labs_pivot, on='hadm_id', how='left')

rx_counts = prescriptions.groupby('hadm_id')['drug'].nunique().reset_index().rename(columns={'drug': 'n_drugs'})
df = df.merge(rx_counts, on='hadm_id', how='left')

procedure_counts = procedures.groupby('subject_id')['icd_code'].nunique().reset_index().rename(columns={'icd_code': 'n_procedures'})
df = df.merge(procedure_counts, on='subject_id', how='left')

for lab in ['creatinine', 'glucose', 'wbc', 'hemoglobin', 'potassium', 'sodium']:
    df[f"{lab}_missing"] = df[lab].isna().astype(int)
    df[lab] = df[lab].fillna(df[lab].median())

for count in ['n_drugs', 'n_procedures', 'n_diagnoses']:
    df[count] = df[count].fillna(0)


encoder = OneHotEncoder(drop='first')

encoded = encoder.fit_transform(df[['gender', 'admission_type_consolidated']])
df[encoder.get_feature_names_out(['gender', 'admission_type_consolidated'])] = encoded.toarray()

df['log(los_days)'] = np.log1p(df['los_days'])
df['log(n_diagnoses)'] = np.log1p(df['n_diagnoses'])
df['log(n_procedures)'] = np.log1p(df['n_procedures'])
df['log(n_drugs)'] = np.log1p(df['n_drugs'])

features = df[[
        'log(los_days)', 'log(n_procedures)', 'log(n_drugs)', 'creatinine',
        'glucose', 'wbc', 'hemoglobin', 'sodium', 'potassium', 'gender_M', 'admission_type_consolidated_emergency',
        'admission_type_consolidated_observation', 'admission_type_consolidated_surgical',
        'admission_type_consolidated_other', 'age', 'log(n_diagnoses)', 'creatinine_missing', 'glucose_missing',
        'wbc_missing', 'hemoglobin_missing', 'sodium_missing', 'potassium_missing'
    ]]

labels = df[['subject_id', 'readmitted_30d']]

features.to_parquet('../../data/processed/features.parquet', index=False)
labels.to_parquet('../../data/processed/labels.parquet', index=False)
