# data_loader.py
import pandas as pd
import re
from config import DATA_PATH, NROWS

def load_and_clean_data():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH, nrows=NROWS).fillna("")

    def parse_age(x):
        if pd.isna(x) or x == '': return None
        m = re.match(r'(\d+)-(\d+)', str(x))
        return (int(m.group(1)) + int(m.group(2))) // 2 if m else None

    def clean_id(x):
        return str(x).replace(',', '') if pd.notna(x) else ''

    def clean_deposit(x):
        try:
            return float(str(x).replace(',', ''))
        except:
            return None

    df['parsed_age'] = df['Age'].apply(parse_age)
    df['patientid'] = df['patientid'].apply(clean_id)
    df['parsed_deposit'] = df['Admission_Deposit'].apply(clean_deposit)
    df['stay_days'] = pd.to_numeric(df['Stay (in days)'], errors='coerce')

    # Mapping colonnes
    mapping = {
        'Department': 'department',
        'doctor_name': 'doctor_name',
        'health_conditions': 'health_conditions',
        'Insurance': 'insurance',
        'Type of Admission': 'admission_type',
        'Ward_Facility_Code': 'ward_code',
        'Severity of Illness': 'severity',
        'Visitors with Patient': 'visitor_count',
    }
    for old, new in mapping.items():
        if old in df.columns:
            df[new] = df[old]

    print(f"{len(df)} patients chargés et nettoyés")
    return df