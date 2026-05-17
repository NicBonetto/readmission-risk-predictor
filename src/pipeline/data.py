import duckdb
import pandas as pd 
from pathlib import Path

DATA_RAW = Path(__file__).parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).parents[2] / "data" / "processed"

def load_mimic_dataset(conn, data_dir=DATA_RAW):
    tables = [
        "admissions",
        "patients",
        "diagnoses_icd",
        "labevents",
        "prescriptions",
        "d_icd_procedures",
        "procedures_icd"
    ]

    for table in tables:
        csv_path = data_dir / "mimic-iv-clinical-database-demo-2.2" / "hosp" / f"{table}.csv"

        if csv_path.exists():
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table} AS 
                SELECT * FROM read_csv_auto('{csv_path}')
           """ )
            print(f"Registered: {table}")
        else:
            print(f"Missing: {table}")

if __name__ == "__main__":
    conn = duckdb.connect('../../data/duckdb.mimic')
    print("Loading MIMIC-IV tables...")
    load_mimic_dataset(conn)
