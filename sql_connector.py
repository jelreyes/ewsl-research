# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:34:12 2024

@author: JSR
"""

from sqlalchemy import create_engine, text
import pandas as pd

def parse_cnf_file(cnf_file_name):
    config_dict = {}
    current_section = None

    with open(cnf_file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip() 
                config_dict[current_section] = {}
            else:
                if '=' in line:
                    key, value = line.split('=', 1)
                    if current_section:
                        config_dict[current_section][key.strip()] = value.strip()  # Remove spaces

    return config_dict

def read_query(query, engine):
    with engine.connect() as connection:
        result = connection.execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

if __name__ == "__main__":
    cnf_file_path = "config.cnf"
    sc = parse_cnf_file(cnf_file_path)
    try:
        host = sc['local']['host']
        username = sc['local']['username'] 
        password = sc['local']['password']
    except KeyError:
        print("Unknown Host")

    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}")
    
    test_query = "SELECT * FROM analysis_db.rain_sat_agb"
    df = read_query(test_query, engine)
    print(df)

    