# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:34:12 2024

@author: JSR
"""

from sqlalchemy import create_engine, text
import pandas as pd

# Global configuration file path
CNF_FILE_PATH = "config.cnf"

def parse_cnf_file(cnf_file_name):
    """
    Parses a configuration file in INI format and returns a dictionary.
    
    Args:
        cnf_file_name (str): Path to the configuration file.
    
    Returns:
        dict: A dictionary containing the parsed configuration.
    """
    config_dict = {}
    current_section = None

    with open(cnf_file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:  # Skip comments and empty lines
                continue

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip()
                config_dict[current_section] = {}
            elif '=' in line and current_section is not None:
                key, value = line.split('=', 1)
                config_dict[current_section][key.strip()] = value.strip()  # Remove spaces

    return config_dict


def read_query(query):
    """
    Executes a SQL query using configuration parameters from a file and returns the results as a DataFrame
    
    Args:
        query (str): The SQL query to execute.
    
    Returns:
        pd.DataFrame: DataFrame containing the query results.
    """
    sc = parse_cnf_file(CNF_FILE_PATH)

    try:
        host = sc['local']['host']
        username = sc['local']['username'] 
        password = sc['local']['password']
    except KeyError as e:
        raise KeyError(f"Configuration error: {e} is missing in the config file.")

    # Create a database connection
    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}")
    
    with engine.connect() as connection:
        result = connection.execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
    return df