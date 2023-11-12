import argparse
from ast import arg
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from modules import HuggingFaceClient
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

"""
Database Stuff Start
"""

env = load_dotenv(dotenv_path='.env')
db_username = os.getenv("db_username")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")
connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

Session = sessionmaker(bind=engine)
session = Session()

def session_commit():
    session.commit()

def session_add(row):
    session.add(row)

Base = declarative_base()

# Dataset ORM
class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    csn_sample_id = Column(Integer)
    rule_id = Column(String)
    source = Column(Text)
    error_count = Column(Integer)
    fatal_error_count = Column(Integer)
    warning_count = Column(Integer)
    fixable_error_count = Column(Integer)
    fixable_warning_count = Column(Integer)
    message = Column(Text)
    severity = Column(Integer)
    line = Column(Integer)

Base.metadata.create_all(engine)


def store_dataset(eslint_out):
    with open(eslint_out,'r') as f:
        eslint_output = json.load(f)

    eslint_df = pd.DataFrame(eslint_output)

    eslint_df['sample_id'] = [x.split('/')[-1].split('.')[0][1:] for x in eslint_df['filePath']]
    
    table_rows = []
    rows_inserted = 0
    for idx, row in eslint_df.iterrows():
        for msg in row['messages']:
            row_dict = {
                'csn_sample_id' : row['sample_id'],
                'rule_id' : msg['ruleId'],
                'source' : row['source'],
                'errorCount' : row['errorCount'],
                'fatalErrorCount' : row['fatalErrorCount'],
                'warningCount' : row['warningCount'],
                'fixableErrorCount' : row['fixableErrorCount'],
                'fixableWarningCount' : row['fixableWarningCount'],
                'message' : msg['message'],
                'severity' : msg['severity'],
                'line' : msg['line'],
            }
            table_rows.append(row_dict)
            
    for row in table_rows:
        new_ds = Dataset(
            csn_sample_id=row['csn_sample_id'],
            rule_id=row['rule_id'] if row['rule_id'] != None else 'js-parser',
            source=row['source'],
            error_count=row['errorCount'],
            fatal_error_count=row['fatalErrorCount'],
            warning_count=row['warningCount'],
            fixable_error_count=row['fixableErrorCount'],
            fixable_warning_count=row['fixableWarningCount'],
            message=row['message'],
            severity=row['severity'],
            line=row['line'],
        )
        session_add(new_ds)
        rows_inserted += 1
    session_commit()
    return rows_inserted
"""
Database Stuff End
"""


"""
API keys
"""
with open('.github-api-token.txt','r') as f:
    GITHUB_TOKEN = f.read()

with open('.hugging-face-token.txt','r') as f:
    HUGGINGFACE_TOKEN = f.read()



"""
Helper functions Start 
"""
def run_eslint(batch_id):
    os.system(f"{os.getcwd()}/eslint.sh {batch_id}")

def getRows(dataset):
    table_rows = []
    for row in dataset['rows']:
        row_data = row['row']
        row_data['dataset_id'] = row['row_idx']
        table_rows.append(row_data)
    return table_rows

def writeJS(path, content):
    try:
        with open(path,'w') as f:
            f.write(content)
    except Exception as e:
        raise e

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size')
    parser.add_argument('--dataset_length')
    parser.add_argument('--dataset_offset')
    parser.add_argument('--samples_dir')
    parser.add_argument('--eslint_outputs_dir')
    args = parser.parse_args()
    return args

def test():
    print(DS_LENGTH, DS_LENGTH, BATCH_SIZE)

"""
Helper functions End
"""    


"""
Main
"""
def main():
    print('Starting to Load Hugging Face Data..')
    hfc = HuggingFaceClient(HUGGINGFACE_TOKEN)
    dataset_index = DS_OFFSET
    rows_added = 0
    pbar = tqdm(np.arange(dataset_index, DS_LENGTH))
    for i in pbar:
        # params = {
        #     "dataset":"code_search_net",
        #     "config":"javascript",
        #     "split":"train",
        #     'offset': dataset_index,
        #     'length': BATCH_SIZE
        # }
        # res_ds = hfc.get('rows', params=params)
        # if res_ds.status_code != 200:
        #     raise Exception(f"Hugging Face client error:\n{res_ds.json()}")
        # dataset = res_ds.json()
        with open('dataset_batch.json','r') as f:
            dataset = json.load(f)
        table_rows = getRows(dataset)
        for row in table_rows:
            writeJS(f"{SAMPLES_DIR}/_{row['dataset_id']}.js", row['func_code_string'])
        
        pbar.set_postfix({'num_errors_warnings': len(table_rows)})
        dataset_index += BATCH_SIZE
        
    idx = 1
    run_eslint(idx)
    if not os.path.exists(f"{ESLINT_OUT_DIR}/eslint_batch_{idx}.json"):
        raise Exception(f"Reading JS binaries failed.")
    rows_added += store_dataset(f"{ESLINT_OUT_DIR}/eslint_batch_{idx}.json")
    
if __name__ == '__main__':
    args = getArgs()
    BATCH_SIZE = int(args.batch_size)
    DS_LENGTH = int(args.dataset_length)
    DS_OFFSET = int(args.dataset_offset)
    SAMPLES_DIR = args.samples_dir
    ESLINT_OUT_DIR = args.eslint_outputs_dir
    main()
    