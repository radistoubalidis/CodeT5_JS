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
class BadSamples(Base):
    __tablename__ = 'bad_samples'

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

class GoodSamples(Base):
    __tablename__ = 'good_samples'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    csn_sample_id = Column(Integer)
    source = Column(Text)

Base.metadata.create_all(engine)


def store_bad_samples(eslint_output):
    eslint_df = pd.DataFrame(eslint_output)

    eslint_df['sample_id'] = [x.split('/')[-1].split('.')[0][1:] for x in eslint_df['filePath']]
    
    table_rows = []
    rows_inserted = 0
    for idx, row in eslint_df.iterrows():
        for msg in row['messages']:
            if len(msg) > 0:
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
        new_ds = BadSamples(
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

def store_good_samples(filepaths):
    for path in filepaths:
        file = open(path,'r')
        contents = file.read()
        new_sample = GoodSamples(
            csn_sample_id=path.split('/')[-1].split('.')[0][1:],
            source=contents
        )
        session_add(new_sample)
    session_commit()    
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

def remove_batch_files():
    if os.path.exists('samples'):
        for file in os.listdir('samples'):
            os.remove(f"{os.getcwd()}/samples/{file}")

def test():
    print(DS_LENGTH, DS_LENGTH, BATCH_SIZE)

class SampleClassifier():
    def read_eslint(batch_idx):
        with open(f"eslint_outputs/eslint_batch_{batch_idx}.json") as f:
            eslint_out = json.load(f)
        return eslint_out
    
    def files_with_no_errors(eslint_out):
        eslint_df = pd.DataFrame(eslint_out)
        return eslint_df.loc[eslint_df['errorCount'] == 0]['filePath'].unique()
    

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
    bad_samples_added = 0
    good_samples_added = 0
    batched_len = DS_LENGTH // BATCH_SIZE
    pbar = tqdm(np.arange(dataset_index, batched_len))
    for i in pbar:
        # Get data
        params = {
            "dataset":"code_search_net",
            "config":"javascript",
            "split":"train",
            'offset': dataset_index,
            'length': BATCH_SIZE
        }
        res_ds = hfc.get('rows', params=params)
        
        # Handle http errors
        if res_ds.status_code != 200:
            raise Exception(f"Hugging Face client error:\n{res_ds.json()}")
        dataset = res_ds.json()
        
        # prepare data and create js files
        table_rows = getRows(dataset)
        for row in table_rows:
            writeJS(f"{SAMPLES_DIR}/_{row['dataset_id']}.js", row['func_code_string'])
        
        # Run eslint on js files
        dataset_index += BATCH_SIZE
        run_eslint(i)
        
        # Handle eslint error
        if not os.path.exists(f"{ESLINT_OUT_DIR}/eslint_batch_{i}.json"):
            raise Exception(f"Reading JS binaries failed.")
        
        # get good samples
        eslint_out = SampleClassifier.read_eslint(i)
        good_samples_files = SampleClassifier.files_with_no_errors(eslint_out)
        store_good_samples(good_samples_files)
        bad_samples_added += store_bad_samples(eslint_out)
        good_samples_added += len(good_samples_files)
        
        remove_batch_files()
        pbar.set_postfix({'num_errors_warnings': bad_samples_added, 'num_good_files':good_samples_added})
        

if __name__ == '__main__':
    args = getArgs()
    BATCH_SIZE = int(args.batch_size)
    DS_LENGTH = int(args.dataset_length)
    DS_OFFSET = int(args.dataset_offset)
    SAMPLES_DIR = args.samples_dir
    ESLINT_OUT_DIR = args.eslint_outputs_dir
    main()
    