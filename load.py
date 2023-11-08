import subprocess
import sys
import pandas as pd
import json

from tqdm import tqdm
from modules import GithubClient, HuggingFaceClient
import esprima
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

"""
Database Stuff Start
"""

env = load_dotenv(dotenv_path='.env')
db_username = env["db_username"]
db_password = env["db_password"]
db_host = env["db_host"]
db_port = env["db_port"]
db_name = env["db_name"]
connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

Session = sessionmaker(bind=engine)
session = Session()

def session_commit():
    session.commit()

def session_add(row):
    session.add(row)

Base = declarative_base()

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

"""
Helper functions End
"""    


"""
Main
"""
DS_LENGTH = 1000
BATCH_SIZE = 100
def main():
    print('Starting to Load Hugging Face Data..')
    hfc = HuggingFaceClient(HUGGINGFACE_TOKEN)
    dataset_index = 0
    for i in tqdm(range(dataset_index, DS_LENGTH)):
        params = {
            "dataset":"code_search_net",
            "config":"javascript",
            "split":"train",
            'offset': dataset_index,
            'length': BATCH_SIZE
        }
        res_ds = hfc.get('rows', params=params)
        if res_ds.status_code != 200:
            raise Exception(f"Hugging Face client error:\n{res_ds.json()}")
        dataset = res_ds.json()
        table_rows = getRows(dataset)
        for row in table_rows:
            writeJS(f"samples/_{row['dataset_id']}.js", row['func_code_string'])
        
        run_eslint(i)
        if not os.path.exists(f"eslint_batch_{i}.json"):
            raise Exception(f"Reading JS binaries failed.")
        store_dataset(f"eslint_batch_{i}.json")
        
        dataset_index += BATCH_SIZE
    
if __name__ == '__main__':
    main()
    