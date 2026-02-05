import os
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
from synapseclient.models import Project
import tempfile
import pandas as pd

PEDS_PROJECT = "syn72418607"
ADULT_PROJECT = "syn72370534"
OVERALL_PROJECT = "syn72370534"
PEDS_FOLDER = "syn72493849"
ADULT_FOLDER = "syn72493850"

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="A simple script for ...")
    parser.add_argument('--start', default=0,type=int)
    parser.add_argument('--end',default=1000,type=int)
    parser.add_argument('--manifest_file', default='temp_manifest_file.tsv',type=str)

    args = parser.parse_args()
    
    sage_pat = os.getenv("SAGE_PAT")
    sage_project_id = OVERALL_PROJECT#ADULT_PROJECT if args.adult else PEDS_PROJECT

    syn = synapseclient.login(authToken=sage_pat)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv', delete=True) as temp:
        data = pd.read_csv(args.manifest_file, sep='\t')
        df = data[args.start:args.end]
        df.to_csv(temp.name, sep='\t', index=False)
        synapseutils.syncToSynapse(
            syn=syn, manifestFile=temp.name, sendMessages=False
        )


if __name__=='__main__':
    main()
