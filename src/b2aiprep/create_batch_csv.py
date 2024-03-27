import csv
from glob import glob

def create_batch_csv(input_dir, out_file):
    
    #input_dir is the top level directory of the b2ai Production directory from Wasabi
    #it is expected to contain subfolders with each institution e.g. MIT, UCF, etc.
    
    #out_file is where a csv file will be saved and should be in the format 'path/name/csv'
    
    audiofiles = glob(f'{input_dir}/*/*.wav')

    with open(out_file, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        for item in audiofiles:
            write.writerow([item])
    
    return out_file