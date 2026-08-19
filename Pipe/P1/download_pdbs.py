from __future__ import annotations
import numpy as np
import os
import time 
import pandas as pd
import sys
import requests
import concurrent.futures
import multiprocessing
import h5py
import molearn
import torch
import glob as glob
import MDAnalysis.analysis.rms as rms
import seaborn as sns
import pickle
import networkx as nx
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import tree



def download2(code, pdir=None, max_retries=3):
    """Download a PDB file with retry mechanism and failure handling"""
    base_url = "https://files.rcsb.org/download"
    code = str(code).strip().upper()
    pdb_url = f"{base_url}/{code}.pdb"
    f_p = os.path.join(pdir, f"{code}.pdb")

    for attempt in range(max_retries):
        try:
            response = requests.get(pdb_url, stream=True, timeout=10)
            if response.status_code == 404:
                print(f"{code} does not exist (404 Not Found)")
                return None
            response.raise_for_status()
            
            with open(f_p, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Check file size to prevent empty files
            if os.path.getsize(f_p) == 0:
                print(f"{code}.pdb download failed (empty file), retrying {attempt+1}/{max_retries}...")
                os.remove(f_p)
                continue  # Retry

            print(f"{code}.pdb downloaded successfully")
            return f_p
        except requests.exceptions.RequestException as e:
            print(f"{code}.pdb download failed, retrying {attempt+1}/{max_retries}... Error: {e}")
            time.sleep(2)

    print(f"{code}.pdb download ultimately failed")
    return None


def download_pdbs(pdb_list, pdir=None):
    """Download multiple PDB files"""
    default_dir = "./PDBs"
    pdir = os.path.abspath(pdir if pdir else default_dir)
    os.makedirs(pdir, exist_ok=True)

    # Get already downloaded PDB files to avoid duplicate downloads
    existing_files = {os.path.splitext(f)[0] for f in os.listdir(pdir)}

    for code in pdb_list:
        if code not in existing_files:
            file_path = download2(code, pdir=pdir)
            if file_path:
                print(f"{code} downloaded successfully")
            else:
                print(f"{code} download failed")


def parallel_download(pdb_list, pdir=None):
    """Download PDB files in parallel"""
    num_workers = min(20, multiprocessing.cpu_count()*2)  # Limit the number of threads
    chunk_size = max(10, len(pdb_list) // num_workers)  # Each thread handles at least 10 PDB files
    splited_pdb_lists = [pdb_list[i:i+chunk_size] for i in range(0, len(pdb_list), chunk_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda sublist: download_pdbs(sublist, pdir=pdir), splited_pdb_lists)
            

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structsCSV", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    
    structure_path = args.structsCSV
    pdb_data = pd.read_csv(structure_path, sep = "\t", header=0, engine='python')
    pdb_data['Accession'] = pdb_data['Accession'].str.upper()

    pdbs_ids = pdb_data['Accession'].tolist()
    parallel_download(pdbs_ids,args.out)

    folder_path = args.out
    file_names = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    pdb_raw = pd.DataFrame({"PDBs": file_names})

    pdb_data['Downloaded'] = pdb_data['Accession'].str.upper().isin(pdb_raw['PDBs']).map({True: True, False: False})

    counts = pdb_data['Downloaded'].value_counts().to_dict()
    print(f"Downloaded: {counts[True]}, Failed: {counts[False]}")

    fail_list = pdb_data[pdb_data['Downloaded']==False]
    fail_list.to_csv('fail_list.csv')

if __name__ == "__main__":
    main()
