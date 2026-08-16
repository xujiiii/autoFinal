
import numpy as np
import os
import glob 
import time 
import pandas as pd
import shutil
import re
import mdtraj as md
import MDAnalysis as mda
import pickle
import pickle as p 
import csv
import sys
import subprocess
import xml.etree.ElementTree as ET
import requests
import concurrent.futures
import multiprocessing
import nglview as nv
import h5py
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import html
import biobox as bb
import tempfile
import matplotlib.pyplot as plt
import molearn
import torch
import glob as glob
import MDAnalysis.analysis.rms as rms
import seaborn as sns
import pickle
import networkx as nx
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))

from Bio import PDB
from tqdm import tqdm
from time import time as t
from urllib.request import urlretrieve as download
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Blast.Applications import NcbipsiblastCommandline
from collections import defaultdict
from Bio.PDB import PPBuilder
from Bio.SeqUtils import seq1
from modeller import *
from modeller.automodel import *
from Bio.PDB import PDBParser
from MDAnalysis.analysis import align
from mpi4py import MPI
from glob import glob
from pprint import pprint as pp
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from glob import glob as g
from tqdm.notebook import tqdm
from collections import Counter
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
from pymol import cmd
from scipy.interpolate import interp1d
from mpl_toolkits import mplot3d
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from matplotlib.colors import BoundaryNorm
from numpy.linalg import norm
from molearn.data import PDBData
from molearn.trainers import Trainer
from molearn.models.small_foldingnet import Small_AutoEncoder
from molearn.analysis.analyser import MolearnAnalysis
from copy import deepcopy
from molearn.analysis import MolearnGUI
from scipy.ndimage import rotate
from numpy.linalg import inv
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from itertools import combinations
from urllib.request import urlretrieve
from urllib.error import URLError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz
from sklearn import tree
from scipy.stats import pearsonr
from numpy.linalg import norm
from molearn.data import PDBData
from molearn.trainers import Trainer
from molearn.models.small_foldingnet import Small_AutoEncoder
from molearn.analysis.analyser import MolearnAnalysis

from sklearn import tree
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trainidx", required=True, type=Path)
    ap.add_argument("--testidx", required=True, type=Path)
    ap.add_argument("--combined-pdb", required=True, type=Path)
    args = ap.parse_args()
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import glob
    
    # file_pattern = '/data/student/xujia/autoFinal/output_data/f3_out/best.ckpt'

    file_pattern=args.checkpoint

    

    matching_files = sorted(glob.glob(str(file_pattern)))

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No files matched the pattern: {file_pattern}")

    networkfile = matching_files[0]

    checkpoint = torch.load(networkfile, map_location=torch.device('cpu'),weights_only=False)
    net = Small_AutoEncoder(**checkpoint['network_kwargs'])
    net.load_state_dict(checkpoint['model_state_dict'])

    print("Matched files:", matching_files)
    print("Using file:", networkfile)
    print("Network kwargs:", checkpoint['network_kwargs'])

    import os
    data = PDBData()
    # folder_name = 'Results/fitted_matlab_segments/mustang_endsAlignment_cleaned_noOutliers'
    # #folder_name="Results/fitted_matlab_segments/foldseek_endsAlignment_cleaned_noOutliers_noPCAoutliers"
    # combined_file_path = os.path.join(folder_name, 'combined.pdb')
    
    data.import_pdb(filename=args.combined_pdb)
    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    data.prepare_dataset()



    from copy import deepcopy
    MA = MolearnAnalysis()
    MA.set_network(net)

    # checkpoint_suffix = 'cleaned_noOutlier_11_checkpoint_newsplit'
    # train_file = f"train_model_indices_{checkpoint_suffix}.txt"
    # test_file = f"test_model_indices_{checkpoint_suffix}.txt"

    train_file = args.trainidx
    test_file = args.testidx

    train_indices = np.loadtxt(train_file, dtype=int)
    test_indices = np.loadtxt(test_file, dtype=int)

    print("Train indices loaded:", train_indices.shape)
    print("Test indices loaded:", test_indices.shape)
    print(test_indices)

    data_train = deepcopy(data)
    data_test = deepcopy(data)
    data_train.dataset = data.dataset[train_indices]
    data_test.dataset = data.dataset[test_indices]
    data_train.indices = train_indices
    data_test.indices = test_indices

    MA.set_dataset("training", data_train)
    MA.set_dataset("test", data_test)

    def ifnotmake(dir_path):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        return dir_path

    print(sorted(np.append(train_indices, test_indices)))

    ifnotmake(out/'Results/run_trial_BRAFActivationLoop_postalign_cleaned_noOutlier_11_checkpoint_newsplit/getDatasetTrial_train/')
    for i, index in enumerate(train_indices):
        print(index)
        data._mol.set_current(index)  # Switch to the frame at the given index
        pdb_lines = data._mol.get_pdb_data() 
        with open(out/f'Results/run_trial_BRAFActivationLoop_postalign_cleaned_noOutlier_11_checkpoint_newsplit/getDatasetTrial_train/s{i}.pdb', 'w') as f:
            for line in pdb_lines:
                if isinstance(line, list) and line[0] == 'ATOM':
                    atom_serial = line[1]
                    atom_name = line[2]
                    res_name = line[3]
                    chain_id = line[4]
                    res_seq = line[5]
                    x, y, z = line[6:9]
                    occupancy = line[10]
                    b_factor = line[9]
                    element = line[11]

                    pdb_line = f"{'ATOM':<6}{atom_serial:>5} {atom_name:^4}{res_name:>4} {chain_id}{res_seq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element:>2}"
                    f.write(pdb_line + '\n')
        

        
    ifnotmake(out/'Results/run_trial_BRAFActivationLoop_postalign_cleaned_noOutlier_11_checkpoint_newsplit/getDatasetTrial_test/')
    for i, index in enumerate(test_indices):
        print(index)
        data._mol.set_current(index) 
        pdb_lines = data._mol.get_pdb_data() 
        with open(out/f'Results/run_trial_BRAFActivationLoop_postalign_cleaned_noOutlier_11_checkpoint_newsplit/getDatasetTrial_test/s{i}.pdb', 'w') as f:
            for line in pdb_lines:
                if isinstance(line, list) and line[0] == 'ATOM':
                    atom_serial = line[1]
                    atom_name = line[2]
                    res_name = line[3]
                    chain_id = line[4]
                    res_seq = line[5]
                    x, y, z = line[6:9]
                    occupancy = line[10]
                    b_factor = line[9]
                    element = line[11]

                    pdb_line = f"{'ATOM':<6}{atom_serial:>5} {atom_name:^4}{res_name:>4} {chain_id}{res_seq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element:>2}"
                    f.write(pdb_line + '\n')
                    
                    
    MA.batch_size = 8
    MA.processes = 4


    import pandas as pd

    saveName = '_foldingnet_checkpoint'


    err_train = MA.get_error('training')
    df = pd.DataFrame(err_train, columns=['training_error'])
    # train index 1 means the error belongs to the pdb with model index equals to the index recorded in train_idx.txt at position 1(start by 0)
    df.to_csv(out/'err_train.csv', index_label='train_index')

    err_test = MA.get_error('test')
    df = pd.DataFrame(err_test, columns=['training_error'])
    df.to_csv(out/'err_test.csv', index_label='train_index')
    
    data = [err_train]
    f = plt.figure(figsize=(10, 10))
    sns.violinplot(data)
    q3 = np.percentile(err_train, 75)
    plt.axhline(y=q3, color='r', linestyle='--', linewidth=1.5, label=f'Q3 ({q3:.2f})')
    plt.legend(loc='upper right', fontsize=20)
    plt.ylabel('RMSD [$\AA$]',fontsize=20)
    plt.title('Reconstruction error between encoded and decoded training dataset',fontsize=20)
    plt.show()
    f.savefig(out/f'err_train_{saveName}.pdf')

    data = [err_test]
    f = plt.figure(figsize=(10, 10))
    sns.violinplot(data)
    q3 = np.percentile(err_test, 75)
    plt.axhline(y=q3, color='r', linestyle='--', linewidth=1.5, label=f'Q3 ({q3:.2f})')
    plt.legend(loc='upper right', fontsize=20)
    plt.ylabel('RMSD [$\AA$]',fontsize=20)
    plt.title('Reconstruction error between encoded and decoded test dataset',fontsize=20)
    plt.show()
    f.savefig(out/f'err_test_{saveName}.pdf')




if __name__ == "__main__":
    main()