from pathlib import Path # type: ignore
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler


CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}



CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 21
CHARPROTSET = { "A": 1, "C": 2,  "D": 3, "E": 4, "F": 5,"G": 6, 
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, 
                 "N": 12,"P": 13, "Q": 14, "R": 15, "S": 16,  
                 "T": 17, "V": 18, "W": 19, 
                 "Y": 20, "X": 21
              }

CHARPROTLEN = 21

def label_sequence(line, MAX_SEQ_LEN):   
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] 
    return X

    
    
class MyDataset_PDBBind2020(Dataset):
    def __init__(self, data_path, phase, name, max_seq_len, max_smi_len, scale_target, scale_inputs, scalers):
        data_path = Path(data_path)
        all_df = pd.read_csv(data_path / f"{name}_{phase}.csv", sep='\t')
        
        if scale_target:
            self.affinity =  scalers['-logKd/Ki'].transform(all_df[['-logKd/Ki']])
        else:    
            self.affinity = all_df['-logKd/Ki']
        
        self.smi =  all_df['smi_isomeric']  
        #self.smi =  all_df['smi_canonical']
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len
        self.prots = all_df['sequence_file']

        cols = ['MW', 'LogP','NumHDonors','NumHAcceptors','TPSA','NumRotatableBonds']
        if scale_inputs:
            for c in cols: #scalers:  In scaler exists and  '-logKd/Ki'
                 all_df[c]= scalers[c].transform(all_df[[c]])
        self.smile_desc = all_df[cols].to_numpy().tolist()


        self.length = len(self.smi)
        
    
    def __getitem__(self, idx):
        aug_smile = self.smi[idx]
        protseq   = self.prots[idx]

        return (

                label_smiles(aug_smile, self.max_smi_len),
                label_sequence(protseq, self.max_seq_len),
                np.array(self.smile_desc[idx], dtype=np.float32),
                np.array(self.affinity[idx], dtype=np.float32))

    def __len__(self):
        return self.length
    
    
def get_scalers_PDBBind2020(data_path, phase, name):    
    def get_scaler(df,col):
        return MinMaxScaler().fit(df[[col]])

    data_path = Path(data_path)
    df = pd.read_csv(data_path / f"{name}_{phase}.csv", sep='\t')
    data_scalers = {col: get_scaler(df, col)
                       for col in ['MW', 'LogP','NumHDonors','NumHAcceptors','TPSA','NumRotatableBonds', '-logKd/Ki']}
    return data_scalers




class MyDataset_LB_PDBBind2020(Dataset):
    def __init__(self, data_path, phase, name, max_seq_len, max_smi_len, scale_target, scale_inputs, scalers):
        data_path = Path(data_path)
        all_df = pd.read_csv(data_path / f"{name}_{phase}.csv")
        
        if scale_target:
            self.affinity =  scalers['value'].transform(all_df[['value']])
        else:    
            self.affinity = all_df['value']
        
        self.smi =  all_df['smiles']  
        #self.smi =  all_df['smi_canonical']
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len
        self.prots = all_df['seq']
        
        self.length = len(self.smi)
        
    
    def __getitem__(self, idx):
        aug_smile = self.smi[idx]
        protseq   = self.prots[idx]

        return (

                label_smiles(aug_smile, self.max_smi_len),
                label_sequence(protseq, self.max_seq_len),
                np.array(self.affinity[idx], dtype=np.float32), #dummy epistrefei to idio
                np.array(self.affinity[idx], dtype=np.float32))

    def __len__(self):
        return self.length
    
    
def get_scalers_LB_PDBBind2020(data_path, phase, name):    
    def get_scaler(df,col):
        return MinMaxScaler().fit(df[[col]])

    data_path = Path(data_path)
    df = pd.read_csv(data_path / f"{name}_{phase}.csv")
    data_scalers = {col: get_scaler(df, col)
                       for col in ['value']}
    return data_scalers





class MyDataset_pdbbind2016(Dataset):
    def __init__(self, data_path, phase, name, max_seq_len, max_smi_len, scale_target, scale_inputs, scalers):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / f"{name}_affinity_data.csv", sep='\t')
        
                
        if scale_target:  
            affinity_df['-logKd/Ki'] =  MinMaxScaler().fit_transform(affinity_df[['-logKd/Ki']])  


        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity
            
                 
        ligands_df = pd.read_csv(data_path / f"{name}_{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        self.pdbids = ligands_df['pdbid'].values
        self.max_seq_len = max_seq_len

        prot_df = pd.read_csv(data_path / f"{name}_{phase}_seq.csv")
        prots = {i["id"]: i["seq"] for _, i in prot_df.iterrows()}
        self.prots = prots

        self.length = len(self.smi)

    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]    
        aug_smile =   self.smi[pdbid]
        protseq = self.prots[pdbid]

        return (
                label_smiles(aug_smile, self.max_smi_len),
                label_sequence(protseq,self.max_seq_len),
                np.array(self.affinity[pdbid], dtype=np.float32), #dummy
                np.array(self.affinity[pdbid], dtype=np.float32))

    def __len__(self):
        return self.length
    
         
class MyDataset_davis_kiba(Dataset):
    def __init__(self, data_path, phase, name, max_seq_len, max_smi_len, scale_target, scale_inputs, scalers):
        data_path = Path(data_path)
        all_df = pd.read_csv(data_path / f"{name}_{phase}.csv")
        
        
        if scale_target:
            self.affinity =  scalers['affinity'].transform(all_df[['affinity']])
        else:    
            self.affinity = all_df['affinity']
        
        self.smi =  all_df['compound_iso_smiles']  
        #self.smi =  all_df['smi_canonical']
        self.max_smi_len = max_smi_len
        #self.pdbids = ligands_df['pdbid'].values
        self.max_seq_len = max_seq_len
        self.prots = all_df['target_sequence']


        self.length = len(self.smi)
        
    
    def __getitem__(self, idx):
        aug_smile = self.smi[idx]
        protseq   = self.prots[idx]

        return (
                label_smiles(aug_smile, self.max_smi_len),
                label_sequence(protseq, self.max_seq_len),
                np.array(self.affinity[idx], dtype=np.float32), #dummy
                np.array(self.affinity[idx], dtype=np.float32))

    def __len__(self):
        return self.length
    
def get_scalers_davis_kiba(data_path, phase, name):    
    def get_scaler(df,col):
        return MinMaxScaler().fit(df[[col]])

    data_path = Path(data_path)
    df = pd.read_csv(data_path / f"{name}_{phase}.csv")
    data_scalers = {col: get_scaler(df, col)
                       for col in  ['affinity']}
    return data_scalers
