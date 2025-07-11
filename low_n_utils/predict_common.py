import os
import pickle
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/wangqihan/Low-N-improvement/function")
import config 


def load_rep_by_seq(rep_path, seq_list, model_name, top_model_name):
    rep_list = []
    if 'Alphafold' in model_name:
        for seq in seq_list:
            with open(f'{rep_path}/{d[seq]}.pkl', "rb") as f:
                dick = pickle.load(f)
                rep_list.append(dick["pair"])
    else:
        for seq in seq_list:
            rep_list.append(np.load(f'{rep_path}/{d[seq]}.npy'))
    if 'UniRep' in model_name or model_name == 'OneHot':
        rep_array = np.stack(rep_list, 0)
    elif 'PtsRep' in model_name and top_model_name == 'lin':
        rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
    elif 'PtsRep' in model_name and top_model_name == 'nn':
        rep_array = np.stack(rep_list, 0)
    elif 'Alphafold' in model_name and top_model_name == 'lin':
        rep_array = np.stack([np.mean(np.mean(s, axis=0), axis=0) for s in rep_list], 0)
    elif 'Alphafold' in model_name and top_model_name == 'nn':
        rep_array = np.stack([np.mean(s, axis=0) for s in rep_list], 0)
    return rep_array



def load_rep_by_name(rep_path, name_list, model_name, top_model_name):
    rep_list = []
    if 'Alphafold' in model_name:
        for name in name_list:
            with open(f'{rep_path}/{name}.pkl', "rb") as f:
                dick = pickle.load(f)
                rep_list.append(dick['representations'][config.ALPHAFOLD2["target"]])
    elif 'Openfold' in model_name:
        for name in name_list:
            with open(f'{rep_path}/{name}.pkl', "rb") as f:
                dick = pickle.load(f)
                rep_list.append(dick[config.OPENFOLD["target"]])
    else:        
        for name in name_list:
            rep_list.append(np.load(f'{rep_path}/{name}.npy'))
    if 'UniRep' in model_name or model_name == 'OneHot':
        rep_array = np.stack(rep_list, 0)
    elif 'PtsRep' in model_name and top_model_name == 'lin':
        rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
    elif 'PtsRep' in model_name and top_model_name == 'nn':
        rep_array = np.stack(rep_list, 0)
        # rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
    elif 'Alphafold' in model_name and top_model_name == 'lin':
        rep_array = np.stack([np.mean(np.mean(s, axis=0), axis=0) for s in rep_list], 0)
    elif 'Alphafold' in model_name and top_model_name == 'nn':
        rep_array = np.stack([np.mean(s, axis=0) for s in rep_list], 0)
    elif 'Openfold' in model_name and top_model_name == 'lin':
        rep_array = np.stack([np.mean(np.mean(s, axis=0), axis=0) for s in rep_list], 0)
    elif 'Openfold' in model_name and top_model_name == 'nn':
        rep_array = np.stack([np.mean(s, axis=0) for s in rep_list], 0)
    return rep_array

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

class DataframeOperation:
    def __init__(self, data_frame_1, data_frame_2):
        self.data_frame_1 = data_frame_1
        self.data_frame_2 = data_frame_2
        self.column_1 = self.data_frame_1.columns.tolist()
        self.column_2 = self.data_frame_2.columns.tolist()
        if len(self.column_1) >= len(self.column_2):
            self.column_long = self.column_1
            self.column_short = self.column_2
        else:
            self.column_long = self.column_2
            self.column_short = self.column_1
        for i in self.column_short:
            assert i in self.column_long

    def difference_set(self):
        df_3 = self.data_frame_1.append(self.data_frame_2)
        df_3 = df_3.drop_duplicates(subset=self.column_short, keep=False)
        return df_3

    def intersection(self):
        df_3 = pd.merge(self.data_frame_1, self.data_frame_2, on=self.column_short)
        return df_3

    def union(self):
        df_3 = pd.merge(self.data_frame_1, self.data_frame_2, on=self.column_short, how='outer')
        return df_3