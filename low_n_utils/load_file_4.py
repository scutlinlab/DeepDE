import os
import sys
import warnings
import random
import copy
import pickle
import subprocess
from tqdm import tqdm
import re

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.autograd import Variable
from pathlib2 import Path

import torch
import torch.nn as nn
# from npy_data_loader import DistanceWindow
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from typing import Dict, Tuple, Union
from scipy import stats
from sklearn import metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/common')
#'/home/caiyi/github/low-N-protein-engineering-master/analysis/common'
import data_io_utils
import paths
import constants
import utils

sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/common')
from process_pdb import process_pdb
import deeppbs
import rep_utils
import self_embedding_tt as se
from process_pdb import process_pdb

sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A003_policy_optimization/')
#'/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization/'
import models
import A003_common

sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing')
#'/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing'
import A006_common
from unirep import babbler1900 as babbler

sys.path.append('/home/wangqihan/low-N-protein-engineering-master/泛化验证')
import net_MLP as net

'''改进版本
可以兼容Sarkisyan泛化数据集
进行了泛化测试，在输出S和P的文件和输出召回曲线的文件命名处均有改动'''


TRAINING_SET_FILE = paths.SARKISYAN_SPLIT_1_FILE ## use split 1
UNIREP_BATCH_SIZE = 400#3500
TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']
SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3

def config_list(task_name,task_path,train_num):
    task_list = os.listdir(task_path)
    if task_name == 'eUniRep':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-ET_Global_Init_1') and train_num in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    elif task_name == 'UniRep':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-ET_Global_Init_1') and '0096' in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    elif task_name == 'Random_UniRep':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-ET_Random_Init_1') and train_num in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    elif task_name == 'OneHot':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-OneHot') and train_num in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    elif task_name == 'ePtsRep':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-ET_Global_Init_1') and '0096' in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    elif task_name == 'PtsRep':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-ET_Global_Init_1') and '0096' in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    elif task_name == 'Random_PtsRep':
        config_list = []
        for name in task_list:
            if name.startswith('GFP_SA_config-ET_Global_Init_1') and '0096' in name:
                if int(name.split('-')[3][1]) < 5 and name.split('-')[4] != 'SparseRefit_False' and name.split('-')[4] != 'SmallTrust':
                    config_list.append(name)
    return config_list

def load_config(config_name,task_path,n_train_seqs):
    config_file = task_path + '/' + config_name
# config_file = str(sys.argv[1])
    print('Config file:', config_file)

    with open(config_file, 'rb') as f:
        config = pickle.load(f)
        
    seed = config['seed']
    if n_train_seqs == config['n_train_seqs']:
        n_train_seqs = config['n_train_seqs']
    model_name = config['model'] 
    n_chains = config['n_chains']
    T_max = config['T_max']
    sa_n_iter = config['sa_n_iter']
    temp_decay_rate = config['temp_decay_rate']
    min_mut_pos = config['min_mut_pos']
    max_mut_pos = config['max_mut_pos']
    nmut_threshold = config['nmut_threshold']
    output_file = config['output_file']
    return config,seed,n_train_seqs,model_name,n_chains,T_max,sa_n_iter,temp_decay_rate,min_mut_pos,max_mut_pos,nmut_threshold,output_file

def load_model(top_model_name,ebd_path_dict,seq_name_path,train_seqs,task_name):
    rep_list = []
    ebd_path = ebd_path_dict[task_name]
    with open(seq_name_path) as f:
        lines = f.readlines()
    d = {}
    for line in lines:
        line = line.split()
        d[line[0]] = line[1]
    for seq in train_seqs:
        rep_list.append(np.load(f'{ebd_path}/{d[seq]}.npy'))
    if task_name == 'eUniRep' or task_name == 'UniRep' or task_name == 'Random_UniRep' or task_name == 'OneHot':
        train_reps = np.array(rep_list)
    elif top_model_name != 'nn':
        train_reps = np.stack([np.mean(s, axis=0) for s in rep_list],0)
    elif top_model_name == 'nn':
        train_reps = np.array(rep_list)
    return train_reps

def load_model_25000(top_model_name,seed,ebd_path_dict,train_name_path,n_train_seqs,task_name):
    rep_list = []
    ebd_path = ebd_path_dict[task_name]
    # with open(seq_name_path) as f:
    # with open(train_name_path + '/set_train_' + str(seed) + '.txt') as f:
    with open(train_name_path + str(n_train_seqs) + '/set_train_' + str(seed) + '.txt') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()[0]
        rep_list.append(np.load(ebd_path + '/' + line))
    if task_name == 'eUniRep' or task_name == 'UniRep' or task_name == 'Random_UniRep' or task_name == 'OneHot':
        train_reps = np.array(rep_list)
    elif top_model_name != 'nn':
        train_reps = np.stack([np.mean(s, axis=0) for s in rep_list],0)
    elif top_model_name == 'nn':
        train_reps = np.array(rep_list)
    return train_reps

def select_train_seq(top_model_name,task_name,use_25000_train_seqs,real_value_path,train_name_path,seed,n_chains,n_train_seqs,ebd_path_dict,seq_name_path,
TRAINING_SET_FILE = TRAINING_SET_FILE,SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = SIM_ANNEAL_INIT_SEQ_MUT_RADIUS):
    if use_25000_train_seqs == 'True':
        with open(real_value_path) as f:
            lines = f.readlines()
        dic = {}
        for line in lines:
            dic[line.split()[0]] = float(line.split()[1])
        f.close()
        train_qfunc = []
        # with open(train_name_path) as f1:
        # with open(train_name_path + '/set_train_' + str(seed) + '.txt') as f1:
        with open(train_name_path + str(n_train_seqs) + '/set_train_' + str(seed) + '.txt') as f1:
            train_lines = f1.readlines()
        for train_line in train_lines:
            train_line = train_line.split('.')[0]
            real_value = float(dic[train_line])
            train_qfunc.append(real_value)
        train_qfunc = np.array(train_qfunc)
    else:
        print('Syncing data')
        np.random.seed(seed)
        random.seed(seed)
        print('Setting up training data')
        train_df = pd.read_csv(TRAINING_SET_FILE)#从csv文件里面抽取制定数量的训练集序列
        sub_train_df = train_df.sample(n=n_train_seqs)
        print(sub_train_df.head())

        train_seqs = list(sub_train_df['seq'])#训练集序列
        train_qfunc = np.array(sub_train_df['quantitative_function'])#训练集序列的真实值
    init_seqs = A006_common.propose_seqs(#模型设计的序列
            [constants.AVGFP_AA_SEQ]*n_chains, 
            [SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*n_chains, 
            min_pos=A006_common.GFP_LIB_REGION[0], 
            max_pos=A006_common.GFP_LIB_REGION[1])
    mu_muts_per_seq = 1.5*np.random.rand(n_chains) + 1
    if use_25000_train_seqs == 'True':
        train_reps = load_model_25000(top_model_name,seed,ebd_path_dict,train_name_path,n_train_seqs,task_name)
    else:
        train_reps = load_model(top_model_name,ebd_path_dict,seq_name_path,train_seqs,task_name)
    return train_reps,train_qfunc,init_seqs,mu_muts_per_seq

#Generate representations

def top_model(seed, task_name, top_model_name,config_1,train_reps,train_qfunc,TOP_MODEL_DO_SPARSE_REFIT,TOP_MODEL_ENSEMBLE_NMEMBERS = TOP_MODEL_ENSEMBLE_NMEMBERS,
TOP_MODEL_SUBSPACE_PROPORTION = TOP_MODEL_SUBSPACE_PROPORTION,TOP_MODEL_NORMALIZE = TOP_MODEL_NORMALIZE,
TOP_MODEL_PVAL_CUTOFF = TOP_MODEL_PVAL_CUTOFF):
    if top_model_name == 'lin':
        top_model = A003_common.train_ensembled_ridge(
                train_reps, 
                train_qfunc, 
                n_members=TOP_MODEL_ENSEMBLE_NMEMBERS, 
                subspace_proportion=TOP_MODEL_SUBSPACE_PROPORTION,
                normalize=TOP_MODEL_NORMALIZE, 
                do_sparse_refit=TOP_MODEL_DO_SPARSE_REFIT, 
                pval_cutoff=TOP_MODEL_PVAL_CUTOFF
            )
    elif top_model_name == 'nn':
        if task_name == 'eUniRep' or task_name == 'UniRep' or task_name == 'Random_UniRep' or task_name == 'OneHot':
            top_model = net.train_mlp_uni(seed, train_reps, train_qfunc, config_1)
        else:
            top_model = net.train_mlp(seed, train_reps, train_qfunc, config_1)
            # top_model = net.train_mlp_fh(seed, train_reps, train_qfunc, config_1)
    # top_model = A003_common.cv_train_ridge_with_sparse_refit(
    #     train_reps, 
    #     train_qfunc, 
    #     normalize=TOP_MODEL_NORMALIZE, 
    #     do_sparse_refit=TOP_MODEL_DO_SPARSE_REFIT, 
    #     pval_cutoff=TOP_MODEL_PVAL_CUTOFF
    # )
    return top_model

def get_predict_data(top_model_name,config_1,task_name, real_value_dict, test_list_dict, choose_ebd, seed, top_model, test_list_path, choose_test_data, test_data_separate):
    test_reps = []
    test_names = []
    d1 = {}
    d = {}
    if choose_test_data != 'ALL' or 'Global' or 'Global_24' or 'Global_96' or 'Random_24' or 'Random_96' or 'Onehot_24' or 'Onehot_96':
        real_value_path = real_value_dict[choose_test_data]
        ebd_path = choose_ebd[choose_test_data]
        path = ebd_path[task_name]
        test_list = test_list_dict[choose_test_data]
        with open(test_list) as f1:
            test_lines = f1.readlines()
            f1.close()
            
        for f in test_lines:
            f = f.split()[0]
            test_reps.append(np.load(f'{path}/{f}'))
            # test_reps.append(np.load(f'{path}/{f}')[5:-3])#为了保证SN数据集的长度一致临时改动
            test_names.append(f[:-4])
        print('len-test:', len(test_names))
        if task_name == 'eUniRep' or task_name == 'UniRep' or task_name == 'Random_UniRep' or task_name == 'OneHot':
            test_reps = np.array(test_reps)
        elif top_model_name != 'nn':
            test_reps = np.stack([np.mean(s, axis=0) for s in test_reps],0)
        elif top_model_name == 'nn':
            test_reps = np.array(test_reps)
        print('test_reps:',test_reps.shape)
        if top_model_name == 'nn':
            yhat = top_model.predict(test_reps, config_1)
        else:
            yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
        # yhat = top_model.predict(test_reps)
        for i, y in enumerate(yhat):
            d1[test_names[i]] = y
        with open(real_value_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0] in d1:
                group = 'GFP'
                if group not in d:
                    d[group] = [line]
                else:
                    d[group].append(line)
    
    else:
        ebd_path = choose_ebd['ALL']
        path = ebd_path[task_name]
        for f in os.listdir(path):
            test_reps.append(np.load(f'{path}/{f}'))
            test_names.append(f[:-4])
        if task_name == 'eUniRep' or task_name == 'UniRep' or task_name == 'Random_UniRep' or task_name == 'OneHot':
            test_reps = np.array(test_reps)
        else:
            test_reps = np.stack([np.mean(s, axis=0) for s in test_reps],0)
        print('test_reps:',test_reps.shape)
        yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
        # yhat = top_model.predict(test_reps)
        for i, y in enumerate(yhat):
            d1[test_names[i]] = y
        with open(test_list_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            if choose_test_data == 'ALL':
                if line[0].startswith('GFP_SimAnneal-'):
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Global_24':
                if line[0].startswith('GFP_SimAnneal-ET_Global') and '0024' in line[0]:
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Global':
                if line[0].startswith('GFP_SimAnneal-ET_Global'):
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Global_96':
                if line[0].startswith('GFP_SimAnneal-ET_Global') and '0096' in line[0]:
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Random_24':
                if line[0].startswith('GFP_SimAnneal-ET_Random') and '0024' in line[0]:
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Random_96':
                if line[0].startswith('GFP_SimAnneal-ET_Random') and '0096' in line[0]:
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Onehot_24':
                if line[0].startswith('GFP_SimAnneal-OneHot') and '0024' in line[0]:
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
            if choose_test_data == 'Onehot_96':
                if line[0].startswith('GFP_SimAnneal-OneHot') and '0096' in line[0]:
                    if test_data_separate == 'True':
                        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
                    else:
                        group = 'GFP'
                    if group not in d:
                        d[group] = [line]
                    else:
                        d[group].append(line)
    return yhat, d1,d

def output_index_25000(d1,d,data_save,num,task_name,n_train_seqs,recell_output_path):
    print('len_d_is:',len(d),len(d1))
    d_real = {k: [] for k in d}
    d_model = {k: [] for k in d}
    for g, group in d.items():
        for m in group:
            d_real[g].append(float(m[1]))
            d_model[g].append(d1[m[0]])
    if data_save == 'True':
        real = copy.copy(d_real['GFP'])
        predict = copy.copy(d_model['GFP'])
        print(len(real))
        print(len(predict))
        assert len(real) == len(predict)
        predict.extend(real)
        #print('len_data_list_is:', predict)
        data_array = np.array(predict)
        print('data_array_is:',data_array)
        name_list = []
        for i in range(len(real)):
            name_list.append(i)
        all_data = np.array([data_array,name_list])
        np.save(recell_output_path + str(task_name) + '_' + str(n_train_seqs) + '_34526_split' + '/gfp_20210407_T12__KNN_double_brightonly_4263-' + str(num) + '.npy', all_data)
    r_model = 0
    rho_model = 0
    acc_model = 0
    auc_model = 0
    mse_model = 0
    for g in d:
        print(len(d_model[g]), len(d_real[g]), len(d_real[g]))
        r_model += pearson(d_model[g], d_real[g])
        rho_model += spearman(d_model[g], d_real[g])
        a, b = classify(d_model[g], d_real[g], 1, 3.41076484873156)
        acc_model += a
        auc_model += b
    r_model /= len(d.keys())
    rho_model /= len(d.keys())
    acc_model /= len(d.keys())
    auc_model /= len(d.keys())
    mse_model /= len(d.keys())
    return r_model, rho_model, acc_model, auc_model, mse_model

def output_index(d1,d,data_save,test_data_separate,seed,task_name,n_train_seqs,choose_test_data,recell_output_path):
    print('len_d_is:',len(d),len(d1))
    #print(d.keys())
    d_real = {k: [] for k in d}
    d_unirep_400 = {k: [] for k in d}
    d_unirep = {k: [] for k in d}
    for g, group in d.items():
        for m in group:
            # if m[0].startswith('GFP_SimAnneal-'):
            #     if float(m[1]) > float(3.4):
            d_real[g].append(float(m[1]))
            # d_unirep[g].append(float(m[2]))
            d_unirep_400[g].append(d1[m[0]])
                
    if data_save == 'True' and test_data_separate == 'False':
        real = copy.copy(d_real['GFP'])
        predict = copy.copy(d_unirep_400['GFP'])
        print(len(real))
        print(len(predict))
        assert len(real) == len(predict)
        predict.extend(real)
        #print('len_data_list_is:', predict)
        data_array = np.array(predict)
        print('data_array_is:',data_array)
        name_list = []
        for i in range(len(real)):
            name_list.append(i)
        all_data = np.array([data_array,name_list])
        np.save(recell_output_path + str(task_name) + '_' + str(n_train_seqs) + '/gfp_20210407_T12__KNN_double_brightonly_4263-' + str(seed) + '.npy', all_data)# + '_' + choose_test_data

    r_unirep_400 = 0
    rho_unirep_400 = 0
    acc_unirep_400 = 0
    auc_unirep_400 = 0
    mse_unirep_400 = 0
    r_unirep = 0
    rho_unirep = 0
    acc_unirep = 0
    auc_unirep = 0
    mse_unirep = 0

    for g in d:
        print(len(d_unirep_400[g]), len(d_real[g]), len(d_real[g]))
        r_unirep_400 += pearson(d_unirep_400[g], d_real[g])
        rho_unirep_400 += spearman(d_unirep_400[g], d_real[g])
        a, b = classify(d_unirep_400[g], d_real[g], 1, 3.41076484873156)
        acc_unirep_400 += a
        auc_unirep_400 += b
        # r_unirep += pearson(d_unirep[g], d_real[g])
        # rho_unirep += spearman(d_unirep[g], d_real[g])
        # a, b = classify(d_unirep[g], d_real[g], 1, 3.41076484873156)
        # acc_unirep += a
        # auc_unirep += b

    r_unirep_400 /= len(d.keys())
    rho_unirep_400 /= len(d.keys())
    acc_unirep_400 /= len(d.keys())
    auc_unirep_400 /= len(d.keys())
    mse_unirep_400 /= len(d.keys())
    # r_unirep /= len(d.keys())
    # rho_unirep /= len(d.keys())
    # acc_unirep /= len(d.keys())
    # auc_unirep /= len(d.keys())
    # mse_unirep /= len(d.keys())
    return r_unirep, rho_unirep, acc_unirep, auc_unirep, mse_unirep, r_unirep_400, rho_unirep_400, acc_unirep_400, auc_unirep_400, mse_unirep_400

def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def spearman(a,b):
    return stats.spearmanr(a, b).correlation

def classify(a, b, thlda, thldb):#用于算结果时的分类
    pred = a
    tgt = b

    tp = fp = tn = fn = 1e-9
    pred_bin = []
    real_bin = []
    for i in range(len(tgt)):
        pred_i = pred[i]
        tgt_i = tgt[i]
        if pred_i > thlda and tgt_i > thldb:
            tp += 1
        if pred_i < thlda and tgt_i > thldb:
            fn += 1
        if pred_i > thlda and tgt_i < thldb:
            fp += 1
        if pred_i < thlda and tgt_i < thldb:
            tn += 1

        if pred_i >= thlda:
            pred_bin.append(1)
        else:
            pred_bin.append(0)
        if tgt_i >= thldb:
            real_bin.append(1)
        else:
            real_bin.append(0)
    print(tp, tn ,fp ,fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    acc_t = tp / (tp + fp)
    acc_f = tn / (tn + fn)
    recall_t = tp / (tp + fn)
    recall_f = tn / (tn + fp)
    fpr, tpr, _ = metrics.roc_curve(real_bin, pred_bin)
    auc = metrics.auc(fpr, tpr)
    return acc, auc

def data_save(use_25000_train_seqs,d1,d,data_save,test_data_separate,seed,output_path,task_name,n_train_seqs,choose_test_data,train_num,top_model_name,recell_output_path):
    if use_25000_train_seqs == 'True' and choose_test_data == '25000':
        r_model, rho_model, acc_model, auc_model, mse_model = output_index_25000(d1,d,data_save,seed,task_name,n_train_seqs,recell_output_path)
        with open(output_path + '/Low_N_test/' + str(task_name) + '/' + 'output_' + 'train_25000_' + task_name + '_' + str(n_train_seqs) + '_25000' + '.txt', 'a') as f: 
            f.write(f'{task_name}\t{r_model}\t{rho_model}\t{acc_model}\t{auc_model}\n')
    elif use_25000_train_seqs == 'False' and choose_test_data == '25000':
        r_model, rho_model, acc_model, auc_model, mse_model = output_index_25000(d1,d,data_save,seed,task_name,n_train_seqs,recell_output_path)
        with open(output_path + '/Low_N_test/' + str(task_name) + '/' + 'output_' + task_name + '_' + str(n_train_seqs) + '_25000' + '.txt', 'a') as f: 
            f.write(f'{task_name}\t{r_model}\t{rho_model}\t{acc_model}\t{auc_model}\n')
    elif use_25000_train_seqs == 'True' and choose_test_data != '25000':
        r_unirep, rho_unirep, acc_unirep, auc_unirep, mse_unirep, r_unirep_400, rho_unirep_400, acc_unirep_400, auc_unirep_400, mse_unirep_400 = output_index(d1,d,data_save,test_data_separate,seed,task_name,n_train_seqs,choose_test_data,recell_output_path)  
        with open(output_path + '/Low_N_test/' + str(task_name) + '/' + 'output_' + 'train_25000_' + task_name + '_' + str(n_train_seqs) + '_' + choose_test_data + '.txt', 'a') as f: 
            f.write(f'{task_name + train_num}\t{r_unirep_400}\t{rho_unirep_400}\t{acc_unirep_400}\t{auc_unirep_400}\t{r_unirep}\t{rho_unirep}\t{acc_unirep}\t{auc_unirep}\n')
    elif use_25000_train_seqs == 'False' and choose_test_data != '25000':
        r_unirep, rho_unirep, acc_unirep, auc_unirep, mse_unirep, r_unirep_400, rho_unirep_400, acc_unirep_400, auc_unirep_400, mse_unirep_400 = output_index(d1,d,data_save,test_data_separate,seed,task_name,n_train_seqs,choose_test_data,recell_output_path)  
        with open(output_path + '/Low_N_test/' + str(task_name) + '/' + 'output_' + task_name + '_' + str(n_train_seqs) + '_' + choose_test_data + '_' + top_model_name + '.txt', 'a') as f: #+ '_split' 
            f.write(f'{task_name + str(n_train_seqs)}\t{r_unirep_400}\t{rho_unirep_400}\t{acc_unirep_400}\t{auc_unirep_400}\t{r_unirep}\t{rho_unirep}\t{acc_unirep}\t{auc_unirep}\n')
    
