import os
import sys
import warnings
import random
import copy
import pickle
import subprocess

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from scipy import stats
from sklearn import metrics


# sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis//common')
from low_n_utils import data_io_utils, paths, constants, low_n_utils

# sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis//A003_policy_optimization/')
from low_n_utils import models, A003_common

# sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing')
from low_n_utils import A006_common
from low_n_utils.unirep import babbler1900 as babbler

def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def spearman(a,b):
    return stats.spearmanr(a, b).correlation


def classify(a, b, thlda, thldb):
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
    # return {'acc': acc, 'acc_t': acc_t, 'acc_f': acc_f, 
    #         'recall_t': recall_t, 'recall_f': recall_f, 'auc': auc}

### CONFIGURATION ###
# Load config
config_file = str(sys.argv[1])
print('Config file:', config_file)

with open(config_file, 'rb') as f:
    config = pickle.load(f)
    
seed = config['seed']
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

for k in config:
    if k == 'init_seqs':
        print(k + ':', config[k][:5])
    else:
        print(k + ':', config[k])

## do few iterations to debug
#sa_n_iter = 3
#print('WARNING: doing a debugging number of iterations')


# Hard constants
TRAINING_SET_FILE = paths.SARKISYAN_SPLIT_1_FILE ## use split 1

UNIREP_BATCH_SIZE = 3500

# Use params defined in A003 models.py as these were used for the
# retrospective data efficiency work.
TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_DO_SPARSE_REFIT = config.get('sparse_refit', True) # CHANGED 8/10/19
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']

SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3


assert n_chains <= UNIREP_BATCH_SIZE

#####################

# Sync required data
# print('Syncing data')
# data_io_utils.sync_s3_path_to_local(TRAINING_SET_FILE, is_single_file=True)
# data_io_utils.sync_s3_path_to_local(paths.EVOTUNING_CKPT_DIR)

# Set seeds. This locks in the set of training sequences
# This also locks in the initial sequences used for SA
# as well as each chain's mutation rate.
np.random.seed(seed)
random.seed(seed)

# Grab training data. note the subset we grab will be driven by the random seed above.
print('Setting up training data')
train_df = pd.read_csv(TRAINING_SET_FILE)
sub_train_df = train_df.sample(n=n_train_seqs)

print(sub_train_df.head())

train_seqs = list(sub_train_df['seq'])

# print('AAA',train_seqs)
train_qfunc = np.array(sub_train_df['quantitative_function'])
# print('BBB',train_qfunc)

# Somewhat out of place, but set initial sequences for simulated annealing as well
# as the mutation rate for each chain.
# init_seqs = A006_common.propose_seqs(
#         [constants.AVGFP_AA_SEQ]*n_chains, 
#         [SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*n_chains, 
#         min_pos=A006_common.GFP_LIB_REGION[0], 
#         max_pos=A006_common.GFP_LIB_REGION[1])
# mu_muts_per_seq = 1.5*np.random.rand(n_chains) + 1
# print('mu_muts_per_seq:', mu_muts_per_seq) # debug



# Set up base model
# print('Setting up base model')
# tf.reset_default_graph()

# if model_name == 'ET_Global_Init_1':
#     base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
#     print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
# elif model_name == 'ET_Global_Init_2':
#     base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
#     print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
# elif model_name == 'ET_Random_Init_1':
#     base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
#     print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
# elif model_name =='OneHot':
#     # Just need it to generate one-hot reps.
#     # Top model created within OneHotRegressionModel doesn't actually get used.
#     base_model = models.OneHotRegressionModel('EnsembledRidge') 
# else:
#     assert False, 'Unsupported base model'
    
    
    
# Start a tensorflow session
# with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())

    # # Generate representations
    # def generate_reps(seq_list):        
    #     if 'babbler1900' == base_model.__class__.__name__:
    #         assert len(seq_list) <= UNIREP_BATCH_SIZE
    #         hiddens = base_model.get_all_hiddens(seq_list, sess)
    #         rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            
    #     elif 'OneHotRegressionModel' == base_model.__class__.__name__:
    #         rep = base_model.encode_seqs(seq_list)
            
    #     return rep
     
    # print('Generating training seq reps')
    # train_reps = generate_reps(train_seqs)
with open('/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/gfp_seq2name.txt') as f:
    lines = f.readlines()
d = {}
for line in lines:
    line = line.split()
    # print(line)
    d[line[0]] = line[1]

def load_ptsrep(seq_list):
    rep_list = []
    # rep_path = '/share/seqtonpy/gfp/knn_self_512_full/self_20201216_4__self_20201215_5_sota_right_n2_knnnet150_del_tor'
    rep_path = '/home/caiyi/embed/gfp/random_ptsrep'
    for seq in seq_list:
        rep_list.append(np.load(f'{rep_path}/{d[seq]}.npy'))
    return np.stack([np.mean(s, axis=0) for s in rep_list],0)

train_reps = load_ptsrep(train_seqs)
print(train_reps.shape)
    
# Build & train the top model.
print('Building top model with sparse refit = ', str(TOP_MODEL_DO_SPARSE_REFIT))
top_model = A003_common.train_ensembled_ridge(
    train_reps, 
    train_qfunc, 
    n_members=TOP_MODEL_ENSEMBLE_NMEMBERS, 
    subspace_proportion=TOP_MODEL_SUBSPACE_PROPORTION,
    normalize=TOP_MODEL_NORMALIZE, 
    do_sparse_refit=TOP_MODEL_DO_SPARSE_REFIT, 
    pval_cutoff=TOP_MODEL_PVAL_CUTOFF
)


test_reps = []
test_names = []
# path = '/home/caiyi/data/gfp/sa_ptsrep'
path = '/home/caiyi/embed/gfp/sa_random_ptsrep'
for f in os.listdir(path):
    test_reps.append(np.load(f'{path}/{f}'))
    test_names.append(f[:-4])
test_reps = np.stack([np.mean(s, axis=0) for s in test_reps],0)
d1={}

yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)


for i, y in enumerate(yhat):
    d1[test_names[i]] = y
# print(d1)

import re
with open('/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/sa.txt') as f:
    lines = f.readlines()
d = {}
for line in lines:
    line = line.split()
    if line[0].startswith('GFP_SimAnneal-ET_Global') and '0024' in line[0]:
        group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
        if group not in d:
            d[group] = [line]
        else:
            d[group].append(line)
d_real = {k: [] for k in d}
d_pts = {k: [] for k in d}
d_unirep = {k: [] for k in d}
for g, group in d.items():
    for m in group:
        d_real[g].append(float(m[1]))
        d_unirep[g].append(float(m[2]))
        d_pts[g].append(d1[m[0]])

r_pts = 0
rho_pts = 0
acc_pts = 0
auc_pts = 0
mse_pts = 0
r_unirep = 0
rho_unirep = 0
acc_unirep = 0
auc_unirep = 0
mse_unirep = 0

for g in d:
    print(len(d_pts[g]), len(d_real[g]), len(d_real[g]))
    r_pts += pearson(d_pts[g], d_real[g])
    rho_pts += spearman(d_pts[g], d_real[g])
    a, b = classify(d_pts[g], d_real[g], 1, 3.41076484873156)
    acc_pts += a
    auc_pts += b
    r_unirep += pearson(d_unirep[g], d_real[g])
    rho_unirep += spearman(d_unirep[g], d_real[g])
    a, b = classify(d_unirep[g], d_real[g], 1, 3.41076484873156)
    acc_unirep += a
    auc_unirep += b

r_pts /= len(d.keys())
rho_pts /= len(d.keys())
acc_pts /= len(d.keys())
auc_pts /= len(d.keys())
mse_pts /= len(d.keys())
r_unirep /= len(d.keys())
rho_unirep /= len(d.keys())
acc_unirep /= len(d.keys())
auc_unirep /= len(d.keys())
mse_unirep /= len(d.keys())
print(f'{sys.argv[1]}\t{r_pts}\t{rho_pts}\t{acc_pts}\t{auc_pts}\t{r_unirep}\t{rho_unirep}\t{acc_unirep}\t{auc_unirep}\n')
with open('random_ptsrep_output.txt', 'a') as f:
    f.write(f'{sys.argv[1]}\t{r_pts}\t{rho_pts}\t{acc_pts}\t{auc_pts}\t{r_unirep}\t{rho_unirep}\t{acc_unirep}\t{auc_unirep}\n')

# with open('predict.p', 'wb') as f:
#     pickle.dump(file=f, obj=top_model.predict(train_reps, return_std=True, return_member_predictions=True))
# Do simulated annealing
# def get_fitness(seqs):
#     reps = generate_reps(seqs)
#     yhat, yhat_std, yhat_mem = top_model.predict(reps, 
#             return_std=True, return_member_predictions=True)
            
#     nmut = utils.levenshtein_distance_matrix(
#             [constants.AVGFP_AA_SEQ], list(seqs)).reshape(-1)
    
#     mask = nmut > nmut_threshold
#     yhat[mask] = -np.inf 
#     yhat_std[mask] = 0 
#     yhat_mem[mask,:] = -np.inf 
    
#     return yhat, yhat_std, yhat_mem  

# sa_results = A006_common.anneal(
#     init_seqs, 
#     k=SIM_ANNEAL_K, 
#     T_max=T_max, 
#     mu_muts_per_seq=mu_muts_per_seq,
#     get_fitness_fn=get_fitness,
#     n_iter=sa_n_iter, 
#     decay_rate=temp_decay_rate,
#     min_mut_pos=min_mut_pos,
#     max_mut_pos=max_mut_pos)

    
# Aggregate results and export
# results = {
#     # 'sa_results': sa_results,
#     'top_model': top_model,
#     'train_df': sub_train_df,
#     'train_seq_reps': train_reps,
#     'base_model': model_name
# }
    
# output_file = os.path.basename(output_file)
# with open(output_file, 'wb') as f:
#     pickle.dump(file=f, obj=results)
# print(f'Dumped to {output_file}')
# print('Syncing results to S3')
# print('Post-publication note: Skipping. Public S3 sync for this bucket is disabled.')
# Sync step below is commented out. Left for the curious reader.
# 
# Sync up to S3.
#cmd = ('aws s3 cp %s s3://efficient-protein-design/chip_1/simulated_annealing/GFP/%s' % 
#       (output_file, output_file))
#subprocess.check_call(cmd, shell=True)


# Check if there is a standard out log file passed by Hyperborg. Sync it to s3 if so.
#possible_log = config_file.replace('.p', '.log')
#if os.path.exists(possible_log):
#    cmd = ('aws s3 cp %s s3://efficient-protein-design/chip_1/simulated_annealing/GFP/%s' % 
#       (possible_log, possible_log))
#    subprocess.check_call(cmd, shell=True)
    
