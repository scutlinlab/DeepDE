import os
import sys
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# sys.path.insert(0, '/home/caiyi/github/low-N-protein-engineering-master/ptsrep')
# import low_n_utils
# from low_n_utils import Semilabel
import misc_utils
from misc_utils import Semilabel, PtsRep

sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/common')
#'/home/caiyi/github/low-N-protein-engineering-master/analysis/common'
import paths
import constants
import utils

sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/common')
from process_pdb import process_pdb
import rep_utils

sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization/')
#'/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization/'
import models
import A003_common

sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing')
#'/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing'
import A006_common

# path1 = "/home/joseph/KNN/outputs/self_20201215_5_sota_right_n2_knnnet150_del_tor/50_Linear.pth"
path1 = '/home/caiyi/seqrep/outputs/models/ptsrep__20121505_5_sota_right_n2_knnnet150_del_tor/50_model.pth.tar'
# path3 = '/home/wangqihan/1emm-mod.pdb'
path3 = '/home/caiyi/data/gfp/src/pdb/1emm-mod.pdb'


# CONFIGURATION
config_file = str(sys.argv[1])
print('Config file:', config_file)

with open(config_file, 'rb') as f:
    config = pickle.load(f)
    
seed = config['seed']
# n_train_seqs = config['n_train_seqs']
n_train_seqs = 117
model_name = config['model'] 
# n_chains = config['n_chains']
n_chains = 500
T_max = config['T_max'][0:n_chains]
sa_n_iter = 3000   # config['sa_n_iter']
temp_decay_rate = config['temp_decay_rate']
min_mut_pos = config['min_mut_pos']
max_mut_pos = config['max_mut_pos']
nmut_threshold = config['nmut_threshold']
# output_file = config['output_file']
output_file = 'mcmc-ptsrep-400-3500-3000.p'

for k in config:
    if k == 'init_seqs':
        print(k + ':', config[k][:5])
    else:
        print(k + ':', config[k])

device_num = 5
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)
# orign_arrays = {}

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


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

np.random.seed(seed)
random.seed(seed)

# Grab training data. note the subset we grab will be driven by the random seed above.
print('Setting up training data')
train_df = pd.read_csv(TRAINING_SET_FILE)
sub_train_df = train_df.sample(n=n_train_seqs)

print(sub_train_df.head())

train_seqs = list(sub_train_df['seq'])
train_qfunc = np.array(sub_train_df['quantitative_function'])

# Somewhat out of place, but set initial sequences for simulated annealing as well
# as the mutation rate for each chain.
init_seqs = A006_common.propose_seqs(
        [constants.AVGFP_AA_SEQ]*n_chains, 
        [SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*n_chains, 
        min_pos=A006_common.GFP_LIB_REGION[0], 
        max_pos=A006_common.GFP_LIB_REGION[1])
# print("init_seq;",len(init_seqs))
# print(init_seqs[0])
# print(len(init_seqs[0]))
mu_muts_per_seq = 1.5 * np.random.rand(n_chains) + 1
print('mu_muts_per_seq:', mu_muts_per_seq[:10], '......') # debug


with open('/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/gfp_seq2name.txt') as f:
    lines = f.readlines()
d = {}
for line in lines:
    line = line.split()
    # print(line)
    d[line[0]] = line[1]

def load_ptsrep(seq_list):
    rep_list = []
    rep_path = '/share/joseph/seqtonpy/gfp/knn_self_512_full/self_20201216_4__self_20201215_5_sota_right_n2_knnnet150_del_tor'
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

coefs = []
alphas = []
for i in range(top_model.n_members):
    coefs.append(top_model.model_ensemble[i]['model'].coef_)
    alphas.append(top_model.model_ensemble[i]['model'].alpha_)
print('mean_coef', np.mean(np.abs(coefs)))
print('mean_alpha', np.mean(alphas))
exit()
# import random
# test_reps = []
# test_names = []
# path = '/home/caiyi/data/gfp/sa_ptsrep'
# for f in os.listdir(path):
#     test_reps.append(np.load(f'{path}/{f}'))
#     test_names.append(f[:-4])
# test_reps = np.stack([np.mean(s, axis=0) for s in test_reps],0)
# print('test_reps:',test_reps.shape)
# d1={}

# yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)

# random.shuffle(test_names)
# for i, y in enumerate(yhat):
#     d1[test_names[i]] = y
# # print(d1)

# import re
# with open('/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/sa.txt') as f:
#     lines = f.readlines()
# d = {}
# for line in lines:
#     line = line.split()
#     if line[0].startswith('GFP_SimAnneal-ET_Global') and '0096' in line[0]:
#         group = re.sub(r'-seq_idx_\d+_\d+$', '', line[0])
#         if group not in d:
#             d[group] = [line]
#         else:
#             d[group].append(line)
# d_real = {k: [] for k in d}
# d_pts = {k: [] for k in d}
# d_unirep = {k: [] for k in d}
# for g, group in d.items():
#     for m in group:
#         d_real[g].append(float(m[1]))
#         d_unirep[g].append(float(m[2]))
#         d_pts[g].append(d1[m[0]])

# r_pts, rho_pts, acc_pts, auc_pts, mse_pts, r_unirep, rho_unirep, acc_unirep, auc_unirep, mse_unirep = [0 for _ in range(10)]

# for g in d:
#     print(len(d_pts[g]), len(d_real[g]), len(d_real[g]))
#     r_pts += low_n_utils.pearson(d_pts[g], d_real[g])
#     rho_pts += low_n_utils.spearman(d_pts[g], d_real[g])
#     a, b = low_n_utils.classify(d_pts[g], d_real[g], 1, 3.41076484873156)
#     acc_pts += a
#     auc_pts += b
#     r_unirep += low_n_utils.pearson(d_unirep[g], d_real[g])
#     rho_unirep += low_n_utils.spearman(d_unirep[g], d_real[g])
#     a, b = low_n_utils.classify(d_unirep[g], d_real[g], 1, 3.41076484873156)
#     acc_unirep += a
#     auc_unirep += b

# r_pts /= len(d.keys())
# rho_pts /= len(d.keys())
# acc_pts /= len(d.keys())
# auc_pts /= len(d.keys())
# mse_pts /= len(d.keys())
# r_unirep /= len(d.keys())
# rho_unirep /= len(d.keys())
# acc_unirep /= len(d.keys())
# auc_unirep /= len(d.keys())
# mse_unirep /= len(d.keys())
# # print(f'{sys.argv[1]}\t{r_pts}\t{rho_pts}\t{acc_pts}\t{auc_pts}\t{r_unirep}\t{rho_unirep}\t{acc_unirep}\t{auc_unirep}\n')
# with open('output.txt', 'a') as f: 
#     f.write(f'{sys.argv[1]}\t{r_pts}\t{rho_pts}\t{acc_pts}\t{auc_pts}\t{r_unirep}\t{rho_unirep}\t{acc_unirep}\t{auc_unirep}\n')



# EMBEDDING

def to_embedding(knr_list, model_path):
    ebd_list = []
    # full_dataset = low_n_utils.Knnonehot(knr_list)
    full_dataset = misc_utils.Knnonehot(knr_list)
    data_loader = DataLoader(dataset=full_dataset, shuffle=False, batch_size=1)
    print("data_loader:",len(data_loader))

    with torch.no_grad():
        # model = torch.load(model_path, map_location=device)
        model = PtsRep(input_size=135, hidden_size=384, vocab_size=20, dropout=0.1).to('cuda:0')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        model.is_training = False

    for arrays in tqdm(data_loader, ascii=True):
        arrays = arrays.to(device).float()
        pred = model(arrays[0]).float()
        ebd_list.append(pred.data.cpu().numpy())
    ebd_reps = np.stack([np.mean(s,axis = 0) for s in ebd_list],0)
    return ebd_reps

HYDROPATHICITY = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                  1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]

BULKINESS = [11.5, 13.46, 11.68, 13.57, 19.8, 3.4, 13.69, 21.4, 15.71, 21.4,
             16.25, 12.82, 17.43, 14.45, 14.28, 9.47, 15.77, 21.57, 21.67, 18.03]

FLEXIBILITY = [14.0, 0.05, 12.0, 5.4, 7.5, 23.0, 4.0, 1.6, 1.9, 5.1,
               0.05, 14.0, 0.05, 4.8, 2.6, 19.0, 9.3, 2.6, 0.05, 0.05]

AA_LIST = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']

hydropathicity = (5.5 - np.array(HYDROPATHICITY)) / 10
bulkiness = np.array(BULKINESS) / 21.67
flexibility = (25 - np.array(FLEXIBILITY)) / 25


def substitute(knr, ref_seq, seq):
    new_knr = np.copy(knr)
    muts = []
    for i, aa in enumerate(seq):
        if aa != ref_seq[i]:
            muts.append((i, aa))
    for pos, aa_name in muts:
        aa = AA_LIST.index(aa_name)
        idx = np.round(new_knr[:, :, 5] * 200) + np.arange(223).reshape(223, 1)
        tmp = new_knr[np.where(idx == pos)]
        tmp[:, -3:] = (hydropathicity[aa], bulkiness[aa], flexibility[aa])
        new_knr[np.where(idx == pos)] = tmp
    return new_knr
    

def to_ebd(state_seqs, path1, path3):
    knr_list = []
    chain, model = 'A', '1'
    pdb_profile, atom_lines = process_pdb(path3, atoms_type=['N', 'CA', 'C'])
    atoms_data = atom_lines[chain, model]
    coord_array_ca, acid_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=['N', 'CA', 'C'])
    print(len(coord_array))
    
    ref_seq = state_seqs[0]
    array_ref = rep_utils.get_knn_150(coord_array, ref_seq)
    for seq in tqdm(state_seqs, ascii=True):
        # knr = rep_utils.get_knn_150(coord_array, seq)
        knr = substitute(array_ref, ref_seq, seq)
        knr_list.append(knr)
    ebd_reps = to_embedding(knr_list,path1)
    print(ebd_reps.shape)
    return ebd_reps


# SIMULATED ANNEALING
def get_fitness(seqs1):
    seqs = []
    for seq in seqs1:
        seqs.append(seq[6:-9])
    reps = to_ebd(seqs, path1 = path1,path3 = path3)
    print('reps:',reps.shape)
    yhat, yhat_std, yhat_mem = top_model.predict(reps, 
            return_std=True, return_member_predictions=True)
            
    nmut = utils.levenshtein_distance_matrix(
            [constants.AVGFP_AA_SEQ], list(seqs1)).reshape(-1)
    print(nmut)
    
    mask = nmut > nmut_threshold
    yhat[mask] = -np.inf 
    yhat_std[mask] = 0 
    yhat_mem[mask,:] = -np.inf 
    
    return yhat, yhat_std, yhat_mem  

sa_results = A006_common.anneal(
    init_seqs, 
    k=SIM_ANNEAL_K, 
    T_max=T_max, 
    mu_muts_per_seq=mu_muts_per_seq,
    get_fitness_fn=get_fitness,
    n_iter=sa_n_iter, 
    decay_rate=temp_decay_rate,
    min_mut_pos=min_mut_pos,
    max_mut_pos=max_mut_pos)

results = {
    'sa_results': sa_results,
    'top_model': top_model,
    'train_df': sub_train_df,
    'train_seq_reps': train_reps,
    'base_model': model_name
}

output_file = os.path.basename(output_file)
with open('/home/caiyi/github/low-N-protein-engineering-master/ptsrep/' + output_file, 'wb') as f:
    pickle.dump(file=f, obj=results)
print(f'Dumped to {output_file}')
