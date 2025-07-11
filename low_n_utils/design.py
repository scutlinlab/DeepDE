import os
import shutil
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json
import tensorflow as tf
import argparse

import misc_utils
from misc_utils import PtsRep

# sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/common')
from low_n_utils import paths, low_n_utils

# sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/common')
from low_n_utils.process_pdb import process_pdb
from low_n_utils import rep_utils

# sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization/')
from low_n_utils import models, A003_common

# sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing')
from low_n_utils import A006_common, ga
from low_n_utils.unirep import babbler1900 as babbler


parser = argparse.ArgumentParser()
parser.add_argument('-cf', '--config_path')
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-n', '--n_train', default=None, type=int)
parser.add_argument('-e', '--exp_name', default=None)
parser.add_argument('-g', '--gpu', default=None)

args = parser.parse_args()

with open(args.config_path) as f:
    config = json.load(f)

torch.set_num_threads(1)
gpu = args.gpu if args.gpu else config['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
model_path = config['model_path']
load_model_method = config['load_model_method']
pdb_path = config['pdb_path']
rep_path = config['rep_path']
train_rep_src = config['train_rep_src']
wt_seq = config['wt_seq']
seq_start, seq_end = config['seq_start'], config['seq_end']
struct_seq_len = config['struct_seq_len']
min_pos, max_pos = config['min_pos'], config['max_pos']
training_set_file = config['training_set_file']
fitness_name = config['fitness_name']
use_ga = config['use_ga']
sampling_method = config['sampling_method']

if use_ga:
    propose_seqs_fn = ga.propose_seqs
    anneal_fn = ga.anneal
    algo_name = 'ga'
else:
    propose_seqs_fn = A006_common.propose_seqs
    anneal_fn = A006_common.anneal
    algo_name = 'mcmc'

seed = config['seed']
n_train_seqs = args.n_train if args.n_train else config['n_train_seqs']
model_name = args.model if args.model else config['model']
n_chains = config['n_chains']
T_max = np.array([config['T_max']] * 3500)[0:n_chains]
sa_n_iter = config['sa_n_iter']
nmut_threshold = config['nmut_radius']
exp_name = args.exp_name if args.exp_name else config['exp_name']


# Hard constants

UNIREP_BATCH_SIZE = config['batch_size']

# Use params defined in A003 models.py as these were used for the
# retrospective data efficiency work.
TOP_MODEL_ENSEMBLE_NMEMBERS = config['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = config['subspace_proportion']
TOP_MODEL_NORMALIZE = config['normalize']
TOP_MODEL_DO_SPARSE_REFIT = config['sparse_refit']  # CHANGED 8/10/19
TOP_MODEL_PVAL_CUTOFF = config['pval_cutoff']

for k in config:
    if not k.startswith('/'):
        print(k + ':', config[k])

device_num = 0
torch.cuda.set_device(device_num)
device = torch.device('cuda:%d' % device_num)

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


# assert n_chains <= UNIREP_BATCH_SIZE

#####################

np.random.seed(seed)
random.seed(seed)

# Grab training data. note the subset we grab will be driven by the random seed above.
print('Setting up training data')
train_df = pd.read_csv(training_set_file)
if sampling_method == 'random':
    print(n_train_seqs)
    sub_train_df = train_df.sample(n=n_train_seqs)
elif sampling_method == 'all':
    if fitness_name == 'stability':
        sub_train_df = train_df[train_df[fitness_name].notnull()]
    else:
        sub_train_df = train_df
elif sampling_method == '50-50_1.0':
    sub_train_df1 = train_df[train_df[fitness_name] >= 1]
    sub_train_df2 = train_df[train_df[fitness_name] < 1].sample(n=len(sub_train_df1))
    sub_train_df = pd.concat([sub_train_df1, sub_train_df2])
else:
    raise NameError('Wrong value of `sampling_method`')

n_train_seqs = len(sub_train_df)
output_file = f'{exp_name}_{fitness_name}_{algo_name}_{model_name}_{n_train_seqs}_{n_chains}_{sa_n_iter}.p'
config['output_file'] = output_file
save_json_path = f'/home/caiyi/low_n/outputs/configs/{output_file[:-2]}.json'
if not os.path.exists(save_json_path):
    shutil.copy(args.config_path, save_json_path)

print(sub_train_df.head())

train_seqs = list(sub_train_df['seq'])
if train_rep_src == 'load_by_name':
    train_names = list(sub_train_df['name'])
train_qfunc = np.array(sub_train_df[fitness_name])

# Somewhat out of place, but set initial sequences for simulated annealing as well
# as the mutation rate for each chain.
init_seqs = propose_seqs_fn(
        [wt_seq]*n_chains, 
        [config['init_nmut_radius']]*n_chains, 
        min_pos=min_pos, 
        max_pos=max_pos)

mu_muts_per_seq = 1.5 * np.random.rand(n_chains) + 1
print('mu_muts_per_seq:', mu_muts_per_seq[:10], '......') # debug

if train_rep_src == 'load_by_seq':
    with open('/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/gfp_seq2name.txt') as f:
        lines = f.readlines()
    d = {}
    for line in lines:
        line = line.split()
        # print(line)
        d[line[0]] = line[1]


def load_ptsrep_by_seq(seq_list):
    rep_list = []
    for seq in seq_list:
        rep_list.append(np.load(f'{rep_path}/{d[seq]}.npy'))
    return np.stack([np.mean(s, axis=0) for s in rep_list],0)


def load_ptsrep_by_name(name_list):
    rep_list = []
    for name in name_list:
        rep_list.append(np.load(f'{rep_path}/{name}.npy'))
    return np.stack([np.mean(s, axis=0) for s in rep_list],0)

# EMBEDDING

def knr2ptsrep(knr_list, model_path):
    ebd_list = []
    full_dataset = misc_utils.Knnonehot(knr_list)
    data_loader = DataLoader(dataset=full_dataset, shuffle=False, batch_size=config['batch_size'])
    print("data_loader:", len(data_loader))

    with torch.no_grad():
        if load_model_method == 'full_model':
            model = torch.load(model_path, map_location=device)
        elif load_model_method == 'state_dict':
            model = PtsRep(input_size=135, hidden_size=384, vocab_size=20, dropout=0.1).to('cuda:0')
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        else:
            raise NameError('No such model loading method!')
        model.eval()
        model.is_training = False

    for arrays in tqdm(data_loader, ascii=True):
        arrays = arrays.to(device).float()
        # pred = model(arrays[0]).float()
        pred = model(arrays).float()
        ebd_list.append(pred.data.cpu().numpy())
    # ebd_reps = np.stack([np.mean(s, axis=0) for s in ebd_list], axis=0)
    ebd_reps = np.vstack([np.mean(s, axis=1) for s in ebd_list])
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
        # 200为KNR中位置编码使用的normalize
        idx = np.round(new_knr[:, :, 5] * 200) + np.arange(struct_seq_len).reshape(struct_seq_len, 1)
        tmp = new_knr[np.where(idx == pos)]
        tmp[:, -3:] = (hydropathicity[aa], bulkiness[aa], flexibility[aa])
        new_knr[np.where(idx == pos)] = tmp
    return new_knr
    

def pdb_and_seq_to_ptsrep(state_seqs, model_path, pdb_path):
    knr_list = []
    chain, model = 'A', '1'
    pdb_profile, atom_lines = process_pdb(pdb_path, atoms_type=['N', 'CA', 'C'])
    atoms_data = atom_lines[chain, model]
    coord_array_ca, struct_aa_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=['N', 'CA', 'C'])
    print(len(coord_array))
    
    ref_seq = state_seqs[0]
    rep_utils.validate_aligning(struct_aa_array, ref_seq, allowed_mismatches=50)
    array_ref = rep_utils.get_knn_150(coord_array, ref_seq)
    for seq in tqdm(state_seqs, ascii=True):
        # knr = rep_utils.get_knn_150(coord_array, seq)
        knr = substitute(array_ref, ref_seq, seq)
        knr_list.append(knr)
    ebd_reps = knr2ptsrep(knr_list, model_path)
    print(ebd_reps.shape)
    return ebd_reps

if train_rep_src == 'load_by_seq':
    train_reps = load_ptsrep_by_seq(train_seqs)
elif train_rep_src == 'load_by_name':
    train_reps = load_ptsrep_by_name(train_names)
elif train_rep_src == 'generate':
    if 'ptsrep' in model_name:
        train_seqs_trimmed = []
        for seq in train_seqs:
            train_seqs_trimmed.append(seq[seq_start:seq_end])
        train_reps = pdb_and_seq_to_ptsrep(train_seqs_trimmed, model_path, pdb_path)
    elif 'unirep' in model_name or 'onehot' in model_name:
        print('Setting up base model')
        tf.reset_default_graph()
        if model_name == 'unirep':
            UNIREP_WEIGHT_PATH = '/home/caiyi/github/unirep_embedding/1900_weights/'
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
            print('Loading weights from:', UNIREP_WEIGHT_PATH)
        elif model_name == 'eunirep_1':
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
            print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
        elif model_name == 'eunirep_2':
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
            print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
        elif model_name == 'random_unirep':
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
            print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
        elif model_name =='onehot':
            # Just need it to generate one-hot reps.
            # Top model created within OneHotRegressionModel doesn't actually get used.
            base_model = models.OneHotRegressionModel('EnsembledRidge')
        elif model_name == 'eunirep_petase': 
            UNIREP_WEIGHT_PATH = '/home/caiyi/unirep/eunirep_petase_21091902_30/'
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
            print('Loading weights from:', UNIREP_WEIGHT_PATH)
        else:
            assert False, 'Unsupported base model'
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        def generate_reps(seq_list):        
            if 'babbler1900' == base_model.__class__.__name__:
                assert len(seq_list) <= UNIREP_BATCH_SIZE
                hiddens = base_model.get_all_hiddens(seq_list, sess)
                rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
                
            elif 'OneHotRegressionModel' == base_model.__class__.__name__:
                rep = base_model.encode_seqs(seq_list)
                
            return rep

        print('Generating training seq reps')
        train_reps = generate_reps(train_seqs)
    else:
        raise NameError(f'Incorrect model name: {model_name}')

print('train_reps:', train_reps.shape)
    
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
coef_mean = np.mean(np.abs(coefs))
alpha_mean = np.mean(alphas)
print('mean_coef', coef_mean)
print('mean_alpha', alpha_mean, alphas[:10])
np.save(f'/home/caiyi/low_n/outputs/top_models/{output_file[:-2]}_coefs.npy', np.array(coefs))
np.save(f'/home/caiyi/low_n/outputs/top_models/{output_file[:-2]}_alphas.npy', np.array(alphas))


if 'ptsrep' in model_name:
    # SIMULATED ANNEALING
    def get_fitness(seqs1):
        seqs = []
        for seq in seqs1:
            seqs.append(seq[seq_start:seq_end])
        reps = pdb_and_seq_to_ptsrep(seqs, model_path, pdb_path)
        print('reps:', reps.shape)
        yhat, yhat_std, yhat_mem = top_model.predict(reps, 
                return_std=True, return_member_predictions=True)
                
        nmut = low_n_utils.levenshtein_distance_matrix(
                [wt_seq], list(seqs1)).reshape(-1)
        print(nmut[:50])
        
        mask = nmut > nmut_threshold
        yhat[mask] = -np.inf 
        yhat_std[mask] = 0 
        yhat_mem[mask,:] = -np.inf
        
        return yhat, yhat_std, yhat_mem, nmut
elif 'unirep' in model_name or 'onehot' in model_name:
    def get_fitness(seqs):
        reps = generate_reps(seqs)
        yhat, yhat_std, yhat_mem = top_model.predict(reps,
                return_std=True, return_member_predictions=True)
                
        nmut = low_n_utils.levenshtein_distance_matrix(
                [wt_seq], list(seqs)).reshape(-1)
        print(nmut[:50])
        
        mask = nmut > nmut_threshold
        yhat[mask] = -np.inf
        yhat_std[mask] = 0
        yhat_mem[mask,:] = -np.inf
        
        return yhat, yhat_std, yhat_mem, nmut


train_info = {
    'top_model': top_model,
    'train_df': sub_train_df,
    'train_seq_reps': train_reps,
    'base_model': model_name
}

# output_file_train = os.path.basename(output_file_train)
output_file = os.path.basename(output_file)
sa_results = anneal_fn(
    init_seqs, 
    k=config['k'], 
    T_max=T_max, 
    mu_muts_per_seq=mu_muts_per_seq,
    get_fitness_fn=get_fitness,
    n_iter=sa_n_iter, 
    decay_rate=config['temp_decay_rate'],
    min_mut_pos=min_pos,
    max_mut_pos=max_pos,
    save_results=config['save_every_iter'],
    output_file=output_file,
    train_info=train_info,
    config=config)

results = train_info
results.update({'sa_results': sa_results})

if train_rep_src == 'generate':
    sess.close()

with open('/home/caiyi/low_n/outputs/pkl/' + output_file , 'wb') as f:
    pickle.dump(file=f, obj=results)
print(f'Dumped to {output_file}')

if algo_name == 'mcmc':
    with open('/home/caiyi/low_n/outputs/summary.txt', 'a') as f:
        fitness_init = np.mean(sa_results['fitness_history'][0])
        fitness_std_init = np.mean(sa_results['fitness_std_history'][0])
        fitness_last = np.mean(sa_results['fitness_history'][-1])
        fitness_std_last = np.mean(sa_results['fitness_std_history'][-1])
        fitness_best = np.mean(sa_results['best_fitness'])
        fitness_std_best = np.std(sa_results['best_fitness'])
        nmut_init = np.mean(sa_results['nmut_history'][0])
        nmut_std_init = np.std(sa_results['nmut_history'][0])
        nmut_last = np.mean(sa_results['nmut_history'][-1])
        nmut_std_last = np.std(sa_results['nmut_history'][-1])
        nmut_best = np.mean(sa_results['best_nmut'])
        nmut_std_best = np.std(sa_results['best_nmut'])

        fit_his = sa_results['fitness_history'][1:]
        accept_history = [[] for i in range(len(fit_his))]
        delta_history = [[] for i in range(len(fit_his))]
        last_iter = sa_results['fitness_history'][0]

        for i, iter_ in enumerate(fit_his):
            for j in range(len(iter_)):
                if iter_[j] == last_iter[j]:
                    accept_history[i].append(0)
                    delta_history[i].append(0)
                else:
                    accept_history[i].append(1)
                    delta_history[i].append(iter_[j] - last_iter[j])
            last_iter = iter_
        
        accept_times_mean = np.mean(np.sum(accept_history, 0))
        print('mean_accepct_times:', accept_times_mean)

        delta_nonzeros = np.array(delta_history).ravel()[np.flatnonzero(delta_history)]
        delta_mean = np.mean(delta_nonzeros)
        print('mean_delta:', round(delta_mean, 4))

        f.write(f'{exp_name}\t{n_train_seqs}\t{model_name}\t{coef_mean}\t{alpha_mean}\t{fitness_init}\t'
                f'{fitness_std_init}\t{fitness_last}\t{fitness_std_last}\t{fitness_best}\t'
                f'{fitness_std_best}\t{nmut_init}\t{nmut_std_init}\t{nmut_last}\t{nmut_std_last}\t'
                f'{nmut_best}\t{nmut_std_best}\t{accept_times_mean}\t{delta_mean}\n')
