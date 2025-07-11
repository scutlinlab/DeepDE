import os
import sys
import warnings
import multiprocessing as mp
import random
import copy
import pickle

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


from . import data_io_utils, paths, constants, low_n_utils


def convert_result_vals_to_mat(res):
    res['seq_history'] = np.stack(res['seq_history'], 0)
    res['fitness_history'] = np.stack(res['fitness_history'], 0)
    res['fitness_std_history'] = np.stack(res['fitness_std_history'], 0)
    #res['fitness_mem_pred_history'] = np.stack(res['fitness_mem_pred_history'], 0)
        
    return res

def isolate_highly_functional_trajectories(res, burnin, fitness_threshold):
    fit_mat = res['fitness_history'] # iter x chains
    
    # 1. Select trajectories that reach functional sequences
    # after the specified number of burn-in iterations.
    bi_fit_mat = fit_mat[burnin:]
    
    mask = np.any(bi_fit_mat > fitness_threshold, axis=0)
    
    return mask

def isolate_valley_crossing_trajectories(res, burnin, fitness_threshold, valley_threshold=0.2):
    fit_mat = res['fitness_history'] # iter x chains
    
    bi_fit_mat = fit_mat[burnin:]
    
    functional_mask = np.any(bi_fit_mat > fitness_threshold, axis=0)
    valley_mask = np.any(fit_mat < valley_threshold, axis=0) # dont include burn in
    
    return np.logical_and(functional_mask, valley_mask)
    
    
def build_pwm(seqs):
    ohe = np.stack([low_n_utils.encode_aa_seq_as_one_hot(s, flatten=False) for s in seqs], 0)
    pwm = np.mean(ohe, axis=0) + 1e-6
    return pwm
    
def calc_effective_num_residues_per_site(seqs):
    pwm = build_pwm(seqs)
    return np.exp(-np.sum(pwm*np.log(pwm), axis=0))

def get_best_sequence_in_each_trajectory(res, burnin=0, max_sa_itr=None):
    seq_mat = res['seq_history']
    fit_mat = res['fitness_history']
    fit_std_mat = res['fitness_std_history']
    
    if max_sa_itr is None:
        max_sa_itr = fit_mat.shape[0]
    
    best_seq_idx = np.argmax(fit_mat[burnin:max_sa_itr,:], axis=0) + burnin
    
    best_seqs = []
    best_seq_fitness = []
    best_seq_fitness_std = []
    for i in range(seq_mat.shape[1]):
        best_seqs.append(seq_mat[best_seq_idx[i], i])
        best_seq_fitness.append(fit_mat[best_seq_idx[i], i])
        best_seq_fitness_std.append(fit_std_mat[best_seq_idx[i], i])
        
    return np.array(best_seqs), np.array(best_seq_fitness), np.array(best_seq_fitness_std), best_seq_idx

def obtain_top_n_functional_seqs(res, burnin, n=100):
    ufit = -np.sort(-np.unique(res['fitness_history'].reshape(-1)))
    
    idx = 1
    while True:
        int_traj_mask = isolate_highly_functional_trajectories(res, burnin, ufit[idx])
        
        idx += 1
        if np.sum(int_traj_mask) >= n:
            break

    best_seqs, best_seq_fitness, best_seq_fitness_std, best_seq_idx = get_best_sequence_in_each_trajectory(res)
    
    return best_seqs[int_traj_mask], best_seq_fitness[int_traj_mask], best_seq_fitness_std[int_traj_mask], int_traj_mask


def load_results(res_file):
    with open(res_file, 'rb') as f:
        res = pickle.load(f)
        
    res_sa = convert_result_vals_to_mat(res['sa_results'])
    return res, res_sa
    

def select_top_seqs(res_file, nseq_select, burnin=250, max_sa_itr=None):
    print('SELECTION')
    res_file_name = os.path.basename(res_file)
        
    print(res_file)
    print('Loading results and converting SA histories to numpy arrays')
    res, res_sa = load_results(res_file)
    fit_mat = res_sa['fitness_history']

    init_fitness = fit_mat[0,:]

    print('Selecting top sequences')
    # First identify the best sequence in each SA trajectory.
    top_seqs, top_seq_fitness, _, top_seq_idx = get_best_sequence_in_each_trajectory(
            res_sa, burnin=burnin, max_sa_itr=max_sa_itr)

    # Now, select the top seqs of the best-in-trajectory sequences. 
    # These are are our official selections!
    # top_seq_idx is an index for each trajectory that says where in the trajectory the best sequence is. 
    sidx = np.argsort(-top_seq_fitness)
    top_sidx = sidx[:nseq_select]

    trajectory_indices_yielding_top_seqs = top_sidx
    seq_indices_inside_top_trajectories = top_seq_idx[top_sidx]
    selected_top_seqs = top_seqs[top_sidx] ## official selection
    selected_top_seq_fitness = top_seq_fitness[top_sidx] ## official selection
    
    # selected_top_ensemble_fitness_preds = []
    # for i in range(len(trajectory_indices_yielding_top_seqs)):
    #     selected_top_ensemble_fitness_preds.append(
    #         res_sa['fitness_mem_pred_history'][seq_indices_inside_top_trajectories[i]][
    #             trajectory_indices_yielding_top_seqs[i]]
    #     )
        
    # Turn these selections into a dataframe
    id_prefix = res_file_name.replace('.p', '')
    fit_mat_idx = [str(s[0]) + '_' + str(s[1]) for s in list(zip(*[list(seq_indices_inside_top_trajectories), 
           list(trajectory_indices_yielding_top_seqs)]))]
    seq_ids = [id_prefix + '-seq_idx_' + fmi for fmi in fit_mat_idx]
    
    select_df = pd.DataFrame()
    select_df['id'] = seq_ids
    select_df['seq_idx'] = seq_indices_inside_top_trajectories # row idx of res_sa['fitness_history']
    select_df['trajectory_idx'] = trajectory_indices_yielding_top_seqs # col idx of res_sa['fitness_history']
    select_df['predicted_fitness'] = selected_top_seq_fitness
    # select_df['ensemble_predicted_fitness'] = selected_top_ensemble_fitness_preds
    select_df['seq'] = selected_top_seqs
    
    return select_df, res, res_sa


def predict_activity(test_seqs, model_name):
    import os
    import sys
    import shutil
    import random
    import numpy as np
    import pandas as pd
    import torch
    import json
    import tensorflow as tf
    import argparse
    import pickle
    from scipy import stats
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    from low_n_utils import paths
    from low_n_utils import models, A003_common
    from low_n_utils.unirep import babbler1900 as babbler
    from low_n_utils.process_pdb import process_pdb
    from low_n_utils import rep_utils
    import misc_utils
    from misc_utils import PtsRep

    sys.path.append('/home/caiyi/low_n/')

    config = {}
    config['batch_size'] = 1000
    UNIREP_BATCH_SIZE = 1000
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = 0
    load_model_method = 'state_dict'
    pdb_path = '/home/caiyi/data/petase/pdb/6qgc.pdb'
    seq_start = 28
    seq_end = 290
    test_rep_src = 'generate'
    if model_name == 'eunirep_petase':
        top_model_path = '/home/caiyi/low_n/outputs/top_models/21111504_paumnewcode7all_activity_ga_eunirep_petase_301_1000_1000_model.pkl'
        model_path = ''
    elif model_name == 'eptsrep_petase':
        top_model_path = '/home/caiyi/low_n/outputs/top_models/21111501_papmnewcode7all_activity_mcmc_eptsrep_petase_301_1000_1000_model.pkl'
        model_path = '/home/caiyi/seqrep/outputs/models/lm__21091902_eptsrep_384_petase_correct-knr/30_27651_model.pth.tar'
        
    with open(top_model_path, 'rb') as f:
        top_model = pickle.load(f)
    coefs = []
    alphas = []
    for i in range(top_model.n_members):
        coefs.append(top_model.model_ensemble[i]['model'].coef_)
        alphas.append(top_model.model_ensemble[i]['model'].alpha_)
    coef_mean = np.mean(np.abs(coefs))
    alpha_mean = np.mean(alphas)
    print('mean_coef', coef_mean)
    print('mean_alpha', alpha_mean, alphas[:10])


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

    def substitute(knr, ref_seq, seq):
        new_knr = np.copy(knr)
        muts = []
        for i, aa in enumerate(seq):
            if aa != ref_seq[i]:
                muts.append((i, aa))
        for pos, aa_name in muts:
            aa = AA_LIST.index(aa_name)
            # 200为KNR中位置编码使用的normalize
            struct_seq_len = knr.shape[0]
            idx = np.round(new_knr[:, :, 5] * 200) + np.arange(struct_seq_len).reshape(struct_seq_len, 1)
            tmp = new_knr[np.where(idx == pos)]
            tmp[:, -3:] = (hydropathicity[aa], bulkiness[aa], flexibility[aa])
            new_knr[np.where(idx == pos)] = tmp
        return new_knr
        

    def pdb_and_seq_to_ptsrep(seqs, model_path, pdb_path):
        knr_list = []
        chain, model = 'A', '1'
        pdb_profile, atom_lines = process_pdb(pdb_path, atoms_type=['N', 'CA', 'C'])
        atoms_data = atom_lines[chain, model]
        coord_array_ca, struct_aa_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=['N', 'CA', 'C'])
        print(len(coord_array))
        ref_seq = seqs[0]
        rep_utils.validate_aligning(struct_aa_array, ref_seq, allowed_mismatches=20)
        knr_ref = rep_utils.get_knn_150(coord_array, ref_seq)
        for seq in tqdm(seqs, ascii=True):
            # knr = rep_utils.get_knn_150(coord_array, seq)
            knr = substitute(knr_ref, ref_seq, seq)
            knr_list.append(knr)
        ebd_reps = knr2ptsrep(knr_list, model_path)
        print(ebd_reps.shape)
        return ebd_reps

    if 'unirep' in model_name or 'onehot' in model_name:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        print('Setting up base model')
        tf.reset_default_graph()
        if model_name == 'unirep':
            UNIREP_WEIGHT_PATH = '/home/caiyi/github/unirep_embedding/1900_weights/'
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config=tf_config)
            print('Loading weights from:', UNIREP_WEIGHT_PATH)
        elif model_name == 'eunirep_1':
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH, config=tf_config)
            print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
        elif model_name == 'eunirep_2':
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH, config=tf_config)
            print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
        elif model_name == 'random_unirep':
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH, config=tf_config)
            print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
        elif model_name =='onehot':
            # Just need it to generate one-hot reps.
            # Top model created within OneHotRegressionModel doesn't actually get used.
            base_model = models.OneHotRegressionModel('EnsembledRidge')
        elif model_name == 'eunirep_petase': 
            UNIREP_WEIGHT_PATH = '/home/caiyi/unirep/eunirep_petase_21091902_30/'
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config=tf_config)
            print('Loading weights from:', UNIREP_WEIGHT_PATH)
        elif model_name == 'eunirep_favor': 
            UNIREP_WEIGHT_PATH = '/home/caiyi/unirep/eunirep_favor_21100101_23/'
            base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config=tf_config)
            print('Loading weights from:', UNIREP_WEIGHT_PATH)
        else:
            assert False, 'Unsupported base model'
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

    # if test_rep_src == 'load_by_seq':
    #     test_reps = load_rep_by_seq(test_rep_path, test_seqs)
    # elif test_rep_src == 'load_by_name':
    #     test_reps = load_rep_by_name(test_rep_path, test_names)
    if test_rep_src == 'generate':
        if 'ptsrep' in model_name:
            test_seqs_trimmed = []
            for seq in test_seqs:
                test_seqs_trimmed.append(seq[seq_start:seq_end])
            test_reps = pdb_and_seq_to_ptsrep(test_seqs_trimmed, model_path, pdb_path)
        elif 'unirep' in model_name or 'onehot' in model_name:
            test_reps = generate_reps(test_seqs)
    else:
        raise NameError('No such test_rep_src!')

    yhat, yhat_std, yhat_mem = top_model.predict(test_reps,
                return_std=True, return_member_predictions=True)
    return np.array(yhat)


def filter_and_select_top_seqs(res_file, nseq_select, burnin=250, max_sa_itr=None):
    print('SELECTION')
    res_file_name = os.path.basename(res_file)
        
    print(res_file)
    print('Loading results and converting SA histories to numpy arrays')
    res, res_sa = load_results(res_file)
    fit_mat = res_sa['fitness_history']

    init_fitness = fit_mat[0,:]

    print('Selecting top sequences')
    # First identify the best sequence in each SA trajectory.
    top_seqs, top_seq_fitness, _, top_seq_idx = get_best_sequence_in_each_trajectory(
            res_sa, burnin=burnin, max_sa_itr=max_sa_itr)
    

    # Now, select the top seqs of the best-in-trajectory sequences. 
    # These are are our official selections!
    # top_seq_idx is an index for each trajectory that says where in the trajectory the best sequence is. 
    sidx = np.argsort(-top_seq_fitness)
    top_sidx = sidx[:nseq_select]

    trajectory_indices_yielding_top_seqs = top_sidx
    seq_indices_inside_top_trajectories = top_seq_idx[top_sidx]
    selected_top_seqs = top_seqs[top_sidx] ## official selection
    selected_top_seq_fitness = top_seq_fitness[top_sidx] ## official selection
    selected_seqs_activity = predict_activity(top_seqs, 'eunirep_petase')[top_sidx]
    
    # selected_top_ensemble_fitness_preds = []
    # for i in range(len(trajectory_indices_yielding_top_seqs)):
    #     selected_top_ensemble_fitness_preds.append(
    #         res_sa['fitness_mem_pred_history'][seq_indices_inside_top_trajectories[i]][
    #             trajectory_indices_yielding_top_seqs[i]]
    #     )
        
    # Turn these selections into a dataframe
    id_prefix = res_file_name.replace('.p', '')
    fit_mat_idx = [str(s[0]) + '_' + str(s[1]) for s in list(zip(*[list(seq_indices_inside_top_trajectories), 
           list(trajectory_indices_yielding_top_seqs)]))]
    seq_ids = [id_prefix + '-seq_idx_' + fmi for fmi in fit_mat_idx]
    
    select_df = pd.DataFrame()
    select_df['id'] = seq_ids
    select_df['seq_idx'] = seq_indices_inside_top_trajectories # row idx of res_sa['fitness_history']
    select_df['trajectory_idx'] = trajectory_indices_yielding_top_seqs # col idx of res_sa['fitness_history']
    select_df['predicted_fitness'] = selected_top_seq_fitness
    # select_df['ensemble_predicted_fitness'] = selected_top_ensemble_fitness_preds
    select_df['seq'] = selected_top_seqs
    select_df['activity'] = selected_seqs_activity
    
    return select_df, res, res_sa
