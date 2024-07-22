import os
import sys
import warnings
import random
import copy
import pickle
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from low_n_utils import paths
from low_n_utils import models, A003_common
from low_n_utils import A006_common
from low_n_utils import sequence_selection
from low_n_utils.unirep import babbler1900 as babbler
from low_n_utils import data_io_utils
from low_n_utils import constants
from low_n_utils import utils


PROTEIN = 'GFP'
UNIREP_BATCH_SIZE = 3500
TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']
BURNIN = 250
MAX_SA_ITR = None # if None use all of them.
NSEQ_SELECT = 3000


def load_base_model(model_name):
    if model_name == 'ET_Global_Init_1':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    elif model_name == 'ET_Global_Init_2':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
    elif model_name == 'ET_Random_Init_1':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
    elif model_name == 'eUniRep_petase': 
        UNIREP_WEIGHT_PATH = '/home/caiyi/unirep/eunirep_petase_21091902_30/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    elif model_name =='OneHot':
        # Just need it to generate one-hot reps.
        # Top model created within OneHotRegressionModel doesn't actually get used.
        base_model = models.OneHotRegressionModel('EnsembledRidge') 
    else:
        assert False, 'Unsupported base model'

    return base_model


# Generate representations
def generate_reps(seq_list, base_model, sess):        
    if 'babbler1900' == base_model.__class__.__name__:
        assert len(seq_list) <= UNIREP_BATCH_SIZE
        hiddens = base_model.get_all_hiddens(seq_list, sess)
        rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)

    else: # one hot model
        rep = base_model.encode_seqs(seq_list)

    return rep

def load_results(res_file):
    with open(res_file, 'rb') as f:
        res = pickle.load(f)

    res_sa = sequence_selection.convert_result_vals_to_mat(res['sa_results'])
    return res, res_sa


def select_top_seqs(res_file, nseq_select, burnin=250, max_sa_itr=None):
    print('SELECTION')
    res_file_name = os.path.basename(res_file)

    print(res_file)
    print('Loading results and converting SA histories to numpy arrays')
    res, res_sa = load_results(res_file)
    print('res_sa:', res_sa.keys())
    '''debug'''
    # print(len(res_sa["seq_history"][250]))
    # with open('/home/wangqihan/Low_n_alphafold_test/design/design_seqs/eUniRep_pet_0_stability_b0_seq_history_1250.txt', 'a') as f1:
    #     for seq in res_sa["seq_history"][1250]:
    #         f1.write(str(seq) + '\n')
    print("init_seqs(raw_seq):", len(res_sa["init_seqs"]))
    fit_mat = res_sa['fitness_history']
    # print(res_sa['init_seqs'][0])
    # with open('/home/wangqihan/Low_n_alphafold_test/design/design_seqs/eUniRep_pet_0_stability_b0_100_raw_data.txt', 'a') as f1:
    #     for seq in res_sa['init_seqs']:
    #         f1.write(str(seq) + '\n')
    # with open('/home/wangqihan/low-N-protein-engineering-master/ptsrep/eUniRep_design_data_96.txt', 'a') as f2:
    #     for seq in res_sa['mu_muts_per_seq']:
    #         f2.write(str(seq) + '\n')
    # print(res_sa['mu_muts_per_seq'])

    init_fitness = fit_mat[0, :]

    print('Selecting top sequences')
    # First identify the best sequence in each SA trajectory.
    top_seqs, top_seq_fitness, _, top_seq_idx = sequence_selection.get_best_sequence_in_each_trajectory(
        res_sa, burnin=burnin, max_sa_itr=max_sa_itr)
    print("top_seqs:", len(top_seqs))
    # with open('/home/wangqihan/Low_n_alphafold_test/design/design_seqs/eUniRep_pet_0_stability_b0_100_design_data.txt', 'a') as f2:
    #     for seq in top_seqs:
    #         f2.write(str(seq) + '\n')
    print('top_1_seqs: ',top_seqs[0])
    print('top_seq_fitness: ',top_seq_fitness)
    print('top_seqs_idx: ',top_seq_idx[420], top_seq_idx[2497], top_seq_idx[2798])
    
    # Now, select the top seqs of the best-in-trajectory sequences.
    # These are are our official selections!
    # top_seq_idx is an index for each trajectory that says where in the trajectory the best sequence is.
    sidx = np.argsort(-top_seq_fitness)
    top_sidx = sidx[:nseq_select]

    trajectory_indices_yielding_top_seqs = top_sidx
    seq_indices_inside_top_trajectories = top_seq_idx[top_sidx]
    selected_top_seqs = top_seqs[top_sidx]  ## official selection
    selected_top_seq_fitness = top_seq_fitness[top_sidx]  ## official selection
    print(len(selected_top_seq_fitness))
    selected_top_ensemble_fitness_preds = []
    for i in range(len(trajectory_indices_yielding_top_seqs)):
        selected_top_ensemble_fitness_preds.append(
            res_sa['fitness_mem_pred_history'][seq_indices_inside_top_trajectories[i]][
                trajectory_indices_yielding_top_seqs[i]]
        )

    # Turn these selections into a dataframe
    id_prefix = res_file_name.replace('.p', '')
    fit_mat_idx = [str(s[0]) + '_' + str(s[1]) for s in list(zip(*[list(seq_indices_inside_top_trajectories),
                                                                   list(trajectory_indices_yielding_top_seqs)]))]
    seq_ids = [id_prefix + '-seq_idx_' + fmi for fmi in fit_mat_idx]

    select_df = pd.DataFrame()
    select_df['id'] = seq_ids
    select_df['seq_idx'] = seq_indices_inside_top_trajectories  # row idx of res_sa['fitness_history']
    select_df['trajectory_idx'] = trajectory_indices_yielding_top_seqs  # col idx of res_sa['fitness_history']
    select_df['predicted_fitness'] = selected_top_seq_fitness
    select_df['ensemble_predicted_fitness'] = selected_top_ensemble_fitness_preds
    select_df['seq'] = selected_top_seqs

    return select_df, res, res_sa


def validate_top_seqs(select_df, output_dir, res, res_sa, burnin=250, max_sa_itr=None):
    fit_mat = res_sa['fitness_history']
    seq_mat = res_sa['seq_history']

    print('VALIDATION')
    trajectory_indices_yielding_top_seqs = np.array(select_df['trajectory_idx'])
    seq_indices_inside_top_trajectories = np.array(select_df['seq_idx'])
    selected_top_seq_fitness = np.array(select_df['predicted_fitness'])
    top_seq_fitness_ensemble = np.stack(select_df['ensemble_predicted_fitness'])
    selected_top_seqs = np.array(select_df['seq'])

    # First check that after all the manipulation we did, that manually extracting the
    # sequence and its fitness based on the identified indices lines up with the what
    # the selection code provides.
    print('Validating sequence selection doing a manual re-extraction')
    for i in range(len(trajectory_indices_yielding_top_seqs)):
        man_sel_fitness_ens = res_sa['fitness_mem_pred_history'][
            seq_indices_inside_top_trajectories[i]][trajectory_indices_yielding_top_seqs[i]]
        man_sel_fitness = fit_mat[seq_indices_inside_top_trajectories[i],
                                  trajectory_indices_yielding_top_seqs[i]]
        man_sel_seq = seq_mat[seq_indices_inside_top_trajectories[i],
                              trajectory_indices_yielding_top_seqs[i]]

        assert man_sel_fitness == selected_top_seq_fitness[i]
        assert man_sel_seq == selected_top_seqs[i]

    # Re-calculate the ensemble's prediction and make sure it lines up with the extracted
    # fitness.
    recalc_pred_fitness = np.mean(np.stack(select_df['ensemble_predicted_fitness']), axis=1)
    assert np.allclose(recalc_pred_fitness - selected_top_seq_fitness, 0, atol=1e-5)

    # Validate these sequences are good. Score again with the original sparse refit
    # top model as well as the non-sparse refit model
    print('Rescoring sequences')
    print('\tLoading base model')
    tf.reset_default_graph()
    print("res_file:" ,res_file)
    base_model_name = res_file.split('-')[1]
    if base_model_name == 'LargeMut':
        base_model_name = res_file.split('-')[2]
    print('base_model_name',base_model_name)
    base_model = load_base_model(base_model_name)

    train_df = res['train_df']
    train_seqs = list(train_df['seq'])
    train_qfunc = np.array(train_df['quantitative_function'])

    # Generate reps for the sequences
    print('\tGenerating reps')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_reps = generate_reps(train_seqs, base_model, sess)
        top_seq_reps = generate_reps(list(selected_top_seqs), base_model, sess)
        wt_seq_rep = generate_reps([WT_SEQ], base_model, sess)

    # Train an NSR top model
    print('\tTraining an NSR top model')
    nsr_top_model = A003_common.train_ensembled_ridge(
        train_reps,
        train_qfunc,
        n_members=TOP_MODEL_ENSEMBLE_NMEMBERS,
        subspace_proportion=TOP_MODEL_SUBSPACE_PROPORTION,
        normalize=TOP_MODEL_NORMALIZE,
        do_sparse_refit=False,
        pval_cutoff=TOP_MODEL_PVAL_CUTOFF
    )

    sr_top_model = res['top_model']

    # Score WT seqs
    nsr_yhat_wt = nsr_top_model.predict(wt_seq_rep)
    print(nsr_yhat_wt)
    sr_yhat_wt = sr_top_model.predict(wt_seq_rep)
    print(sr_yhat_wt)

    # Score the the top sequences.
    nsr_yhat_top = nsr_top_model.predict(top_seq_reps)
    print(nsr_yhat_top)
    sr_yhat_top = sr_top_model.predict(top_seq_reps)
    print(sr_yhat_top)

    # First make sure that the freshly predicted fitness of the top seqs match the recorded ones.
    assert np.corrcoef(sr_yhat_top, selected_top_seq_fitness)[0, 1] > 0.99

    print('Generating validation plots')
    ## Now generate a bunch of plots
    # sr_vs_nsr_pred_plot(nsr_yhat_wt, sr_yhat_wt, nsr_yhat_top, sr_yhat_top, output_dir)
    # top_seq_and_traj_plot(fit_mat, trajectory_indices_yielding_top_seqs,
    #                       seq_indices_inside_top_trajectories, output_dir)

    # # all fitnesses for best-in-trajectory sequences
    # _, all_top_seq_fitness, _, _ = sequence_selection.get_best_sequence_in_each_trajectory(
    #     res_sa, burnin=burnin, max_sa_itr=max_sa_itr)
    # qfunc_hist_plot(sr_yhat_wt, fit_mat[0], all_top_seq_fitness, output_dir)

    # seq_dist_summary_plots(list(selected_top_seqs), selected_top_seq_fitness,
    #                        top_seq_fitness_ensemble, sr_yhat_wt, output_dir)

result_files = sorted(glob.glob('/home/wangqihan/Low_n_alphafold_test/design/220106_lin_pet_stability_ePtsRep_96_GA_0.p'))

# Special case globs
#result_files = sorted(glob.glob(os.path.join(data_dir, '*SparseRefit_False*.p')))
# print(os.path.join(data_dir, PROTEIN + '_SimAnneal*.p'))
print(result_files)

for res_file in result_files:
    output_dir = res_file.replace('.p', '-selected_seqs')
    output_file = os.path.join(output_dir, 'selected_seqs_df.pkl')

    if not os.path.exists(output_file):
        os.makedirs(output_dir, exist_ok=True)

        select_df, res, res_sa = select_top_seqs(res_file, NSEQ_SELECT, burnin=BURNIN, max_sa_itr=MAX_SA_ITR)
        print("select_df seq num:" ,len(list(select_df['seq'])))
        with open('/home/wangqihan/Low_n_alphafold_test/design/design_ptsrep_ga_seqs/ePtsRep_pet_0_stability_ga_top3000.txt', 'a') as f3:
            for seq in list(select_df['seq']):
                f3.write(str(seq) + '\n')
        # print(res)
        print(len(res_sa))
        validate_top_seqs(select_df, output_dir, res, res_sa, burnin=BURNIN, max_sa_itr=MAX_SA_ITR)
        select_df.to_pickle(output_file)
    else:
        print('Already done:', output_file)