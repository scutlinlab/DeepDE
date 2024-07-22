import os
import random
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import argparse
import pickle
import csv
from scipy import stats

from function import config 
from function import predict_common
import uni_ebd as ub
import net_MLP as net
from low_n_utils import paths
from low_n_utils import models, A003_common
# from low_n_utils import A006_common
from low_n_utils import A006_common
from low_n_utils import ga


TRAINING_SET_FILE = paths.SARKISYAN_SPLIT_1_FILE ## use split 1
# UNIREP_BATCH_SIZE = 400#3500
UNIREP_BATCH_SIZE = config.UNIREP_BATCH_SIZE
TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']
SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-n', '--n_train', default=None, type=int)
parser.add_argument('-e', '--exp_name', default=None)
parser.add_argument('-g', '--gpu', default=0)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-dt', '--do_test', default=False, type=bool)
parser.add_argument('-dd', '--do_design', default=False, type=bool)
parser.add_argument('-st', '--save_test_result', default=True, type=bool)
parser.add_argument('-dm', '--design_method', default="MCMC", type=str)
parser.add_argument('-dp', '--do_predict', default=False, type=bool)


args = parser.parse_args()

torch.set_num_threads(1)
gpu = config.PROGRAM_CONTROL["gpu"]
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
pts_model_path = config.PARAMETER['pts_model_path']
train_rep_src = config.PARAMETER['train_rep_src']
test_rep_src = config.PARAMETER['test_rep_src']
seq_start, seq_end = config.TARGETR_SEQ_INFORMATION['seq_start'], config.TARGETR_SEQ_INFORMATION['seq_end']
sampling_method = config.PARAMETER['sampling_method']
top_model_name = config.PARAMETER['top_model_name']
training_objectives = config.PARAMETER['training_objectives']
reference_seq = config.TARGETR_SEQ_INFORMATION["pet_wt_seq"]
n_chains = config.PARAMETER["n_chains"]
T_max = np.ones(config.PARAMETER["T_max_shape"],) * 0.01
sa_n_iter = config.PARAMETER["sa_n_iter"]
temp_decay_rate = config.PARAMETER["temp_decay_rate"]
max_mut_pos = config.TARGETR_SEQ_INFORMATION["max_pos"]
min_mut_pos = config.TARGETR_SEQ_INFORMATION["min_pos"]
nmut_threshold = config.PARAMETER["nmut_threshold"]

seed = config.PROGRAM_CONTROL["seed"]
do_test = config.PROGRAM_CONTROL["do_test"]
do_design = config.PROGRAM_CONTROL["do_design"]
design_method = config.PROGRAM_CONTROL["design_method"]
save_test_result = config.PROGRAM_CONTROL["save_test_result"]
do_predict = config.PROGRAM_CONTROL["do_predict"]
n_train_seqs = config.PARAMETER['n_train_seqs']
model_name = config.PARAMETER['model']
exp_name = config.PARAMETER['exp_name']
training_set_file = config.TRAIN_CSV_DICT[training_objectives]

init_seqs = A006_common.propose_seqs(
        [reference_seq]*n_chains, 
        [SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*n_chains, 
        min_pos=A006_common.GFP_LIB_REGION[0], 
        max_pos=A006_common.GFP_LIB_REGION[1])
mu_muts_per_seq = 1.5*np.random.rand(n_chains) + 1
print('mu_muts_per_seq:', mu_muts_per_seq) # debug

if 'stability' in training_objectives:
    print("pet_stability")
    fitness_name = 'stability'#选择是进行GFP任务还是稳定性任务
else:
    fitness_name = 'activity'


if "UniRep" in model_name:
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.15
    sess = tf.Session(config=tf_config)
    base_model = ub.select_basemodel(model_name, UNIREP_BATCH_SIZE, tf_config, sess)
    # sess = tf.Session()
    # saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
# saver.save(sess, '/home/caiyi/low_n/outputs/unirep.ckpt')

# Hard constants

# UNIREP_BATCH_SIZE = config['embed_batch_size']

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
train_df = pd.read_csv(training_set_file + str(seed + 1) + ".csv")
# train_df = pd.read_csv(training_set_file)


if sampling_method == 'random':
    print('n_train_seqs:', n_train_seqs)
    if 'stability' in training_objectives:
        sub_train_df = train_df[train_df['stability'].notnull()].sample(n = n_train_seqs)
    else:
        sub_train_df = train_df.sample(n=n_train_seqs)
elif sampling_method == 'all':
    if 'stability' in training_objectives:
        sub_train_df = train_df[train_df['stability'].notnull()]
        print(type(sub_train_df))
        print('train_num:' ,len(list(sub_train_df['name'])))
    else:
        sub_train_df = train_df
elif sampling_method == '50-50_1.0':
    sub_train_df1 = train_df[train_df[fitness_name] >= 1]
    sub_train_df2 = train_df[train_df[fitness_name] < 1].sample(n=len(sub_train_df1))
    sub_train_df = pd.concat([sub_train_df1, sub_train_df2])
elif sampling_method == "positive":
    if 'stability' in training_objectives:
        sub_train_df = train_df[train_df['stability'].notnull() >= 1]
        print(type(sub_train_df))
        print('train_num:' ,len(list(sub_train_df['name'])))
    else:
        sub_train_df = train_df[train_df[fitness_name] >= 1]
else:
    raise NameError('Wrong value of `sampling_method`')

print('train_df:')
print(sub_train_df.head())

if train_rep_src == 'load_by_name':
    train_names = list(sub_train_df['name'])
else:
    train_seqs = list(sub_train_df['seq'])
train_qfunc = np.array(sub_train_df[fitness_name])

if train_rep_src == 'load_by_seq' or train_rep_src == 'load_by_name':
    train_rep_path = config.CHOOSE_TRAIN_EBD[training_objectives][model_name]#PDB文件

if train_rep_src == 'load_by_seq' or test_rep_src == 'load_by_seq':
    with open(config.PARAMETER['seq2name_file']) as f:
        lines = f.readlines()
    d = {}
    for line in lines:
        line = line.split()
        # print(line)
        d[line[0]] = line[1]

if train_rep_src == 'load_by_seq':
    train_reps = predict_common.load_rep_by_seq(train_rep_path, train_seqs, model_name, top_model_name)
elif train_rep_src == 'load_by_name':
    train_reps = predict_common.load_rep_by_name(train_rep_path, train_names, model_name, top_model_name)
elif train_rep_src == 'generate':
    if 'PtsRep' in model_name:
        seq_to_ebd = ub.SeqEbdPtsRep(device, model_name)
        train_seqs_trimmed = []
        for seq in train_seqs:
            train_seqs_trimmed.append(seq[seq_start:seq_end])
        train_reps = seq_to_ebd.generate_reps(train_seqs_trimmed)
    elif 'UniRep' in model_name or 'onehot' in model_name:
        seq_to_ebd = ub.SeqEbdUniRep(sess, base_model)
        print('Setting up base model')
        print('Generating training seq reps')
        train_reps = seq_to_ebd.generate_reps(train_seqs)
    elif "alphafold" in model_name:
        print('Setting up base model')
        print('Generating training seq reps')

    else:
        raise NameError(f'Incorrect model name: {model_name}')

print('train_reps:', train_reps.shape)


# top_model = nn.train_mlp(train_reps, train_qfunc, config)

if top_model_name == 'lin':
    print('Building lin top model')
    top_model = A003_common.train_ensembled_ridge(
            train_reps, 
            train_qfunc, 
            n_members=TOP_MODEL_ENSEMBLE_NMEMBERS, 
            subspace_proportion=TOP_MODEL_SUBSPACE_PROPORTION,
            normalize=TOP_MODEL_NORMALIZE, 
            do_sparse_refit=True, 
            pval_cutoff=TOP_MODEL_PVAL_CUTOFF
        )
elif top_model_name == 'nn':
    print('Building MLP top model')
    if 'UniRep' in model_name or model_name == 'OneHot':
        top_model = net.train_mlp_uni(seed, train_reps, train_qfunc)
    else:
        # top_model = net.train_mlp(seed, train_reps, train_qfunc, config)
        top_model = net.train_mlp_Lengthen(seed, train_reps, train_qfunc)

train_info = {
    'top_model': top_model,
    'train_df': sub_train_df,
    'train_seq_reps': train_reps,
    'base_model': model_name
}

if 'PtsRep' in model_name:
    ebd_to_fitness = ub.SeqEbdPtsRep(device, model_name, top_model, reference_seq, nmut_threshold)
elif 'UniRep' in model_name or 'onehot' in model_name:
    ebd_to_fitness = ub.SeqEbdUniRep(sess, base_model, top_model, reference_seq, nmut_threshold)

output_dir = config.PARAMETER['output_dir']
if do_test:
    test_set_file = config.TEST_CSV_DICT[training_objectives]
    if test_rep_src == 'load_by_seq' or test_rep_src == 'load_by_name':
        test_rep_path = config.CHOOSE_TEST_EBD[training_objectives][model_name]
        print(test_rep_path)
    test_df = pd.read_csv(test_set_file + str(seed + 1) + ".csv")
    # test_df = pd.read_csv(test_set_file)
    n_test_seqs = len(test_df)
    print('test_df:')
    print(test_df.head())
    if test_rep_src == 'load_by_seq':
        test_seqs = list(test_df['seq'])
        test_reps = predict_common.load_rep_by_seq(test_rep_path, test_seqs, model_name, top_model_name)
        if top_model_name == 'nn':
            yhat = top_model.predict(test_reps)
        else:
            yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
    elif test_rep_src == 'load_by_name':
        test_names = list(test_df['name'])
        test_reps = predict_common.load_rep_by_name(test_rep_path, test_names, model_name, top_model_name)
        if top_model_name == 'nn':
            yhat = top_model.predict(test_reps)
        else:
            yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
    elif test_rep_src == 'generate':
        test_seqs = list(test_df['seq'])
        if 'PtsRep' in model_name:
            # test_seqs_trimmed = []
            # for seq in test_seqs:
            #     test_seqs_trimmed.append(seq[seq_start:seq_end])
            if top_model_name == 'nn':
                test_reps = ebd_to_fitness.generate_reps(test_seqs)
                yhat = top_model.predict(test_reps)
            else:
                yhat, yhat_std, yhat_mem = ebd_to_fitness.get_fitness(test_seqs)
            
        elif 'UniRep' in model_name or 'onehot' in model_name:
            if top_model_name == 'nn':
                test_reps = ebd_to_fitness.generate_reps(test_seqs)
                yhat = top_model.predict(test_reps)
            else:
                yhat, yhat_std, yhat_mem = ebd_to_fitness.get_fitness(test_seqs) 
    else:
        raise NameError('No such test_rep_src!')
    test_qfunc = np.array(test_df[fitness_name])
    # test_qfunc = np.array(test_df['activity'])
    # print('test_reps:', test_reps.shape)

    # if top_model_name == 'nn':
    #     yhat = top_model.predict(test_reps)
    # else:
    #     yhat, yhat_std, yhat_mem = ebd_to_fitness.get_fitness(test_reps)
    print(yhat)
    print(test_qfunc)
    r = stats.pearsonr(test_qfunc, yhat)[0]

    rho = stats.spearmanr(test_qfunc, yhat).correlation
    pred_and_real = np.array(list(yhat) + list(test_qfunc))
    name_list = []
    for i in range(len(test_qfunc)):
        name_list.append(i)
    
    if save_test_result:
        # output_dir = cd.PARAMETER['output_dir']
        output_file = f'{exp_name}_{fitness_name}_{model_name}_{n_train_seqs}_{n_test_seqs}_{seed}.p'
        # config['output_file'] = output_file
        save_json_path = f'{output_dir}/configs/{output_file[:-2]}.json'
        # if not os.path.exists(save_json_path):
        #     shutil.copy(args.config_path, save_json_path)
        output_data = np.array([pred_and_real, name_list], dtype=object)
        predict_common.create_dir_not_exist(f'{output_dir}/recall_file/{model_name}/')
        np.save(f'{output_dir}/recall_file/{model_name}/{output_file[:-2]}.npy', output_data)
        predict_common.create_dir_not_exist(f'{output_dir}/{model_name}/')
        with open(f'{output_dir}/{model_name}/results_acti_pair_{model_name}_{n_train_seqs}_{training_objectives}.txt', 'a') as f:
            f.write(f'{output_file[:-2]}\t{str(seed)}\t{round(r, 4)}\t{round(rho, 4)}\n')

if do_design:
    print("design_method:" ,design_method)
    if design_method == "MCMC":
        sa_results = A006_common.anneal(
                init_seqs, 
                k=SIM_ANNEAL_K, 
                T_max=T_max, 
                mu_muts_per_seq=mu_muts_per_seq,
                get_fitness_fn=ebd_to_fitness.get_fitness,
                n_iter=sa_n_iter, 
                decay_rate=temp_decay_rate,
                min_mut_pos=min_mut_pos,
                max_mut_pos=max_mut_pos)
    elif design_method == 'GA':
        sa_results = ga.anneal(
                init_seqs, 
                k=SIM_ANNEAL_K, 
                T_max=T_max, 
                mu_muts_per_seq=mu_muts_per_seq,
                get_fitness_fn=ebd_to_fitness.get_fitness,
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
    # output_file = os.path.basename(output_file)
    with open(f'{output_dir}/design/{exp_name}_{training_objectives}_{model_name}_{n_train_seqs}_{design_method}_{seed}.p', 'wb') as f:
        pickle.dump(file=f, obj=results)

if do_predict:
    predict_rep_path = config.DO_PREDICT["pre_path"]
    nseq_select = config.DO_PREDICT["nseq_select"]
    save_path = config.DO_PREDICT["save_path"]
    seq_path = config.DO_PREDICT["seq_path"]
    choose_top = config.DO_PREDICT["choose_top"]
    seq_list = []
    pre_name = seq_path.split("/")[-1].split(".")[0]
    if ".txt" in seq_path:
        with open(seq_path, "r") as f:
            seqs = f.readlines()
            for seq in seqs:
                seq_list.append(seq.split()[0])  
        # names = os.listdir(predict_rep_path)
        # pre_name_list = []
        # for name in names:
        #     pre_name_list.append(name.split(".")[0]) 
        pre_name_list = []
        for i in range(len(seq_list)):
            pre_name_list.append(str(i))
    elif ".fasta" in seq_path:
        with open(seq_path, "r") as f:
            lines = f.readlines()
            seq_list = []
            pre_name_list = []
            for i, line in enumerate(lines):
                if i % 2 != 0:
                    seq_list.append(line.split()[0])
                else:
                    pre_name_list.append(line[1: -1])

    print(len(seq_list), len(pre_name_list))
    # print(pre_name_list)
    rep_array = predict_common.load_rep_by_name(predict_rep_path, pre_name_list, model_name, top_model_name)
    print("rep_array: ", rep_array.shape)
    if top_model_name == 'nn':
        yhat = top_model.predict(rep_array)
    else:
        yhat, yhat_std, yhat_mem = top_model.predict(rep_array, return_std=True, return_member_predictions=True)
    print(yhat)
    if choose_top:
        sidx = np.argsort(-yhat)
        top_sidx = sidx[:nseq_select]
        print("debug:", top_sidx[0] == pre_name_list[top_sidx[0]])
        print(top_sidx[0], pre_name_list[top_sidx[0]])
        select_top_seqs = []
        select_top_hat = []
        for i in top_sidx:
            select_top_seqs.append(seq_list[i])
            select_top_hat.append(list(yhat)[i])
        csv_lists = []
        for i in range(len(top_sidx)):
            csv_lists.append([pre_name_list[top_sidx[i]], select_top_seqs[i], select_top_hat[i]])
        with open(f"{save_path}/{model_name}_{config.ALPHAFOLD2['target']}_{design_method}_{seed}_choose_top{str(nseq_select)}_{pre_name}.csv", 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["name", "seq", "yhat"])
            for csv_list in csv_lists:
                csv_writer.writerow(csv_list)
    else:
        csv_lists = []
        for i, name in enumerate(pre_name_list):
            csv_lists.append([name, seq_list[i], list(yhat)[i]])
        with open(f"{save_path}/{model_name}_{config .ALPHAFOLD2['target']}_{design_method}_{seed}_ALL_{pre_name}_GA.csv", 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["name", "seq", "yhat"])
            for csv_list in csv_lists:
                csv_writer.writerow(csv_list)

