import os
import shutil
import random
import numpy as np
import pandas as pd
import torch
import json
import tensorflow as tf
import argparse
from scipy import stats

import pts_ebd
import predict_common
import uni_ebd as ub
import net_MLP as net
from low_n_utils import paths
from low_n_utils import models, A003_common


TRAINING_SET_FILE = paths.SARKISYAN_SPLIT_1_FILE ## use split 1
# UNIREP_BATCH_SIZE = 400#3500
TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']
SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3

parser = argparse.ArgumentParser()
parser.add_argument('-cf', '--config_path')
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-n', '--n_train', default=None, type=int)
parser.add_argument('-e', '--exp_name', default=None)
parser.add_argument('-g', '--gpu', default=None)
parser.add_argument('-s', '--seed', default=None, type=int)

args = parser.parse_args()

with open(args.config_path) as f:
    config = json.load(f)

torch.set_num_threads(1)
gpu = args.gpu if args.gpu else config['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
model_path = config['model_path']
load_model_method = config['load_model_method']
pdb_path = config['pdb_path']
train_rep_path = config['train_rep_path']#PDB文件
test_rep_path = config['test_rep_path']
train_rep_src = config['train_rep_src']
test_rep_src = config['test_rep_src']
wt_seq = config['wt_seq']
seq_start, seq_end = config['seq_start'], config['seq_end']
# struct_seq_len = config['struct_seq_len']
# min_pos, max_pos = config['min_pos'], config['max_pos']
test_set_file = config['test_set_file']#13000测试集的CSV文件
fitness_name = config['fitness_name']#选择是进行GFP任务还是稳定性任务
test_fitness_name = config['test_fitness_name']
sampling_method = config['sampling_method']
# test_name = config['test_name']
top_model_name = config['top_model_name']
test_task = config["test_task"]

seed = args.seed if args.seed else config["seed"]
n_train_seqs = args.n_train if args.n_train else config['n_train_seqs']
model_name = args.model if args.model else config['model']
exp_name = args.exp_name if args.exp_name else config['exp_name']
training_set_file = config['training_set_file']
UNIREP_BATCH_SIZE = config['embed_batch_size']

ebd_path_dict = {'ePtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/evoptsrep_2e-3',
'PtsRep':'/share/joseph/seqtonpy/gfp/knn_self_512_full/self_20201216_4__self_20201215_5_sota_right_n2_knnnet150_del_tor',
'Random_PtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/random_ptsrep', 'eUniRep':'/home/wangqihan/Low-N_test_ebd/eUniRep',
'UniRep':'/home/wangqihan/Low-N_test_ebd/UniRep', 'Random_UniRep':'/home/wangqihan/Low-N_test_ebd/Random_UniRep'}
# 所有模型对应的50000数据集embedding，包含训练集和测试集。
test_ebd_path = {'eUniRep':'/home/wangqihan/Eunirep_ebd','UniRep':'/home/wangqihan/unirep_ebd',
'Random_UniRep':'/home/wangqihan/unirep_random_ebd','OneHot':'/home/wangqihan/onehot_ebd',
'ePtsRep':'/home/caiyi/embed/gfp/sa_ptsrep/','PtsRep':'/home/caiyi/embed/gfp/sa_ptsrep',
'Random_PtsRep':'/home/caiyi/embed/gfp/sa_random_ptsrep'}
# 所有模型的13000（Low-N设计的）测试集数据。
test_csv_dict = {'25000': '/share/joseph/seqtonpy/gfp/gfp.txt', '34536': "/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/sk_test_set_new.csv", 
'34536_split': "/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/sk_split_test_set.csv", 'SN': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_sn.txt',
'FP': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_fp.txt',
'SN_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_sn.txt',
'FP_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_fp.txt',
'LN_design': "/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/designed_test_set_clear.csv"}
test_ebd_path_SN = {'ePtsRep':'/share/jake/av_GFP_ebd/ePtsRep',
'PtsRep':'/share/jake/av_GFP_ebd/PtsRep','Random_PtsRep':'/share/jake/av_GFP_ebd/RPtsRep', 
'eUniRep':'/share/jake/av_GFP_ebd/SN/eUniRep','UniRep':'/share/jake/av_GFP_ebd/SN/UniRep',
'Random_UniRep':'/share/jake/av_GFP_ebd/SN/RUniRep'}
test_ebd_path_FP = {}
choose_test_ebd = {'25000': ebd_path_dict, '34536': ebd_path_dict, '34536_split': ebd_path_dict, 'SN': test_ebd_path_SN,
'FP': test_ebd_path_FP, 'LN_design': test_ebd_path, 'SN_split': test_ebd_path_SN,
'FP_split': test_ebd_path_FP}
test_list_dict = {'25000': "/share/joseph/seqtonpy/gfp/test/set_test.txt", '34536': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list.txt',
'34536_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/split_text_list.txt',
'SN_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list/split_text_list_sn.txt',
'FP_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list/split_text_list_fp.txt'}
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
    print('n_train_seqs:', n_train_seqs)
    if 'stability' in fitness_name:
        sub_train_df = train_df[train_df[fitness_name].notnull()].sample(n = n_train_seqs)
    elif fitness_name == 'activity' or 'quantitative_function':
        sub_train_df = train_df.sample(n=n_train_seqs)
elif sampling_method == 'all':
    if 'stability' in fitness_name:
        sub_train_df = train_df[train_df[fitness_name].notnull()]
    elif fitness_name == 'activity' or 'quantitative_function':
        sub_train_df = train_df
elif sampling_method == '50-50_1.0':
    sub_train_df1 = train_df[train_df[fitness_name] >= 1]
    sub_train_df2 = train_df[train_df[fitness_name] < 1].sample(n=len(sub_train_df1))
    sub_train_df = pd.concat([sub_train_df1, sub_train_df2])
else:
    raise NameError('Wrong value of `sampling_method`')

print('train_df:')
print(sub_train_df.head())

if train_rep_src == 'load_by_name':
    train_names = list(sub_train_df['name'])
else:
    train_seqs = list(sub_train_df['seq'])
train_qfunc = np.array(sub_train_df[fitness_name])

if train_rep_src == 'load_by_seq' or test_rep_src == 'load_by_seq':
    with open(config['seq2name_file']) as f:
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
        train_seqs_trimmed = []
        for seq in train_seqs:
            train_seqs_trimmed.append(seq[seq_start:seq_end])
        train_reps = pts_ebd.pdb_and_seq_to_ptsrep(train_seqs_trimmed, device, config)
    elif 'UniRep' in model_name or 'onehot' in model_name:
        print('Setting up base model')
        print('Generating training seq reps')
        train_reps = ub.seq_to_unirep(train_seqs, config, base_model, sess)
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
    if model_name == 'eUniRep' or model_name == 'UniRep' or model_name == 'Random_UniRep' or model_name == 'OneHot':
        top_model = net.train_mlp_uni(seed, train_reps, train_qfunc, config)
    else:
        # top_model = net.train_mlp(seed, train_reps, train_qfunc, config)
        top_model = net.train_mlp_Lengthen(seed, train_reps, train_qfunc, config)

train_info = {
    'top_model': top_model,
    'train_df': sub_train_df,
    'train_seq_reps': train_reps,
    'base_model': model_name
}

test_set_file = test_csv_dict[test_task]
test_rep_path = choose_test_ebd[test_task][model_name]
print(test_rep_path)
test_df = pd.read_csv(test_set_file)
n_test_seqs = len(test_df)
print('test_df:')
print(test_df.head())
if test_rep_src == 'load_by_seq':
    test_seqs = list(test_df['seq'])
    test_reps = predict_common.load_rep_by_seq(test_rep_path, test_seqs, model_name, top_model_name)
elif test_rep_src == 'load_by_name':
    test_names = list(test_df['name'])
    test_reps = predict_common.load_rep_by_name(test_rep_path, test_names, model_name, top_model_name)
elif test_rep_src == 'generate':
    test_seqs = list(test_df['seq'])
    if 'PtsRep' in model_name:
        test_seqs_trimmed = []
        for seq in test_seqs:
            test_seqs_trimmed.append(seq[seq_start:seq_end])
        test_reps = pts_ebd.pdb_and_seq_to_ptsrep(test_seqs_trimmed, device, config)
    elif 'UniRep' in model_name or 'onehot' in model_name:
        test_reps = ub.seq_to_unirep(test_seqs, config, base_model, sess)
else:
    raise NameError('No such test_rep_src!')
test_qfunc = np.array(test_df[test_fitness_name])
print('test_reps:', test_reps.shape)

if top_model_name == 'nn':
    yhat = top_model.predict(test_reps, config)
else:
    yhat, yhat_std, yhat_mem = top_model.predict(test_reps, return_std=True, return_member_predictions=True)

r = stats.pearsonr(test_qfunc, yhat)[0]

rho = stats.spearmanr(test_qfunc, yhat).correlation
pred_and_real = np.array(list(yhat) + list(test_qfunc))
name_list = []
for i in range(len(test_qfunc)):
    name_list.append(i)
output_file = f'{exp_name}_{fitness_name}_{model_name}_{n_train_seqs}_{n_test_seqs}_{seed}.p'
config['output_file'] = output_file
save_json_path = f'/home/wangqihan/Low_N_test_new/configs/{output_file[:-2]}.json'
if not os.path.exists(save_json_path):
    shutil.copy(args.config_path, save_json_path)
output_data = np.array([pred_and_real, name_list], dtype=object)
predict_common.create_dir_not_exist(f'/home/wangqihan/Low_N_test_new/recall_file/{model_name}/')
np.save(f'/home/wangqihan/Low_N_test_new/recall_file/{model_name}/{output_file[:-2]}.npy', output_data)
predict_common.create_dir_not_exist(f'/home/wangqihan/Low_N_test_new/{model_name}/')
with open(f'/home/wangqihan/Low_N_test_new/{model_name}/results_{model_name}_{n_train_seqs}_{test_task}.txt', 'a') as f:
    f.write(f'{output_file[:-2]}\t{str(seed)}\t{round(r, 4)}\t{round(rho, 4)}\n')
